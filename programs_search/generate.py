"""Generate exact candidate inputs from the variational search problem."""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'programs_search'

from .common import build_native, digest

PACKAGE = Path(__file__).resolve().parent

def validate_settings(s):
    """Reject malformed numerical settings before starting any computation."""
    required = {'reference_digits', 'fit_digits', 'maximum_support_terms', 'scale_levels',
                'state_degree', 'scale_search', 'transport_coarse', 'transport_fine',
                'primal', 'verification', 'price_accuracy_factor'}
    if set(s) != required:
        raise ValueError('Incomplete or unknown search settings.')
    if not all(isinstance(s[k], int) and 6 <= s[k] <= 13 for k in ('reference_digits', 'fit_digits')):
        raise ValueError('Reference and fit precision must be 6..13 decimal digits.')
    if min(s['maximum_support_terms'], s['state_degree'], s['price_accuracy_factor']) < 1 or s['scale_levels'] < 2:
        raise ValueError('Invalid search budget.')
    for name in ('scale_search', 'transport_coarse', 'transport_fine'):
        row = s[name]
        if set(row) != {'order', 'grid', 'degree', 'iterations'} or min(row.values()) < 1:
            raise ValueError('Invalid ' + name + ' settings.')
        if min(row['order'], row['grid']) < 2:
            raise ValueError('Quadrature and state grid need at least two points.')
    primal = s['primal']
    if primal['degree'] < 5 or primal['order'] <= primal['degree'] or primal['iterations'] < 1:
        raise ValueError('Insufficient primal dimension or numerical work budget.')
    if Fraction(primal['radius_goal']) <= 0:
        raise ValueError('The proposed radius goal must be positive.')
    v = s['verification']
    if set(v) != {'reference', 'second', 'primal_second', 'sine'}:
        raise ValueError('Incomplete verifier work configuration.')
    for name in ('reference', 'second', 'primal_second'):
        row = v[name]
        if row['bits'] < 64 or row['order'] < 2:
            raise ValueError('Insufficient interval precision/order.')
        if any(Fraction(row[k]) <= 0 for k in ('flow_tolerance', 'quadrature_tolerance')):
            raise ValueError('Tolerances must be positive.')
    if v['sine']['bits'] < 64 or Fraction(v['sine']['absolute_error']) <= 0:
        raise ValueError('Invalid sine precision.')


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PACKAGE.parent/'generated-inputs')
    parser.add_argument('--work', type=Path, default=PACKAGE.parent/'search-work')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--settings', type=Path, default=PACKAGE / 'settings.json',
                        help='numerical work parameters; contains no candidate data')
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    output, work = args.output.resolve(), args.work.resolve()
    if output == work or output in work.parents or work in output.parents:
        parser.error('--output and --work must be separate directories')
    for path in (output, work):
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            parser.error(str(path) + ' must be absent or empty; existing data are never overwritten')
    settings = json.loads(args.settings.read_text())
    validate_settings(settings)
    import numpy
    import scipy
    work.mkdir(parents=True, exist_ok=True)
    logs = work / 'logs'
    logs.mkdir()
    candidate = work / 'candidate_inputs'
    candidate.mkdir()
    os.environ['SINEJACOBI_SEARCH_WORK'] = str(work)
    os.environ['SINEJACOBI_WORKERS'] = str(args.jobs)
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['PYTHONUNBUFFERED'] = '1'
    started = time.monotonic()
    report = {'status': 'RUNNING', 'output': str(output), 'work': str(work),
              'settings': settings, 'jobs': args.jobs,
              'environment': {'python': sys.version, 'platform': platform.platform(),
                              'numpy': numpy.__version__, 'scipy': scipy.__version__,
                              'jobs_requested': args.jobs,
                              'thread_environment': {k: os.environ.get(k) for k in
                                  ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                                   'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')}},
              'source_sha256': {str(p.relative_to(PACKAGE)): digest(p)
                                for p in sorted(PACKAGE.rglob('*')) if p.suffix in ('.py', '.cpp', '.hpp')},
              'shared_native_sha256': {p.name: digest(p) for p in sorted((PACKAGE.parent/'programs/native').glob('*.hpp'))},
              'steps': []}

    def save():
        report['elapsed_seconds'] = round(time.monotonic() - started, 3)
        write_json(work / 'generation.json', report)

    def run(name, command, env=None):
        row = {'name': name, 'command': list(map(str, command)), 'status': 'RUNNING'}
        report['steps'].append(row)
        save()
        print('Running:', name, flush=True)
        step_start = time.monotonic()
        with (logs / (name + '.log')).open('w') as log:
            process = subprocess.run(list(map(str, command)), cwd=PACKAGE.parent,
                                     env=env, stdout=log, stderr=subprocess.STDOUT)
        row.update(status='COMPLETED' if process.returncode == 0 else 'FAILED',
                   exit_code=process.returncode, seconds=round(time.monotonic() - step_start, 3))
        save()
        if process.returncode:
            raise RuntimeError(name + ' failed; see ' + str(logs / (name + '.log')))

    def module(name, *arguments, label=None):
        run(label or name, [sys.executable, '-B', '-m', 'programs_search.' + name, *arguments])

    try:
        save()
        stationary, reference = work / 'stationary', work / 'reference'
        module('search_reference', '--work', stationary, '--digits', settings['reference_digits'])
        module('fit_reference', '--stationary', stationary, '--work', reference,
               '--digits', settings['fit_digits'], '--maximum-terms', settings['maximum_support_terms'])
        scale_search = work / 'scale-search'
        module('search_transport', '--reference', reference / 'reference.json', '--work', scale_search,
               '--order', settings['scale_search']['order'], '--grid', settings['scale_search']['grid'],
               '--degree', settings['scale_search']['degree'], '--state-degree', settings['state_degree'],
               '--scale-levels', settings['scale_levels'], '--refine-scales',
               '--iterations', settings['scale_search']['iterations'], '--without-second-moment',
               label='search_scales')
        scales = json.loads((scale_search / 'transport.json').read_text())['scales']
        from .model import write_native_parameters
        from .certificate_format import write_transport, verify_transport_polynomial
        write_native_parameters(json.loads((reference / 'reference.json').read_text()), scales,
                                candidate / 'reference.txt')
        previous = scale_search
        for second in (False, True):
            stage = work / ('second' if second else 'transport')
            flags = [] if second else ['--without-second-moment']
            common = ['--reference', reference / 'reference.json', '--scales', *scales, '--state-degree', settings['state_degree']]
            module('search_transport', *common, '--work', stage / 'coarse', '--order', settings['transport_coarse']['order'],
                   '--grid', settings['transport_coarse']['grid'], '--degree', settings['transport_coarse']['degree'],
                   '--iterations', settings['transport_coarse']['iterations'],
                   '--start', previous / 'transport.json', *flags, label=stage.name + '_coarse')
            module('search_transport', *common, '--work', stage, '--order', settings['transport_fine']['order'],
                   '--grid', settings['transport_fine']['grid'], '--degree', settings['transport_fine']['degree'],
                   '--iterations', settings['transport_fine']['iterations'],
                   '--start', stage / 'coarse/transport.json', '--start', previous / 'transport.json',
                   *flags, label=stage.name + '_fine')
            previous = stage
        write_transport(previous / 'transport.json', candidate / 'transport.txt')
        verify_transport_polynomial(candidate / 'transport.txt')
        previous = None
        for second in (False, True):
            stage = work / ('primal_second' if second else 'primal')
            flags = ['--second', '--start', previous / 'primal.json'] if second else []
            module('search_primal', '--reference', reference / 'reference.json', '--work', stage,
                   '--scales', *scales, '--degree', settings['primal']['degree'],
                   '--order', settings['primal']['order'], '--iterations', settings['primal']['iterations'],
                   '--bounded', *flags, label=stage.name + '_search')
            module('format_primal', stage / 'primal.json', stage / 'parameters.txt', label=stage.name + '_format')
            env = os.environ.copy()
            env['SINEJACOBI_TRACE'] = str(stage / 'native_trace.txt')
            env['SINEJACOBI_TRACE_MODE'] = 'summary'
            numerical = settings['verification']['primal_second']
            run(stage.name + '_dyadic', [build_native(), stage / 'parameters.txt',
                                       numerical['bits'], numerical['order'], numerical['flow_tolerance'],
                                       numerical['quadrature_tolerance'], settings['primal']['radius_goal'], stage], env=env)
            previous = stage
        for name in ('profile.txt', 'radius.txt'):
            shutil.copy2(previous / name, candidate / name)
        # Work tolerances and budgets are explicit numerical method settings.
        # The price work target is scaled to the increment actually found.
        proposal = json.loads((work / 'second/transport.json').read_text())
        increment = float(proposal['sampled_increment'])
        import math
        if not math.isfinite(increment) or increment <= 0:
            raise RuntimeError('The search found no positive dual increment.')
        run_configuration = {'format': 'SINEJACOBI_RUN_1', **settings['verification']}
        run_configuration['second'] = dict(run_configuration['second'])
        exponent = math.floor(math.log2(increment / settings['price_accuracy_factor']))
        from decimal import Decimal, localcontext
        with localcontext() as context:
            context.prec = max(64, abs(exponent) + 1)
            run_configuration['second']['price_error'] = str(Decimal(2) ** exponent)
        write_json(candidate / 'run.json', run_configuration)
        write_json(candidate / 'provenance.json', {
            'format': 'SINEJACOBI_SEARCH_ORIGIN_1',
            'sha256': {p.name: digest(p) for p in sorted(candidate.iterdir())},
            'environment': report['environment'], 'settings': settings})
        output.parent.mkdir(parents=True, exist_ok=True)
        # Build next to the destination, then publish the complete directory.
        import tempfile
        publish = Path(tempfile.mkdtemp(prefix='.inputs-search-', dir=output.parent))
        try:
            for file in candidate.iterdir():
                shutil.copy2(file, publish / file.name)
            if output.exists():
                output.rmdir()  # fails if another process has added anything
            publish.rename(output)
        finally:
            if publish.exists():
                shutil.rmtree(publish)
        report['native_build'] = json.loads((work/'native-cache/build.json').read_text())
        report['status'] = 'CANDIDATES_GENERATED'
        report['scope'] = 'Search generation only; run the separate full certificate verifier.'
        save()
        print(report['status'], output, flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        report['status'] = 'FAILED'
        report['error'] = str(exc) or type(exc).__name__
        save()
        print(report['status'] + ': ' + str(exc), file=sys.stderr, flush=True)
        print('No inputs published. Details: ' + str(work / 'generation.json'), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
