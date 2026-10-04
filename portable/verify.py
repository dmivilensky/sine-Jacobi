"""Standalone numerical proof checker; never imports a producer module."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import time

from audit.exact import load_json, need, records, encloses
from audit import checks
from audit.verify import texts, verify
from portable.cover import digest, scan
from portable.environment import runtime

FLAGS = ['-std=c++17', '-O3', '-ffp-contract=off', '-fno-fast-math']
ENTRIES = ('verify_reference', 'verify_transport', 'verify_primal', 'verify_sine')


def safe_name(name):
    p = PurePosixPath(name)
    need(isinstance(name, str) and str(p) == name and not p.is_absolute() and
         '..' not in p.parts and '\\' not in name and name not in ('', '.'), 'unsafe proof path')
    return p


def manifest_check(root):
    manifest = load_json((root/'manifest.json').read_text())
    need(manifest.get('format') == 'SINEJACOBI_PROOF_3', 'unsupported full-proof format')
    need(isinstance(manifest.get('members'), dict), 'missing proof member table')
    for name, entry in manifest['members'].items():
        safe_name(name)
        p = root/name
        need(not p.is_symlink() and p.is_file() and p.resolve().is_relative_to(root.resolve()),
             'missing or unsafe proof member: ' + name)
        need(p.stat().st_size == entry['bytes'] and digest(p) == entry['sha256'],
             'proof member digest mismatch: ' + name)
    required = {'claims.json', 'run.json', 'verify.py', 'portable/verify.py', 'portable/cover.py',
                'data/second/state-cover.bin', 'portable/environment.py', 'native/probe.cpp',
                'audit/exact.py', 'audit/checks.py', 'audit/verify.py', 'audit/report.py'}
    required.update('native/'+n+'.cpp' for n in ENTRIES)
    need(required <= set(manifest['members']), 'incomplete standalone proof')
    # Do not execute unmanifested local modules or headers.
    for folder, suffix in (('native', '.hpp'), ('native', '.cpp'), ('audit', '.py'), ('portable', '.py')):
        for p in (root/folder).glob('*'+suffix):
            need(str(p.relative_to(root)) in manifest['members'], 'unbound checker source')
    return manifest


def compare_intervals(expected, actual, header, label, upper_only=()):
    e, a = records(expected, header), records(actual, header)
    need(set(e) == set(a), label + ': report field mismatch')
    for key in e:
        valid = a[key][1] <= e[key][1] if key in upper_only else encloses(e[key], a[key])
        need(valid, label + ': recomputation not enclosed by claimed ' + key)
    return len(e)


def compare_transport(expected, actual):
    e, a = checks.transport_report(expected, True), checks.transport_report(actual, True)
    for key in ('domain', 'dilation', 'cells', 'leaves'):
        need(e[key] == a[key], 'replayed transport ' + key + ' mismatch')
    for key in ('price', 'undilated_price'):
        need(encloses(e[key], a[key]), 'claimed transport interval does not enclose replay: ' + key)


def native_build(root, work, names=ENTRIES):
    compiler = shutil.which(os.environ.get('CXX', 'c++'))
    need(compiler is not None, 'C++17 compiler is required')
    includes = ['-I'+str(root/'native')]
    libraries = ['-lmpfr', '-lgmp']
    prefix = os.environ.get('SINEJACOBI_NATIVE_PREFIX')
    if prefix:
        prefix = Path(prefix).resolve()
        includes.append('-I'+str(prefix/'include'))
        libraries = [str(prefix/'lib/libmpfr.a'), str(prefix/'lib/libgmp.a')]
    binaries = {}
    build_started = time.monotonic()
    names = tuple(dict.fromkeys(('probe', *names)))
    for name in names:
        out = work/name
        process([compiler, *FLAGS, *includes, root/'native'/(name+'.cpp'), *libraries, '-o', out], work, name+'-build')
        binaries[name] = out
    version = subprocess.check_output([compiler, '--version'], text=True).splitlines()[0]
    build_seconds = round(time.monotonic()-build_started, 3)
    process([binaries['probe']], work, 'runtime-probe')
    probe = (work/'runtime-probe.log').read_text().strip()
    match = re.search(r'MPFR (\S+) GMP (\S+) binary64_digits (\d+) C\+\+ (\d+)', probe)
    need(match is not None, 'runtime probe did not report library versions')
    return binaries, {'compiler': version, 'compiler_sha256': digest(Path(compiler).resolve()),
                      'flags': FLAGS, 'mpfr': match[1], 'gmp': match[2],
                      'binary64_digits': int(match[3]), 'cplusplus': int(match[4]),
                      'compilation_seconds': build_seconds}


def process(args, work, name):
    """Keep a bounded diagnostic tail; no per-evaluation logs escape the workdir."""
    env = os.environ.copy()
    env.update(SINEJACOBI_TRACE=str(work/(name+'.trace')), SINEJACOBI_TRACE_MODE='summary', LC_ALL='C')
    start = time.monotonic()
    tail = bytearray()
    with subprocess.Popen(list(map(str, args)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env) as child:
        for block in iter(lambda: child.stdout.read(8192), b''):
            tail.extend(block)
            if len(tail) > 65536: del tail[:-65536]
        code = child.wait()
    (work/(name+'.log')).write_bytes(tail)
    need(code == 0, name + ' rejected (exit '+str(code)+'):\n' + tail.decode(errors='replace')[-6000:])
    return {'seconds': round(time.monotonic()-start, 3)}


def check(root, work, jobs=1):
    need(jobs >= 1, 'jobs must be positive')
    manifest_check(root)
    bundle = load_json((root/'claims.json').read_text())
    need(not bundle.get('test_fixture'), 'synthetic fixtures cannot receive full-proof acceptance')
    files = texts(bundle)
    config = load_json((root/'run.json').read_text())
    need(config.get('format') == 'SINEJACOBI_RUN_1', 'run configuration format')
    data = work/'data'
    shutil.copytree(root/'data', data)
    for name, text in files.items():
        safe_name(name)
        target = data/name
        need(not target.exists(), 'embedded report collides with a proof witness')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    for name, entry in bundle['artifacts'].items():
        safe_name(name)
        need(digest(data/name) == entry['sha256'] and (data/name).stat().st_size == entry['bytes'],
             'witness identity mismatch: ' + name)
    geometry = scan(data/'second/state-cover.bin', config['second']['bits'])
    geometry['artifact'] = 'second/state-cover.bin'
    bundle['cover_geometry'] = geometry
    # Conditional arithmetic is evaluated once; no PASS_FULL_NUMERICS yet.
    result, rows, parameters = verify(bundle)
    binaries, toolchain = native_build(root, work)
    elapsed = {}

    def reference():
        c = config['reference']; out = work/'reference'; out.mkdir()
        elapsed['reference'] = process([binaries['verify_reference'], data/'reference/reference.txt',
            c['bits'], c['order'], c['flow_tolerance'], c['quadrature_tolerance'], out,
            data/'reference/support-filter-flow.txt'], work, 'reference')
        return compare_intervals(files['reference/reference-values.txt'], (out/'reference-values.txt').read_text(),
                                 'CONTACT_VALUES 1', 'reference')

    def primal():
        c = config['primal_second']; out = work/'primal-values.txt'
        elapsed['primal'] = process([binaries['verify_primal'], data/'primal_second', c['bits'], c['order'],
            c['flow_tolerance'], c['quadrature_tolerance'], out], work, 'primal')
        return compare_intervals(files['primal_second/values.txt'], out.read_text(), 'PRIMAL_VALUES 1',
                                 'primal', ('residual', 'contraction'))

    def sine():
        c = config['sine']; out = work/'sine'; out.mkdir()
        elapsed['sine'] = process([binaries['verify_sine'], c['bits'], c['absolute_error'], out], work, 'sine')
        compare_intervals(files['sine/sine-values.txt'], (out/'sine-values.txt').read_text(), 'SINE_VALUES 1',
                          'sine', ('tail', 'gain_upper'))
        checks.sine((out/'sine-polynomial.txt').read_text(), records((out/'sine-values.txt').read_text(), 'SINE_VALUES 1'))

    print('Rechecking reference flows, whole-cube Jacobian, and sine series.', flush=True)
    with ThreadPoolExecutor(max_workers=min(jobs, 3)) as pool:
        futures = [pool.submit(fn) for fn in (reference, primal, sine)]
        counts = [future.result() for future in futures]
    parts = sorted(name for name in files if re.fullmatch(r'second/price-claims/[^/]+/values.txt', name))
    # The conditional audit already established the exact ordered union of these domains.
    def region(item):
        index, name = item
        p = checks.transport_report(files[name], True)
        half, left, right = p['domain']
        need(left == float(left) and right == float(right), 'nonbinary verification boundary')
        c = config['second']; out = work/('transport-'+str(index)+'.txt')
        timing = process([binaries['verify_transport'], data/'reference/reference.txt', data/'second/transport.txt',
            data/'reference', c['bits'], c['order'], c['flow_tolerance'], c['quadrature_tolerance'],
            data/'second/state-cover.bin', out, half, float(left).hex(), float(right).hex()], work, 'transport-'+str(index))
        compare_transport(files[name], out.read_text())
        print('Verified state region', index+1, '/', len(parts), flush=True)
        return timing
    print('Rechecking every state leaf and every price integral.', flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        elapsed['transport_regions'] = list(pool.map(region, enumerate(parts)))
    # Only here, after all mathematical acceptance readers have succeeded,
    # may assumed native premises be promoted to re-established premises.
    from audit.report import numerical_results
    return {
        'format': 'SINEJACOBI_VERIFICATION_2', 'status': 'PASS_FULL_NUMERICS',
        'scope': 'All saved ODE/Taylor witnesses, quadratures, whole-cube Jacobian, complete state cover, regional prices, sine series and rational consequences rechecked.',
        'numerical_results': numerical_results(result, rows, parameters),
        'trust_basis': 'Bundled acceptance source, analytical lemmas, C++/Python runtime, GMP/MPFR and checked IEEE binary64; native kernels are shared with construction.',
        'verified_report_fields': {'reference': counts[0], 'primal': counts[1]},
        'native_timings': elapsed, 'toolchain': toolchain,
        'environment': runtime(jobs), 'manifest_sha256': digest(root/'manifest.json')}



def main(argv=None, default_root=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=default_root or Path(__file__).resolve().parents[1])
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--output', type=Path, default=Path('verification.json'))
    ap.add_argument('--keep-work', action='store_true')
    a = ap.parse_args(argv)
    ap.error('--jobs must be positive') if a.jobs < 1 else None
    a.root = a.root.resolve()
    protected = {p.resolve() for p in a.root.rglob('*') if p.is_file() and
                 (p.suffix in ('.py','.cpp','.hpp') or p.name in ('claims.json','manifest.json','run.json'))}
    if a.output.resolve().is_relative_to(a.root/'data') or a.output.resolve() in protected:
        print('FAIL: report would overwrite proof input', file=sys.stderr)
        return 1
    if a.output.suffix != '.json':
        print('FAIL: output must be a JSON report', file=sys.stderr)
        return 1
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.output.with_name(a.output.name+'.new')
    temporary.write_text(json.dumps({'status':'INCOMPLETE', 'scope':'Verification has not completed.'})+'\n')
    temporary.replace(a.output)
    work = Path(tempfile.mkdtemp(prefix='sinejacobi-verify-'))
    result = {'status': 'FAIL', 'scope': 'No full-proof acceptance.'}
    start = time.monotonic()
    code = 1
    try:
        result = check(a.root, work, a.jobs)
        code = 0
    except (Exception, KeyboardInterrupt) as exc:
        result['error'] = str(exc) or type(exc).__name__
        code = 1
    finally:
        result['elapsed_seconds'] = round(time.monotonic()-start, 3)
        if a.keep_work or code != 0: result['work_directory'] = str(work)
        else: shutil.rmtree(work)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.output.with_name(a.output.name+'.new')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, default=str)+'\n')
    temporary.replace(a.output)
    print(result['status'], str(a.output), flush=True)
    if code: print(result['error'], file=sys.stderr)
    return code
