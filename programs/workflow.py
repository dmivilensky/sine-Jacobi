"""One resumable workflow: optional search, witness construction, full check."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .common import ROOT, digest, json_write
from .proof import export, check_archive
from audit.exact import need, load_json
from portable.environment import runtime

INPUT_FILES = ('reference.txt', 'transport.txt', 'profile.txt', 'radius.txt', 'run.json')


def source_hashes():
    names = [ROOT/'compute.py', ROOT/'verify_proof.py']
    for directory in ('programs', 'programs_search', 'audit', 'portable'):
        names += [p for p in (ROOT/directory).rglob('*')
                  if p.suffix in ('.py', '.cpp', '.hpp', '.json') or p.name == 'requirements.txt']
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(names)}


def input_hashes(directory):
    directory = Path(directory)
    for name in INPUT_FILES:
        need((directory/name).is_file(), 'missing exact input: '+str(directory/name))
    return {name: digest(directory/name) for name in INPUT_FILES}


def snapshot_inputs(source, destination):
    hashes = input_hashes(source)
    destination.mkdir()
    for name in INPUT_FILES:
        shutil.copyfile(source/name, destination/name)
    if (source/'provenance.json').is_file():
        provenance = load_json((source/'provenance.json').read_text())
        need(all(provenance.get('sha256', {}).get(name) == value for name, value in hashes.items()),
             'input provenance does not match the selected files')
        shutil.copyfile(source/'provenance.json', destination/'provenance.json')
    need(input_hashes(destination) == hashes == input_hashes(source), 'inputs changed while copying')
    return hashes


def write_summary(report_path, target):
    report = load_json(report_path.read_text())
    need(report['status'] == 'PASS_FULL_NUMERICS', 'only full acceptance may publish results.txt')
    n = report['numerical_results']
    lines = ['PASS_FULL_NUMERICS', 'Archive SHA-256: '+report['archive_sha256'],
             'All decimal interval endpoints below are rounded outwards.', '']
    for group, key in (('certificate','bound'), ('certificate','price'),
                       ('reference','base'), ('feasible','energy_ceiling'),
                       ('feasible','residual'), ('feasible','contraction')):
        a = n['intervals'][group][key]['decimal_outward']
        lines.append(group+'/'+key+': ['+', '.join(a)+']')
    lines.append('sine/energy: ['+', '.join(n['sine_recomputed']['energy']['decimal_outward'])+']')
    for key in ('gain_ratio','feasible_improvement_lower','primal_dual_gap_upper'):
        lines.append(key+': ['+', '.join(n['derived'][key]['decimal_outward'])+']')
    lines += ['', 'Cells: '+str(n['counts']['cells']), 'State leaves: '+str(n['counts']['leaves']),
              'Dilation: '+n['cover_geometry']['dilation'],
              'Worst recorded upper bound: '+n['cover_geometry']['max_recorded_upper'],
              'Strict feasible improvement: '+str(n['strict_feasible_improvement']),
              '', 'Exact endpoints, all coefficients, residuals and Jacobian entries: verification.json.',
              'Run environment and phase timings: run.json and verification.json.']
    target.write_text('\n'.join(lines)+'\n')


def run(destination, *, jobs=8, inputs=None, settings=None, resume=False, keep_work=False):
    destination = Path(destination).resolve()
    inputs = Path(inputs).resolve() if inputs else None
    settings = Path(settings or ROOT/'programs_search/settings.json').resolve()
    need(jobs > 0, 'jobs must be positive')
    need(destination != ROOT and not ROOT.is_relative_to(destination), 'output contains package sources')
    for directory in ('inputs','programs','programs_search','audit','portable','tests'):
        need(not destination.is_relative_to(ROOT/directory), 'output overlaps package inputs/sources')
    if inputs:
        need(not inputs.is_relative_to(destination) and not destination.is_relative_to(inputs),
             'selected inputs and output must be separate directories')
    need(not settings.is_relative_to(destination), 'settings must be outside the output directory')
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination/'run.json'
    work = destination/'work'
    local_inputs = destination/'inputs'
    archive = destination/'proof.tar.xz'
    report_path = destination/'verification.json'
    # Lock one output directory for the entire invocation, including resume.
    with (destination/'.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('this output directory is already in use') from exc
        requested = {'mode': 'fixed-inputs' if inputs else 'search',
                     'source_sha256': source_hashes(),
                     'input_sha256': input_hashes(inputs) if inputs else None,
                     'settings_sha256': digest(settings) if inputs is None else None}
        if manifest_path.exists():
            need(resume, 'output already exists; use --resume, verify, or a new --output')
            state = load_json(manifest_path.read_text())
            need(state.get('format') == 'SINEJACOBI_RUN_RECORD_1' and state['request'] == requested,
                 'resume requires unchanged sources, settings and selected inputs')
            need(state.get('status') != 'COMPLETED', 'run already completed; use verify for a fresh replay')
        else:
            need(not resume, '--resume requires an existing run.json')
            need(not any(p.name != '.run.lock' for p in destination.iterdir()), 'new output must be empty')
            state = {'format': 'SINEJACOBI_RUN_RECORD_1', 'request': requested,
                     'created_utc': datetime.now(timezone.utc).isoformat(), 'phases': [], 'inputs_ready': False}
        work.mkdir(exist_ok=True)
        state.pop('error', None)
        state.pop('work_directory', None)
        state.update(status='RUNNING', environment=runtime(jobs))
        json_write(manifest_path, state)
        json_write(report_path, {'status': 'INCOMPLETE', 'scope': 'No full numerical acceptance for this invocation.'})
        (destination/'results.txt').unlink(missing_ok=True)
        start = time.monotonic()

        @contextmanager
        def phase(name):
            row = {'name': name, 'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'RUNNING'}
            state['phases'].append(row)
            json_write(manifest_path, state)
            print('PHASE:', name, flush=True)
            began = time.monotonic()
            try:
                yield
                row['status'] = 'COMPLETED'
            except BaseException:
                row['status'] = 'FAILED'
                raise
            finally:
                row['seconds'] = round(time.monotonic()-began, 3)
                json_write(manifest_path, state)

        try:
            if not state['inputs_ready']:
                # Only these owned, incomplete paths may be replaced on resume.
                if local_inputs.exists(): shutil.rmtree(local_inputs)
                search_work = work/'search'
                if search_work.exists(): shutil.rmtree(search_work)
                with phase('search' if inputs is None else 'snapshot-inputs'):
                    if inputs is None:
                        args = [sys.executable, '-B', '-m', 'programs_search.generate', '--output', local_inputs,
                                '--work', search_work, '--settings', settings, '--jobs', jobs]
                        subprocess.run(list(map(str, args)), cwd=ROOT, check=True)
                        state['generation'] = load_json((search_work/'generation.json').read_text())
                    else:
                        snapshot_inputs(inputs, local_inputs)
                    state['exact_input_sha256'] = input_hashes(local_inputs)
                    state['inputs_ready'] = True
            need(input_hashes(local_inputs) == state['exact_input_sha256'], 'snapshot inputs changed')
            env = os.environ.copy()
            env.update(SINEJACOBI_INPUTS=str(local_inputs), SINEJACOBI_RESULTS=str(work/'build'),
                       SINEJACOBI_NATIVE_CACHE=str(work/'native-cache'), SINEJACOBI_WORKERS=str(jobs),
                       SINEJACOBI_TRACE_MODE='summary')
            if archive.exists():
                need(state.get('archive_sha256') == digest(archive), 'untracked or changed proof archive')
            else:
                with phase('construction'):
                    subprocess.run([sys.executable, '-B', '-m', 'programs.stages', 'build'],
                                   cwd=ROOT, env=env, check=True)
                with phase('packaging'):
                    info = export(work/'build', archive)
                    state['archive_sha256'] = info['sha256']
            with phase('verification'):
                code = check_archive(archive, report_path, jobs, keep_work)
                need(code == 0, 'full verification failed; see verification.json')
            need(source_hashes() == requested['source_sha256'], 'sources changed during the run')
            need(input_hashes(local_inputs) == state['exact_input_sha256'], 'inputs changed during the run')
            write_summary(report_path, destination/'results.txt')
            state.update(status='COMPLETED', verification_status='PASS_FULL_NUMERICS')
            if not keep_work: shutil.rmtree(work)
        except (Exception, KeyboardInterrupt) as exc:
            state.update(status='FAILED', error=str(exc) or type(exc).__name__, work_directory=str(work))
            current = load_json(report_path.read_text())
            if current.get('status') != 'FAIL':
                json_write(report_path, {'status': 'FAIL', 'error': state['error']})
            print('FAILED:', state['error'], file=sys.stderr, flush=True)
            return 1
        finally:
            state['wall_seconds_this_invocation'] = round(time.monotonic()-start, 3)
            state['finished_utc'] = datetime.now(timezone.utc).isoformat()
            json_write(manifest_path, state)
        print('PASS_FULL_NUMERICS:', destination, flush=True)
        return 0
