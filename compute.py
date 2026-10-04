#!/usr/bin/env python3
"""Reproduce the numerical experiment or recheck one self-contained proof."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name, help_text in (
        ('reproduce', 'search for candidates, construct witnesses, then fully verify'),
        ('certify', 'construct and fully verify from existing exact inputs')):
        p = sub.add_parser(name, help=help_text)
        p.add_argument('--output', type=Path, default=ROOT/'results')
        p.add_argument('--jobs', type=int, default=8)
        p.add_argument('--resume', action='store_true', help='resume the same failed/interrupted run')
        p.add_argument('--keep-work', action='store_true', help='also retain successful intermediate files')
        if name == 'certify':
            p.add_argument('--inputs', type=Path, default=ROOT/'inputs')
        else:
            p.add_argument('--settings', type=Path, default=ROOT/'programs_search/settings.json')
    p = sub.add_parser('verify', aliases=['verify-proof'], help='fully recheck an existing proof archive')
    p.add_argument('--proof', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('verification-new.json'))
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--keep-work', action='store_true')
    p = sub.add_parser('generate', help='candidate search only; no full certificate acceptance')
    p.add_argument('--output', type=Path, default=ROOT/'generated-inputs')
    p.add_argument('--work', type=Path, default=ROOT/'search-work')
    p.add_argument('--settings', type=Path, default=ROOT/'programs_search/settings.json')
    p.add_argument('--jobs', type=int, default=8)
    sub.add_parser('doctor', help='compile and run a small arithmetic-environment probe')
    p = sub.add_parser('test', help='developer tests; cannot certify the paper results')
    p.add_argument('--native', action='store_true')
    p.add_argument('--search', action='store_true')
    args = parser.parse_args(argv)
    jobs = getattr(args, 'jobs', 1)
    if jobs < 1: parser.error('--jobs must be positive')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ.update(SINEJACOBI_WORKERS=str(jobs), SINEJACOBI_TRACE_MODE='summary',
                      PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', LC_ALL='C')
    if args.command in ('reproduce', 'certify'):
        output = args.output.resolve()
        os.environ['SINEJACOBI_INPUTS'] = str(output/'inputs')
        os.environ['SINEJACOBI_RESULTS'] = str(output/'work/build')
        from programs.workflow import run
        return run(output, jobs=jobs, inputs=getattr(args,'inputs',None),
                   settings=getattr(args,'settings',None), resume=args.resume, keep_work=args.keep_work)
    if args.command in ('verify', 'verify-proof'):
        from programs.proof import check_archive
        return check_archive(args.proof, args.output, jobs, args.keep_work)
    if args.command == 'generate':
        return subprocess.call([sys.executable, '-B', '-m', 'programs_search.generate',
                                '--output', str(args.output.resolve()), '--work', str(args.work.resolve()),
                                '--settings', str(args.settings.resolve()), '--jobs', str(jobs)], cwd=ROOT)
    if args.command in ('doctor', 'test'):
        # Diagnostics must not populate the default experiment directory.
        os.environ['SINEJACOBI_RESULTS'] = str(ROOT/'.cache/diagnostics')
    if args.command == 'doctor':
        from programs.common import native
        from portable.environment import runtime
        import json
        print(json.dumps(runtime(jobs), indent=2))
        native('probe')
        return 0
    if args.command == 'test':
        env = os.environ.copy()
        env['SINEJACOBI_TEST_NATIVE'] = '1' if args.native else '0'
        code = subprocess.call([sys.executable,'-B','-m','unittest','discover','-s','tests','-v'], cwd=ROOT, env=env)
        if code or not args.search: return code
        return subprocess.call([sys.executable,'-B','-m','unittest','programs_search.test_search','-v'], cwd=ROOT, env=env)
    raise AssertionError('unhandled command')


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print('ERROR:', exc, file=sys.stderr)
        sys.exit(1)
