"""Execution support for the candidate search modules."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

PACKAGE = Path(__file__).resolve().parent


def workers():
    value = int(os.environ.get('SINEJACOBI_WORKERS', '1'))
    if value < 1:
        raise ValueError('SINEJACOBI_WORKERS must be positive')
    return value


def work_root():
    return Path(os.environ.get('SINEJACOBI_SEARCH_WORK', 'search_work')).resolve()


@contextmanager
def cpu_slot(label):
    logs = work_root() / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    guard = None
    while guard is None:
        for index in range(workers()):
            candidate = (logs / f'worker-{index:02d}.log').open('a')
            try:
                fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                candidate.close()
                continue
            guard = candidate
            break
        if guard is None:
            time.sleep(.1)
    start = time.monotonic()
    try:
        print('BEGIN', label, file=guard, flush=True)
        yield
    finally:
        print('END seconds', time.monotonic() - start, file=guard, flush=True)
        guard.close()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_native():
    """Compile only the primal producer; respect Homebrew prefixes."""
    source = PACKAGE / 'native'
    shared = PACKAGE.parent / 'programs/native'
    cache = work_root() / 'native-cache'
    cache.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which(os.environ.get('CXX', 'c++'))
    if not compiler:
        raise RuntimeError('A C++17 compiler is required.')
    flags = ['-std=c++17', '-O3', '-ffp-contract=off', '-fno-fast-math',
             '-I' + str(shared)]
    libraries = ['-lmpfr', '-lgmp']
    prefix = os.environ.get('SINEJACOBI_NATIVE_PREFIX')
    dependencies = []
    if prefix:
        root = Path(prefix).resolve()
        flags.append('-I' + str(root / 'include'))
        archives = [root / 'lib' / ('lib' + name + '.a') for name in ('mpfr', 'gmp')]
        if all(p.is_file() for p in archives):
            libraries = list(map(str, archives))
            dependencies.extend(archives)
        else:
            libraries = ['-L' + str(root / 'lib'), '-Wl,-rpath,' + str(root / 'lib'),
                         '-lmpfr', '-lgmp']
        dependencies += [p for p in (root / 'include').glob('*.h') if p.name in ('gmp.h', 'mpfr.h')]
    entry = source / 'produce_primal.cpp'
    key = {'compiler': str(Path(compiler).resolve()),
           'compiler_sha256': digest(Path(compiler).resolve()),
           'flags': flags, 'libraries': libraries,
           'source_sha256': {p.name: digest(p) for p in [entry, *sorted(shared.glob('*.hpp'))]},
           'dependencies': {str(p): digest(p) for p in dependencies}}
    binary = cache / 'produce_primal'
    receipt = cache / 'build.json'
    with (cache / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = json.loads(receipt.read_text()) if receipt.exists() else {}
        if binary.is_file() and old.get('inputs') == key and old.get('binary_sha256') == digest(binary):
            return binary
        temporary = binary.with_suffix('.new')
        log_path = cache / 'compiler.log'
        with log_path.open('w') as log:
            command = [compiler, *flags, str(entry), *libraries, '-o', str(temporary)]
            print(' '.join(command), file=log, flush=True)
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            temporary.unlink(missing_ok=True)
            raise RuntimeError('Native compilation failed; see ' + str(log_path))
        temporary.replace(binary)
        receipt.write_text(json.dumps({'inputs': key, 'binary_sha256': digest(binary)}, indent=2) + '\n')
    return binary

