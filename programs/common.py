"""Construct native witnesses and retain content-based completion receipts."""
from contextlib import contextmanager, nullcontext
from pathlib import Path
import fcntl
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
INPUTS=Path(os.environ.get('SINEJACOBI_INPUTS',ROOT/'inputs')).resolve()
RESULTS=Path(os.environ.get('SINEJACOBI_RESULTS',ROOT/'results')).resolve()
LOGS=RESULTS/'logs'
FLAGS=['-std=c++17','-O3','-Wall','-Wextra','-Wpedantic','-ffp-contract=off','-fno-fast-math']
DEPENDENCIES={'sine':(), 'reference':(), 'primal_second':(),
              'second':('reference',), 'build':('sine','reference','second','primal_second')}
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.update(PYTHONHASHSEED='0',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',LC_ALL='C')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def json_write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.new')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n'); tmp.replace(path)


def config(stage):
    data=json.loads((INPUTS/'run.json').read_text())
    if data.get('format')!='SINEJACOBI_RUN_1': raise ValueError('unknown run configuration')
    return data[stage]


def toolchain_state():
    compiler=shutil.which(os.environ.get('CXX','c++'))
    result={'compiler':compiler,'flags':FLAGS}
    if compiler:
        result['compiler_sha256']=digest(Path(compiler).resolve())
    prefix=os.environ.get('SINEJACOBI_NATIVE_PREFIX')
    if prefix:
        paths=[Path(prefix)/n for n in ('include/gmp.h','include/mpfr.h','lib/libgmp.a','lib/libmpfr.a')]
    else:
        paths=[Path(n) for n in ('/usr/include/mpfr.h','/usr/include/gmp.h',
               '/usr/include/x86_64-linux-gnu/gmp.h','/usr/local/include/mpfr.h','/usr/local/include/gmp.h')]
        if compiler:
            for library in ('libmpfr.so','libgmp.so','libmpfr.dylib','libgmp.dylib'):
                p=subprocess.check_output([compiler,'-print-file-name='+library],text=True).strip()
                if p!=library: paths.append(Path(p))
    result['dependency_files']={str(p.resolve()):digest(p.resolve()) for p in paths if p.is_file()}
    return result


def workers():
    count=int(os.environ.get('SINEJACOBI_WORKERS','8'))
    if count<1: raise ValueError('worker count must be positive')
    return count


@contextmanager
def cpu_slot(label):
    LOGS.mkdir(parents=True,exist_ok=True); guard=None
    while guard is None:
        for index in range(workers()):
            candidate=(LOGS/f'worker-{index:02d}.log').open('a')
            try: fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: candidate.close(); continue
            guard=candidate; break
        if guard is None: time.sleep(.1)
    start=time.monotonic()
    try:
        print('BEGIN',label,file=guard,flush=True); yield
    except BaseException:
        print('FAILED',label,file=guard,flush=True); raise
    finally:
        print('END seconds',time.monotonic()-start,file=guard,flush=True); guard.close()


def command(argv, trace=None, *, numerical=False, log=None):
    print('+',' '.join(map(str,argv)),flush=True)
    env=os.environ.copy()
    if trace is not None: env['SINEJACOBI_TRACE']=str(trace)
    with cpu_slot(str(argv[0])) if numerical else nullcontext():
        with open(log,'a') if log is not None else nullcontext() as stream:
            if stream is not None:
                print('BEGIN',time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                      ' '.join(map(str,argv)),file=stream,flush=True)
            subprocess.run(list(map(str,argv)),cwd=ROOT,env=env,check=True,
                           stdout=stream,stderr=subprocess.STDOUT if stream else None)


def native(name,*args,trace=None,compile_only=False,source_file=None):
    source=ROOT/'programs/native'; binaries=Path(os.environ.get('SINEJACOBI_NATIVE_CACHE',ROOT/'.cache/native')).resolve()
    binaries.mkdir(parents=True,exist_ok=True); LOGS.mkdir(parents=True,exist_ok=True)
    compiler=shutil.which(os.environ.get('CXX','c++'))
    if not compiler: raise FileNotFoundError('C++17 compiler unavailable; run compute.py doctor')
    target=binaries/name; receipt=binaries/(name+'.json')
    entry=Path(source_file) if source_file is not None else source/(name+'.cpp')
    includes=['-I'+str(source)]; libraries=['-lmpfr','-lgmp']
    prefix=os.environ.get('SINEJACOBI_NATIVE_PREFIX')
    if prefix:
        prefix=Path(prefix).resolve()
        includes+=['-I'+str(prefix/'include')]
        libraries=[str(prefix/'lib/libmpfr.a'),str(prefix/'lib/libgmp.a')]
    sources=[entry,*sorted(source.glob('*.hpp'))]
    key={'toolchain':toolchain_state(),
         'flags':FLAGS,'includes':includes,'libraries':libraries,
         'sources':{p.name:digest(p) for p in sources}}
    if prefix: key['static_libraries']={n:digest(n) for n in libraries}
    with (binaries/'build.lock').open('a') as lock, (LOGS/'compiler.log').open('a') as log:
        fcntl.flock(lock,fcntl.LOCK_EX)
        saved=json.loads(receipt.read_text()) if receipt.exists() else {}
        if not target.exists() or saved.get('inputs')!=key or saved.get('binary_sha256')!=digest(target):
            argv=[compiler,*FLAGS,*includes,entry,*libraries,'-o',target.with_suffix('.new')]
            print('+',' '.join(map(str,argv)),file=log,flush=True)
            with cpu_slot('compile '+name):
                try: subprocess.run(list(map(str,argv)),stdout=log,stderr=subprocess.STDOUT,check=True)
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError('native build failed; inspect '+str(LOGS/'compiler.log')) from exc
            target.with_suffix('.new').replace(target)
            json_write(receipt,{'inputs':key,'binary_sha256':digest(target)})
    if compile_only: return target
    command([target,*args],trace,numerical=True)


def context(name):
    sources=[*ROOT.joinpath('programs').rglob('*.py'),*ROOT.joinpath('programs/native').glob('*.hpp'),
             *ROOT.joinpath('programs/native').glob('*.cpp')]
    sources += [*ROOT.joinpath('portable').glob('*.py'),ROOT/'verify_proof.py']
    sources += [*ROOT.joinpath('audit').glob('*.py')]
    deps={s:digest(RESULTS/s/'complete.json') for s in DEPENDENCIES[name]}
    return {'sources':{str(p.relative_to(ROOT)):digest(p) for p in sorted(sources) if p.is_file()},
            'inputs':{p.name:digest(p) for p in sorted(INPUTS.glob('*')) if p.is_file()},
            'dependencies':deps,'python':sys.version,'platform':platform.platform(),
            'trace_mode':os.environ.get('SINEJACOBI_TRACE_MODE','summary'),
            'native_prefix':os.environ.get('SINEJACOBI_NATIVE_PREFIX'),
            'toolchain':toolchain_state()}


def completed(directory, expected, required):
    path=directory/'complete.json'
    if not path.is_file(): return False
    try:
        value=json.loads(path.read_text())
        if value.get('format')!='SINEJACOBI_STAGE_1' or value.get('context')!=expected: return False
        records=value['outputs']
        return set(required)<=set(records) and all((directory/n).is_file() and
                    records[n]['sha256']==digest(directory/n) for n in records)
    except (OSError,ValueError,KeyError,TypeError): return False


@contextmanager
def calculation(name,required=()):
    if name not in DEPENDENCIES: raise ValueError('unknown stage: '+name)
    LOGS.mkdir(parents=True,exist_ok=True)
    with (LOGS/(name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        destination=RESULTS/name; expected=context(name)
        if completed(destination,expected,required):
            print('available:',destination,flush=True); yield None; return
        if destination.exists():
            raise RuntimeError(f'{destination} is untracked, changed or stale; use a new --output directory.')
        temporary=Path(tempfile.mkdtemp(prefix=name+'-',dir=LOGS))
        print('running:',name,'work:',temporary,flush=True)
        started=time.monotonic()
        try:
            yield temporary
            missing=[n for n in required if not (temporary/n).is_file()]
            if missing: raise RuntimeError('missing stage outputs: '+', '.join(missing))
            if context(name)!=expected: raise RuntimeError('inputs changed during computation')
            tracked=set(required)
            if name=='second':
                tracked.update(str(p.relative_to(temporary)) for p in (temporary/'price-claims').glob('*/values.txt'))
            evidence={n:{'sha256':digest(temporary/n),'bytes':(temporary/n).stat().st_size} for n in sorted(tracked)}
            json_write(temporary/'complete.json',{'format':'SINEJACOBI_STAGE_1','stage':name,
                       'context':expected,'outputs':evidence,'elapsed_seconds':time.monotonic()-started})
            temporary.rename(destination)
            print('completed:',destination,flush=True)
        except BaseException:
            print('FAILED; partial files retained at',temporary,file=sys.stderr,flush=True); raise


def ensure(stage):
    if stage not in DEPENDENCIES: raise ValueError('unknown stage: '+stage)
    LOGS.mkdir(parents=True,exist_ok=True)
    print('stage:',stage,'log:',LOGS/(stage+'.log'),flush=True)
    command([sys.executable,'-B','-m','programs.stages',stage],log=LOGS/(stage+'.log'))
    return RESULTS/stage
