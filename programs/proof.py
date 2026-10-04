"""Export complete portable evidence; publication is not proof acceptance."""
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

from .common import ROOT, INPUTS, digest, json_write
from .pack import collect
from portable.verify import safe_name
from audit.exact import need, load_json

CPP = ('probe', 'verify_reference', 'verify_transport', 'verify_primal', 'verify_sine')


def checker_sources(destination):
    """Explicit source-only snapshot, also used for isolated infrastructure tests."""
    destination = Path(destination)
    for folder in ('audit', 'portable', 'native'):
        (destination/folder).mkdir(parents=True, exist_ok=True)
    for folder in ('audit', 'portable'):
        for source in (ROOT/folder).glob('*.py'):
            shutil.copyfile(source, destination/folder/source.name)
    for source in (ROOT/'programs/native').glob('*.hpp'):
        shutil.copyfile(source, destination/'native'/source.name)
    for name in CPP:
        shutil.copyfile(ROOT/'programs/native'/(name+'.cpp'), destination/'native'/(name+'.cpp'))
    shutil.copyfile(ROOT/'verify_proof.py', destination/'verify.py')


def export(data, output):
    data, output = Path(data).resolve(), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle = collect(data)
    need(not bundle.get('test_fixture'), 'cannot export a synthetic production proof')
    with tempfile.TemporaryDirectory(prefix='sinejacobi-pack-') as tmp:
        root = Path(tmp)
        checker_sources(root)
        shutil.copyfile(INPUTS/'run.json', root/'run.json')
        for name, item in bundle['artifacts'].items():
            target = root/'data'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(data/name, target)
            need(digest(target) == item['sha256'], 'witness changed during export: '+name)
        geometry = bundle['cover_geometry']
        bundle['scope'] = 'Unverified claims until verify.py successfully rechecks every native premise.'
        json_write(root/'claims.json', bundle)
        (root/'README.txt').write_text(
            'SELF-CONTAINED NUMERICAL PROOF\n\n'
            'Requires Python >=3.10, C++17, GMP and MPFR development files.\n'
            'macOS: brew install gmp mpfr\n'
            '       export SINEJACOBI_NATIVE_PREFIX="$(brew --prefix)"\n'
            'Run: python3 -I -B verify.py --jobs 8 --output verification.json\n'
            'No original repository or results directory is needed.\n'
            'PASS_FULL_NUMERICS proves the enclosed numerical results, without any document comparison.\n'
            'Native interval kernels are shared with the producer; this is an independent replay,\n'
            'not an independently developed mathematical implementation or formal proof assistant.\n'
            'The manifest detects byte corruption; it is not a signature or proof of source correctness.\n'
            'Temporary files are removed on success; failures retain diagnostics. --keep-work retains successful work.\n')
        members = {str(p.relative_to(root)): {'bytes': p.stat().st_size, 'sha256': digest(p)}
                   for p in sorted(root.rglob('*')) if p.is_file()}
        manifest = {'format': 'SINEJACOBI_PROOF_3', 'members': members,
                    'uncompressed_bytes': sum(v['bytes'] for v in members.values()),
                    'cover_cells': geometry['cells'], 'cover_leaves': geometry['leaves'],
                    'status': 'EXPORTED_NOT_VERIFIED'}
        json_write(root/'manifest.json', manifest)
        temporary = output.with_name(output.name+'.new')
        with tarfile.open(temporary, 'w:xz', preset=6) as archive:
            for p in sorted(root.rglob('*')):
                if not p.is_file(): continue
                info = tarfile.TarInfo(str(p.relative_to(root)))
                info.size = p.stat().st_size
                info.mode = 0o644
                with p.open('rb') as stream: archive.addfile(info, stream)
        temporary.replace(output)
    return {'archive': str(output), 'bytes': output.stat().st_size, 'sha256': digest(output),
            'uncompressed_payload_bytes': manifest['uncompressed_bytes'],
            'cover_cells': geometry['cells'], 'cover_leaves': geometry['leaves'],
            'status': 'EXPORTED_NOT_VERIFIED'}


@contextmanager
def unpack(path):
    """Extract regular, relative files only; no links or archive-controlled paths."""
    with tempfile.TemporaryDirectory(prefix='sinejacobi-proof-') as tmp:
        root = Path(tmp)
        seen = set()
        total = 0
        with tarfile.open(path, 'r:xz') as archive:
            for item in archive:
                safe_name(item.name)
                need(item.isfile() and item.name not in seen, 'archive links, directories or duplicate names are forbidden')
                seen.add(item.name)
                total += item.size
                need(0 <= item.size <= 16*(1 << 30) and total <= 32*(1 << 30), 'proof archive exceeds extraction budget')
                target = root/item.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(item) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst, length=1 << 20)
        yield root


def check_archive(path, output, jobs=1, keep_work=False):
    output = Path(output).resolve()
    need(jobs >= 1, 'jobs must be positive')
    need(output.name not in ('run.json', 'provenance.json'), 'report would overwrite run metadata')
    need(output.suffix == '.json', 'verification output must be a JSON report')
    for folder in ('inputs','programs','programs_search','audit','portable','tests'):
        need(not output.is_relative_to(ROOT/folder), 'verification output overlaps package inputs/sources')
    need(not output.is_relative_to(INPUTS), 'verification output overlaps selected inputs')
    need(output != Path(path).resolve(), 'verification report would overwrite proof archive')
    # Clear a stale PASS even if decompression or manifest validation fails.
    json_write(output, {'status': 'FAIL', 'scope': 'Verification has not completed.'})
    import time
    started = time.monotonic()
    try:
        with unpack(path) as root:
            extraction_seconds = round(time.monotonic()-started, 3)
            args = [sys.executable, '-I', '-B', root/'verify.py', '--jobs', str(jobs), '--output', output]
            if keep_work: args.append('--keep-work')
            code = subprocess.call(list(map(str, args)), cwd=root)
        result = load_json(output.read_text())
        need((code == 0) == (result.get('status') == 'PASS_FULL_NUMERICS'),
             'verifier exit status and report disagree')
        result['archive_sha256'] = digest(path)
        result['archive_extraction_seconds'] = extraction_seconds
        result['archive_check_seconds'] = round(time.monotonic()-started, 3)
        json_write(output, result)
        return code
    except (Exception, KeyboardInterrupt) as exc:
        json_write(output, {'status': 'FAIL', 'error': str(exc) or type(exc).__name__})
        print('FAIL:', exc, file=sys.stderr)
        return 1

