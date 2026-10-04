"""Collect only defining inputs, interval claims and numerical witnesses."""
from pathlib import Path
from .common import digest, config
from portable.cover import scan


def embedded(path):
    path = Path(path)
    if path.stat().st_size > 8*1024*1024:
        raise ValueError('unexpectedly large interval report: '+str(path))
    text = path.read_text(encoding='utf-8')
    import hashlib
    return {'text': text, 'sha256': hashlib.sha256(text.encode()).hexdigest()}


def collect(data):
    data = Path(data).resolve()
    names = ['reference/reference.txt', 'reference/reference-values.txt', 'second/transport.txt',
             'second/price-claim.txt', 'primal_second/profile.txt', 'primal_second/radius.txt',
             'primal_second/partition.txt', 'primal_second/values.txt',
             'sine/sine-values.txt', 'sine/sine-polynomial.txt']
    parts = sorted((data/'second/price-claims').glob('*/values.txt'))
    if not parts: raise FileNotFoundError('missing candidate price regions')
    names.extend(str(p.relative_to(data)) for p in parts)
    files = {name: embedded(data/name) for name in names}
    witnesses = ['second/state-cover.bin', 'reference/support-filter-flow.txt',
                 'reference/linear-filter.txt', 'reference/support-complement.txt',
                 'reference/support-complement-end.txt']
    witnesses += [f'reference/filter-{i}.txt' for i in range(2)]
    witnesses += [f'reference/complement-{i}.txt' for i in range(3)]
    witnesses += [f'primal_second/{half}/filter-{i}.txt' for half in ('central','cube') for i in range(4)]
    geometry = scan(data/'second/state-cover.bin', bits=config('second')['bits'])
    geometry['artifact'] = 'second/state-cover.bin'
    artifacts = {name: {'sha256': digest(data/name), 'bytes': (data/name).stat().st_size}
                 for name in witnesses}
    return {'format': 'SINEJACOBI_CERTIFICATES_2', 'files': files, 'artifacts': artifacts,
            'cover_geometry': geometry,
            'scope': 'Candidate enclosures; no full numerical acceptance until portable.verify succeeds.'}
