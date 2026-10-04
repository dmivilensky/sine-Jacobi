"""Construct candidate witnesses for ca:transport; acceptance is portable.verify.

Exact polynomial cancellation precedes adaptive spatial/state subdivision.
The reader repeats the flow, complete-cover and price-integral proofs;
convexity of the criterion (lem:conjugate, eq:closed-criterion) justifies one final scale for every accepted cell."""
from .common import native, command, workers, cpu_slot


# A work grid, fixed independently of the CPU count. Each region may refine
# itself further. Convexity of psi, psi(0)=0 (lem:conjugate), and the
# normalized criterion (lem:regular-criterion) allow lambda=min(sigma).
# The native reader proves the
# complete continuous domain and recomputes the price integral afterwards.
PRICE_SEGMENTS = 16
# Fewer verification regions amortize the independent flow replay. Their
# count and boundary rule do not depend on the number of available workers.
PRICE_CHECK_REGIONS = 8


def spatial_regions(out, segments=PRICE_SEGMENTS, *, folder='price-parts'):
    if segments < 1 or segments & (segments-1) or segments > 2**20:
        raise ValueError('spatial work grid must be a representable positive dyadic subdivision')
    return [(half, index/segments, (index+1)/segments,
             out/folder/f'{half*segments+index:04d}')
            for half in range(2) for index in range(segments)]


def merge_price_covers(regions, output):
    """Assemble a proposal only; verify_transport remains the acceptance gate.

    Check the ordered spatial partition, exact dyadic coordinates, counts,
    and common homothety before exposing the assembled cover. State-tree
    completeness, all strict inequalities, and flow/integral enclosures are
    established independently by the native reader on the full cover.
    """
    if all((d/'state-cover.bin').is_file() for _, _, _, d in regions):
        from portable.cover import Reader, MAGIC, HEADER, END
        expected_half, expected_left, cells, leaves, dilation = 0, 0., 0, 0, 1.
        temporary = output.with_suffix('.new')
        with temporary.open('wb') as merged:
            merged.write(MAGIC + HEADER.pack(0, 0, 0., 1.))
            for half, left, right, directory in regions:
                if half != expected_half or left != expected_left:
                    raise ValueError('spatial regions are not an ordered partition')
                reader = Reader(directory/'state-cover.bin')
                for cell in reader:
                    cell.write(merged)
                if (not reader.partial or (reader.half, reader.begin, reader.end) != (half, left, right)):
                    raise ValueError('packed region has wrong domain')
                cells += reader.cells; leaves += reader.count
                dilation = min(dilation, reader.dilation)
                expected_left = right
                if right == 1: expected_half += 1; expected_left = 0.
            if (expected_half, expected_left) != (2, 0.):
                raise ValueError('spatial regions do not exhaust both halves')
            merged.write(b'E' + END.pack(cells, leaves, dilation))
        temporary.replace(output)
        return
    raise ValueError('all production regions must use the packed cover format')


def produce_price_cover(reference, polynomial, cache, out, price_error,
                        *, bits=128, order=12, flow_tol='1e-20', quadrature_tol='1e-14',
                        state_budget=4096, segments=PRICE_SEGMENTS):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    binary = native('produce_transport', compile_only=True)
    regions = spatial_regions(out, segments)
    def produce_region(region):
        half, left, right, directory = region
        directory.mkdir(parents=True, exist_ok=True)
        command([binary, reference, polynomial, cache, bits, order, flow_tol,
                 repr(price_error), quadrature_tol, state_budget, directory,
                 half, left.hex(), right.hex(), 'segment'],
                trace=directory/'price-quadrature.txt', numerical=True,
                log=directory/'produce.log')
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=min(workers(), len(regions))) as pool:
        # The price Jacobian vanishes cubically at t=0; in the measured
        # dense cases the large-t regions dominate the state work. Start
        # them first and interleave both halves to shorten the final tail.
        # This changes scheduling only, never a region's mathematical input.
        dispatch = sorted(regions, key=lambda region: (-region[1], region[0]))
        pending = {pool.submit(produce_region, region): region for region in dispatch}
        for completed, future in enumerate(as_completed(pending), 1):
            future.result()
            half, left, right, _ = pending[future]
            print('completed price regions', completed, '/', len(regions),
                  'half', half, 'interval', left, right,
                  'elapsed seconds', round(time.monotonic()-started, 1), flush=True)
    with cpu_slot('assemble spatial cover '+str(out)):
        merge_price_covers(regions, out/'state-cover.bin')


def verification_regions(out, groups=PRICE_CHECK_REGIONS):
    """Balance state work using already produced leaf counts, deterministically.

    All cuts are existing exact cell boundaries. Counts affect scheduling
    only: every native reader independently validates its resulting domain.
    Minimize the larger leaves-per-group ratio between the two halves, then
    place cuts nearest to equal cumulative leaf counts within each half.
    CPU count and elapsed time play no role in this finite calculation.
    """
    from fractions import Fraction as F
    if groups < 2:
        raise ValueError('verification must cover both spatial halves')
    halves = [[], []]
    expected_half, expected_left = 0, F(0)
    complete = False
    if (out/'state-cover.bin').is_file():
        from portable.cover import Reader
        reader = Reader(out/'state-cover.bin')
        for cell in reader:
            halves[cell.half].append((F(cell.left), F(cell.right), cell.count))
        if reader.partial: raise ValueError('completed spatial cover required')
        complete = True; expected_half = 2; expected_left = F(0)
    else:
        raise FileNotFoundError(out/'state-cover.bin')
    if not complete or expected_half != 2 or expected_left != 0:
        raise ValueError('incomplete spatial work domain')
    groups = min(groups, sum(map(len, halves)))
    totals = [sum(cell[2] for cell in half) for half in halves]
    first = min(range(max(1, groups-len(halves[1])), min(len(halves[0]), groups-1)+1),
                key=lambda n: max(F(totals[0], n), F(totals[1], groups-n)))
    regions = []
    for half, count in enumerate((first, groups-first)):
        cells = halves[half]; cumulative = [0]
        for cell in cells:
            cumulative.append(cumulative[-1]+cell[2])
        cuts = [0]
        for index in range(1, count):
            cut = min(range(cuts[-1]+1, len(cells)-(count-index)+1),
                      key=lambda j: abs(count*cumulative[j]-index*cumulative[-1]))
            cuts.append(cut)
        cuts.append(len(cells))
        for left, right in zip(cuts[:-1], cuts[1:]):
            a, b = cells[left][0], cells[right-1][1]
            if F(float(a)) != a or F(float(b)) != b:
                raise ValueError('work boundary is not exactly representable in binary64')
            regions.append((half, float(a), float(b), out/'price-claims'/f'{len(regions):04d}'))
    return regions


