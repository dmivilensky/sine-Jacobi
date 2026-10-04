"""Construct witnesses only; portable.verify owns full numerical acceptance."""
import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from .common import INPUTS, DEPENDENCIES, calculation, config, ensure, native

REFERENCE_OUTPUTS = ('reference.txt', 'reference-values.txt', 'support-filter-flow.txt',
    'support-complement.txt', 'support-complement-end.txt', 'linear-filter.txt',
    'filter-0.txt', 'filter-1.txt', 'complement-0.txt', 'complement-1.txt', 'complement-2.txt')


def reference():
    with calculation('reference', REFERENCE_OUTPUTS) as out:
        if out is None: return
        shutil.copyfile(INPUTS/'reference.txt', out/'reference.txt')
        c = config('reference')
        # Without a saved flow argument this entry constructs interval witnesses.
        native('verify_reference', out/'reference.txt', c['bits'], c['order'],
               c['flow_tolerance'], c['quadrature_tolerance'], out, trace=out/'quadrature.txt')


def second():
    from audit.checks import polynomial
    from .transport import produce_price_cover
    from .price_claims import propose
    ref = ensure('reference')
    c = config('second')
    with calculation('second', ('transport.txt', 'state-cover.bin', 'price-claim.txt')) as out:
        if out is None: return
        shutil.copyfile(INPUTS/'transport.txt', out/'transport.txt')
        polynomial((out/'transport.txt').read_text())
        produce_price_cover(ref/'reference.txt', out/'transport.txt', ref, out, float(c['price_error']),
                            bits=c['bits'], order=c['order'], flow_tol=c['flow_tolerance'],
                            quadrature_tol=c['quadrature_tolerance'],
                            state_budget=c['state_budget'], segments=c['segments'])
        propose(out, c['verification_regions'], c['bits'])


def primal_second():
    c = config('primal_second')
    required = ['profile.txt', 'radius.txt', 'partition.txt', 'values.txt',
                'central/profile.txt', 'central/partition.txt']
    required += [half+'/filter-'+str(i)+'.txt' for half in ('central', 'cube') for i in range(4)]
    with calculation('primal_second', required) as out:
        if out is None: return
        native('prepare_primal', INPUTS/'profile.txt', INPUTS/'radius.txt', c['bits'], c['order'],
               c['flow_tolerance'], c['quadrature_tolerance'], out, trace=out/'construction.txt')
        (out/'candidate-values.txt').rename(out/'values.txt')


def sine():
    c = config('sine')
    with calculation('sine', ('sine-values.txt', 'sine-polynomial.txt')) as out:
        if out is not None:
            native('verify_sine', c['bits'], c['absolute_error'], out, trace=out/'sum.txt')


def build():
    # second owns reference; common caps native processes through CPU slots.
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(ensure, name) for name in ('sine', 'second', 'primal_second')]
        for future in futures: future.result()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=DEPENDENCIES)
    args = parser.parse_args()
    globals()[args.stage]()
