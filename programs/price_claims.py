"""Propose regional price enclosures; portable.verify owns acceptance.

No state/ODE inequality is accepted here. The final checker replays every
leaf and integral and requires its results to fit these claims.
The roundoff allowance widens a proposal and cannot bypass that gate.
"""
from pathlib import Path
from fractions import Fraction as F
from audit.exact import need, rational, add, scale, mag, pt
from audit.checks import transport_report, merge_prices
from portable.cover import Reader
from .transport import verification_regions


def encoded(value, domain=None):
    header = 'TRANSPORT_CLAIM 1\n' if domain is None else 'TRANSPORT_CLAIM_PART 1\n'
    if domain is not None:
        header += 'domain '+' '.join(map(str, domain))+'\n'
    for key in ('dilation', 'price', 'undilated_price'):
        header += key+' '+' '.join(map(str, value[key]))+'\n'
    return header+f"cells {value['cells']}\nleaves {value['leaves']}\n"


def propose(out, groups, bits):
    out = Path(out)
    regions = verification_regions(out, groups)
    reader = Reader(out/'state-cover.bin')
    stream = iter(reader)
    summaries = []
    for half, left, right, _ in regions:
        total, absolute, count, leaves = pt(0), F(0), 0, 0
        position = F(left)
        while position < F(right):
            cell = next(stream, None)
            need(cell is not None and cell.half == half and F(cell.left) == position
                 and F(cell.right) <= F(right), 'price claim region does not follow cover cells')
            value = scale(tuple(map(rational, cell.integral.split())), 1/F(cell.dilation))
            total = add(total, value)
            absolute += mag(value)
            position = F(cell.right)
            count += 1
            leaves += cell.count
        # Includes division, directed summation, merging and final scaling.
        # Same gamma_n budget as portable.cover.scan; no relative error
        # assumption about a possibly cancelling sum is used.
        operations = 2*count+16
        unit = F(1, 2**(bits-1))
        need(operations*unit < 1, 'regional summation budget exhausted')
        error = operations*unit/(1-operations*unit)*absolute
        summaries.append({'domain': (half, F(left), F(right)),
                          'undilated_price': (total[0]-error, total[1]+error),
                          'cells': count, 'leaves': leaves})
    need(next(stream, None) is None and reader.complete and not reader.partial,
         'price regions do not exhaust the packed cover')
    lam = F(reader.dilation)
    parts = []
    for index, part in enumerate(summaries):
        part.update(dilation=pt(lam), price=scale(part['undilated_price'], lam))
        directory = out/'price-claims'/f'{index:04d}'
        directory.mkdir(parents=True)
        text = encoded(part, part['domain'])
        parts.append(transport_report(text, True))
        (directory/'values.txt').write_text(text)
    total = pt(0)
    for part in parts:
        total = add(total, part['undilated_price'])
    merged = {'dilation': pt(lam), 'undilated_price': total, 'price': scale(total, lam),
              'cells': sum(p['cells'] for p in parts), 'leaves': sum(p['leaves'] for p in parts)}
    merge_prices(parts, merged)
    (out/'price-claim.txt').write_text(encoded(merged))
