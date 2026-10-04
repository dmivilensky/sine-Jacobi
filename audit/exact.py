"""Independent rational arithmetic and elementary-function enclosures.

The 300-bit outward dyadic rounding limits integer growth, not validity.
All acceptance predicates use exceptions, including under python -O.
"""
from fractions import Fraction as F
import json
import re


def need(ok, message):
    if not ok:
        raise ValueError(message)


def rational(s):
    need(isinstance(s, (str, int, F)) and not isinstance(s, bool),
         'rational input must be an exact string or integer')
    if not isinstance(s, str) or '0x' not in s.lower():
        return F(s)
    m = re.fullmatch(r'(-?)0x([0-9a-fA-F]+)(?:\.([0-9a-fA-F]*))?p([+-]?\d+)', s)
    need(m is not None, 'invalid hexadecimal rational: ' + s)
    sign, a, b, e = m.groups()
    b = b or ''
    return (-1 if sign else 1) * F(int(a+b, 16)) * F(2)**(int(e)-4*len(b))


def unique_pairs(pairs):
    result = {}
    for k, v in pairs:
        need(k not in result, 'duplicate JSON key: ' + k)
        result[k] = v
    return result


def load_json(text):
    return json.loads(text, object_pairs_hook=unique_pairs,
                      parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


def records(text, magic):
    lines = text.splitlines()
    need(bool(lines) and lines[0] == magic, 'wrong interval format: ' + magic)
    out = {}
    for line in lines[1:]:
        a = line.split()
        need(len(a) >= 3, 'truncated interval record')
        if magic=='PRIMAL_VALUES 1' and a[0] in ('F','J'):
            count=1 if a[0]=='F' else 2
            need(len(a)==count+3 and all(re.fullmatch(r'0|[1-9][0-9]*',v) for v in a[1:1+count]),'malformed indexed interval')
        else:
            need(len(a)==3,'malformed scalar interval')
        key = '_'.join(a[:-2])
        need(key not in out, 'duplicate interval record: ' + key)
        out[key] = rational(a[-2]), rational(a[-1])
        need(out[key][0] <= out[key][1], 'reversed interval: ' + key)
    need(bool(out), 'empty interval report')
    return out


def pt(x):
    return F(x), F(x)


def add(a, b): return a[0]+b[0], a[1]+b[1]
def neg(a): return -a[1], -a[0]
def sub(a, b): return add(a, neg(b))
def mul(a, b):
    v = [x*y for x in a for y in b]
    return min(v), max(v)
def scale(a, b): return mul(a, pt(b))
def div(a, b):
    need(b[0]*b[1] > 0, 'zero in denominator')
    return mul(a, (1/b[1], 1/b[0]))
def mag(a): return max(abs(x) for x in a)
def encloses(a, b): return a[0] <= b[0] <= b[1] <= a[1]
def isum(values):
    out = pt(0)
    for v in values: out = add(out, v)
    return out


def trim(a):
    a = list(a)
    while len(a) > 1 and a[-1] == 0: a.pop()
    return a


def poly_mul(a, b):
    out = [F(0)]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b): out[i+j] += x*y
    return trim(out)


def derivative(a): return trim([j*a[j] for j in range(1, len(a))] or [F(0)])
def shift(a, n): return trim([F(0)]*n + a)


BITS = 300
UNIT = 1 << BITS


def rd(a):
    return F((a[0]*UNIT).__floor__(), UNIT), F((a[1]*UNIT).__ceil__(), UNIT)
def ra(a, b): return rd(add(a, b))
def rm(a, b): return rd(mul(a, b))
def rv(a, b): return rd(div(a, b))
def rp(a, n):
    need(isinstance(n, int) and n >= 0, 'nonnegative integer power required')
    out = pt(1)
    for _ in range(n): out = rm(out, a)
    return out


def exp_point(x):
    x = F(x)
    negative = x < 0
    x = abs(x)
    squarings = 0
    while x > F(1, 8): x /= 2; squarings += 1
    term = total = pt(1)
    for j in range(1, 101):
        term = rv(rm(term, pt(x)), pt(j)); total = ra(total, term)
    first = term[1]*x/101
    total = rd((total[0], total[1]+first/(1-x/102)))
    for _ in range(squarings): total = rm(total, total)
    return rv(pt(1), total) if negative else total


def rexp(a): return exp_point(a[0])[0], exp_point(a[1])[1]
def atan_partial(x, n):
    return sum(((-1)**j*x**(2*j+1)/F(2*j+1) for j in range(n)), F(0))


def sine_enclosure(degree=3):
    """Derive the infinite sine-energy enclosure without MPFR or programs/."""
    need(isinstance(degree, int) and 0 <= degree <= 50, 'unsupported sine degree')
    lo = 16*atan_partial(F(1,5),100)-4*atan_partial(F(1,239),101)
    hi = 16*atan_partial(F(1,5),101)-4*atan_partial(F(1,239),100)
    p = rd((lo, hi)); a = rexp(neg(p))
    MA = rv(pt(1), sub(pt(1), rp(a, 2)))
    MB = ra(pt(1), rv(scale(a, 2), sub(pt(1), rp(a, 3))))
    factor = rv(pt(512), rp(p, 3))
    def integral(lam):
        v = ra(ra(rv(rp(p,2),pt(lam)),rv(scale(p,2),pt(lam**2))),pt(2/lam**3))
        return rm(rexp(scale(p,-lam)),v)
    A = [F(0)]*(degree*(degree+1)+1)
    B = [F(0)]*(degree**2+1)
    for j in range(degree+1): A[j*(j+1)] = 1
    B[0] = 1
    for j in range(1,degree+1): B[j*j] = 2*(-1)**j
    C = [F(1)]
    for _ in range(6): C = poly_mul(poly_mul(C,A),B)
    total = pt(0)
    for n,c in enumerate(C): total = ra(total,scale(integral(F(2*n+3,2)),c))
    finite = rm(factor,total)
    ta = rv(integral(F(3,2)+(degree+1)*(degree+2)),
            sub(pt(1),rexp(scale(p,-2*(degree+2)))))
    tb = rv(integral(F(3,2)+(degree+1)**2),
            sub(pt(1),rexp(scale(p,-(2*degree+3)))))
    tail = rm(factor,ra(scale(rm(rm(rp(MA,5),rp(MB,6)),ta),6),
                        scale(rm(rm(rp(MA,6),rp(MB,5)),tb),12)))
    return {'pi':p, 'coefficients':C, 'finite_sum':finite, 'tail':tail,
            'energy':(finite[0]-tail[1],finite[1]+tail[1])}


def sqrt_interval(a, digits=24):
    """Enclose both square roots on a decimal grid using integer arithmetic."""
    from math import isqrt
    need(isinstance(digits,int) and digits >= 0, 'invalid decimal precision')
    need(0 <= a[0] <= a[1], 'nonnegative ordered interval required')
    q=10**digits
    def endpoint(x, upper):
        x=F(x); n=x.numerator*q*q; d=x.denominator
        k=isqrt(n//d)
        if upper and k*k*d<n: k+=1
        return F(k,q)
    return endpoint(a[0],False),endpoint(a[1],True)
