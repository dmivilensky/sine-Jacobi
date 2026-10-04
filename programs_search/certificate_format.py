"""Derive and verify rational endpoint cancellations for storage potentials.

Lemma lem:regular-criterion requires polynomial identities at q=0 and q=2.
Fraction arithmetic reconstructs each potential and verifies all eight
quotients, including the derivative relation. The native reader repeats
these identities over Q using GMP before evaluating the state inequalities."""
from __future__ import annotations
from fractions import Fraction as F
from pathlib import Path
import json
import re


def strip(p):
    p = list(p)
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return p


def multiply(a, b):
    out = [F(0)]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i+j] += x*y
    return strip(out)


def derivative(p):
    return strip([i*p[i] for i in range(1, len(p))] or [F(0)])


def shift_power(p, n):
    return [F(0)]*n+list(p)


def divide_endpoint(p, right):
    p = strip(p)
    if p == [F(0)]:
        return p
    if not right:
        if p[0] != 0:
            raise ValueError("polynomial has a nonzero left endpoint value")
        quotient = strip(p[1:])
        divisor = [F(0), F(1)]
    else:
        quotient = [p[0]/2]
        for c in p[1:-1]:
            quotient.append((c+quotient[-1])/2)
        divisor = [F(2), F(-1)]
    if strip(multiply(quotient, divisor)) != strip(p):
        raise ValueError("endpoint cancellation identity failed")
    return quotient


def rational_text(x):
    x = F(x)
    return str(x.numerator) if x.denominator == 1 else str(x.numerator)+"/"+str(x.denominator)


def storage_terms(search, aggregate=True):
    if len(search["labels"]) != len(search["parameters"]) or len(search["basis"]) != len(search["labels"]):
        raise ValueError("inconsistent search coefficient arrays")
    if aggregate:
        groups = {}
        for label, theta, basis in zip(search["labels"], search["parameters"], search["basis"]):
            if basis is None:
                continue
            name = label.split(":")[0]
            p = [F(str(theta))*F(str(c)) for c in basis["q_polynomial"]]
            out = groups.setdefault(name, [F(0)]*len(p))
            out.extend([F(0)]*max(0, len(p)-len(out)))
            for i, c in enumerate(p):
                out[i] += c
        merged = {"labels": [name+":0" for name in groups],
                  "parameters": [1]*len(groups),
                  "basis": [{"q_polynomial": [rational_text(c) for c in p]}
                            for p in groups.values()]}
        return storage_terms(merged, aggregate=False)
    result = []
    for label, theta, basis in zip(search["labels"], search["parameters"], search["basis"]):
        if basis is None:
            continue
        polynomial = [F(str(c)) for c in basis["q_polynomial"]]
        name = label.split(":")[0]
        if name == "h":
            kind, powers = "h", (0, 0)
            raw = [shift_power(derivative(polynomial), 1),
                   shift_power(polynomial, 1), [F(0)], [F(0)]]
        else:
            kind = "u"
            if name == "u0":
                powers = (1, 0)
            elif name == "u1":
                powers = (0, 1)
            else:
                match = re.fullmatch(r"u0\^(\d+)u1\^(\d+)", name)
                if match is None:
                    raise ValueError("unknown state monomial: "+name)
                powers = tuple(map(int, match.groups()))
            total = sum(powers)
            if total < 1:
                raise ValueError("constant state monomials are excluded")
            raw = [shift_power(derivative(polynomial), total),
                   shift_power(polynomial, total-1),
                   shift_power(polynomial, total),
                   shift_power(polynomial, total+1)]
        quotients = [[divide_endpoint(p, right) for p in raw] for right in [False, True]]
        result.append({"label": label, "kind": kind, "powers": powers,
                       "theta": rational_text(F(str(theta))),
                       "polynomial": [rational_text(c) for c in polynomial],
                       "quotients": [[[rational_text(c) for c in p] for p in side]
                                     for side in quotients]})
    return result


def write_transport(search_path, output):
    search = json.loads(Path(search_path).read_text())
    by_label = dict(zip(search["labels"], search["parameters"]))
    terms = storage_terms(search)
    lines = ["TRANSPORT_POLYNOMIAL 1", "2"]
    lines += [rational_text(F(str(s))) for s in search["scales"]]
    lines += [rational_text(F(str(by_label.get(label, 0))))
              for label in ["xi_0", "xi_1", "tau", "chi"]]
    lines += [str(len(terms))]
    for term in terms:
        lines += [term["kind"]+" "+" ".join(map(str, term["powers"]))+" "+term["theta"]]
        for side in term["quotients"]:
            for polynomial in side:
                lines += [str(len(polynomial))+" "+" ".join(polynomial)]
    Path(output).write_text("\n".join(lines)+"\n")
    return search, terms


def verify_transport_polynomial(path):
    """Use the acceptance reader for exact endpoint identities (lem:regular-criterion)."""
    from audit.checks import polynomial
    return polynomial(Path(path).read_text())
