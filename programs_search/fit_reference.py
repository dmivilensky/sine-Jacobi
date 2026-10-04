"""Fit finite rational supports to the newly computed stationary profile.

Positive poles and amplitudes use fit_positive_rational's logarithmic
least-squares model. Multiplier polishing and doubled Gaussian quadrature
orders select a candidate for thm:contact; agreement of approximations
never replaces the independent native enclosure."""
import argparse
import json
from pathlib import Path
import numpy as np
from .search_reference import fit_positive_rational
from .model import Reference, write_native_parameters


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stationary", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--digits", type=int, default=11)
    ap.add_argument("--maximum-terms", type=int, default=8)
    args = ap.parse_args()
    if not 6<=args.digits<=13 or args.maximum_terms<1:
        ap.error('use 6..13 search digits and a positive rational-term budget')
    args.work.mkdir(parents=True, exist_ok=True)
    row = json.loads((args.stationary/"best-stationary-search.json").read_text())
    data = np.load(args.stationary/"best-stationary-search.npz")
    history = []
    best = None
    for terms in range(1, args.maximum_terms+1):
        co, error = fit_positive_rational(data["t"], data["p"], terms,
                                          10.**(-args.digits))
        params = {"schema": 1, "beta": repr(row["beta"]), "b": repr(row["b"]),
                  "gamma": repr(row["gamma"]), "support": [repr(c) for c in co]}
        ref = Reference(params, args.digits+1)
        order = 2**int(np.ceil(args.digits/2))
        polish = ref.polish_multipliers(order, 10.**(-args.digits-2))
        previous = ref.statistics(order)
        while True:
            order *= 2
            current = ref.statistics(order)
            agreement = max(abs(current[k]-previous[k])
                            for k in ["base", "mass", "root_mass", "energy"])
            if agreement <= 10.**(-args.digits):
                break
            previous = current
        record = {"terms": terms, "fit_norm": error, "quadrature_order": order,
                  "quadrature_agreement": agreement, "multipliers": polish,
                  **current, "parameters": params}
        history.append(record)
        print(json.dumps(record), flush=True)
        (args.work/"fit-search.json").write_text(json.dumps(history, indent=2)+"\n")
        if best is None or current["base"] > best["base"]:
            best = record
            (args.work/"reference.json").write_text(json.dumps(params, indent=2)+"\n")
            # No correction scales are selected during reference fitting.
            write_native_parameters(params, [], args.work/"reference.txt")
        if abs(row["energy"]-current["base"]) <= 10.**(-args.digits):
            break


if __name__ == "__main__":
    main()
