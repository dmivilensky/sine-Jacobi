"""Propose a ground-exact polynomial under equality constraints.

SVD of the computed Jacobian rescales the equations; a coefficient cube
bounds the initial change of log drift by sum|delta c_j|, since |T_j|<=1.
SLSQP is Kraft's sequential quadratic programming method (DFVLR-FB 88-28,
1988), as implemented and documented in SciPy 1.17:
https://docs.scipy.org/doc/scipy-1.17.0/reference/optimize.minimize-slsqp.html
Acceptance of the article's ceiling requires the separate whole-cube
inequalities of lem:contraction and prop:primal-jacobian."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.chebyshev import chebfit
from scipy.optimize import minimize
from .model import Reference
from .primal_model import Primal
from .search_reference import EPS, K


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--scales", type=float, nargs="+", required=True)
    ap.add_argument("--degree", type=int, required=True)
    ap.add_argument("--order", type=int, required=True)
    ap.add_argument("--second", action="store_true")
    ap.add_argument("--iterations", type=int, required=True)
    ap.add_argument("--start", type=Path)
    ap.add_argument("--bounded", action="store_true",
                    help="control the proposed change of log drift on each optimization stage")
    ap.add_argument("--work", type=Path, required=True)
    args = ap.parse_args()
    if args.degree<0 or args.order<=args.degree or args.iterations<1:
        ap.error('require degree >= 0, order > degree and positive iteration budget')
    args.work.mkdir(parents=True, exist_ok=True)
    parameters = json.loads(args.reference.read_text())
    scales = args.scales+[float(parameters["beta"])*K]
    if len(scales)+2+int(args.second)>args.degree+1:
        ap.error('the polynomial must have at least as many coefficients as constraints')
    model = Primal(scales, args.degree, args.order, args.second)
    if args.start:
        previous = np.array(json.loads(args.start.read_text())["coefficients"], float)
        initial = np.pad(previous, (0, max(0, args.degree+1-len(previous))))[:args.degree+1]
    else:
        ref = Reference(parameters, digits=14)
        # Fit the computed logarithmic regular drift on Chebyshev nodes.
        x = np.cos(np.pi*(np.arange(args.order)+.5)/args.order)
        _, P = ref.local((x+1)/2)
        initial = chebfit(x, np.log(3/P), args.degree)
    first = model.evaluate(initial)
    U, singular, V = np.linalg.svd(first["jacobian"], full_matrices=False)
    if np.min(singular) <= EPS*np.max(singular):
        raise ArithmeticError("constraint Jacobian is numerically rank deficient")
    whiten = U.T/singular[:, None]
    projected_gradient = first["energy_gradient"]-V.T@(V@first["energy_gradient"])
    energy_scale = max(np.linalg.norm(projected_gradient), EPS**.5)
    # SLSQP uses one tolerance for objective and whitened constraints.  The
    # objective is therefore scaled by its computed tangent gradient, and
    # its tolerance includes the arithmetic noise magnified by whitening.
    solver_tolerance = max(EPS**.8, np.linalg.norm(whiten, np.inf)
                           *np.linalg.norm(model.target, np.inf)*EPS)
    history = []

    def callback(x):
        result = model.evaluate(x)
        row = {"energy": float(result["energy"]),
               "residual": result["residual"].tolist(),
               "coefficients": [repr(float(c)) for c in x]}
        history.append(row)
        with (args.work/"iterations.jsonl").open('a') as stream:
            stream.write(json.dumps({"iteration": len(history), **row})+"\n")
        print(json.dumps({"iteration": len(history), "energy": row["energy"],
                          "residual": row["residual"]}), flush=True)

    current=initial.copy();remaining=args.iterations;radius=1/(args.degree+1)
    while remaining>0:
        origin=current.copy()
        # |T_j|<=1 gives |change(log D)|<=1 on this first cube. This
        # avoids unbounded SQP excursions in almost dependent constraints.
        # The radius doubles only when the computed optimum reaches a face.
        # It is an optimization resource, not the certified primal radius.
        bounds=list(zip(origin-radius,origin+radius)) if args.bounded else None
        result = minimize(lambda x: (model.evaluate(x)["energy"]-first["energy"])/energy_scale, current,
                          jac=lambda x: model.evaluate(x)["energy_gradient"]/energy_scale,
                          constraints={"type": "eq",
                                       "fun": lambda x: whiten@model.evaluate(x)["residual"],
                                       "jac": lambda x: whiten@model.evaluate(x)["jacobian"]},
                          bounds=bounds,method="SLSQP", callback=callback,
                          options={"maxiter": remaining, "ftol": solver_tolerance})
        remaining-=max(1,result.nit);current=result.x
        if not args.bounded or np.max(np.abs(current-origin))<radius*(1-EPS**.5):break
        radius*=2
    final = model.evaluate(result.x)
    report = {"status": "ordinary primal proposal; exact feasibility unverified",
              "success": bool(result.success), "message": str(result.message),
              "degree": args.degree, "order": args.order, "second": args.second,
              "energy_scale": energy_scale, "solver_tolerance": solver_tolerance,
              "bounded_stages": args.bounded,
              "small_scales": args.scales, "beta": parameters["beta"],
              "coefficients": [repr(float(c)) for c in result.x],
              "energy": float(final["energy"]),
              "residual": final["residual"].tolist(),
              "jacobian": final["jacobian"].tolist(),
              "initial_singular_values": singular.tolist(), "history": history}
    (args.work/"primal.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ["history", "jacobian"]}), flush=True)


if __name__ == "__main__":
    main()
