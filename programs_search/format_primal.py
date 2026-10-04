"""Choose correction coordinates and a rational preconditioner.

Column-pivoted QR selects a square Jacobian submatrix (LAPACK DGEQP3):
https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.linalg.qr.html
Its approximate inverse is only a proposed exact rational matrix C.
Lemma lem:contraction applies after ||I-C*DF||<1 and the strict image
inclusion have been proved on the entire cube."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import qr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("proposal", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    p = json.loads(args.proposal.read_text())
    jacobian = np.array(p["jacobian"])
    dimension = jacobian.shape[0]
    orthogonal, triangular, pivots = qr(jacobian, pivoting=True, mode="economic")
    selected = pivots[:dimension]
    inverse = np.linalg.inv(jacobian[:, selected])
    lines = ["GROUND_PRIMAL 1", str(p["beta"]), str(int(p["second"])),
             str(len(p["small_scales"]))]
    lines += [repr(float(s)) for s in p["small_scales"]]
    lines += [str(len(p["coefficients"]))]+p["coefficients"]
    lines += [str(dimension)]+[str(int(i)) for i in selected]
    lines += [repr(float(c)) for c in inverse.ravel()]
    args.output.write_text("\n".join(lines)+"\n")
    algebra={'jacobian':jacobian.tolist(),'QR_orthogonal':orthogonal.tolist(),
             'QR_triangular':triangular.tolist(),'QR_pivots':pivots.tolist(),
             'selected_columns':selected.tolist(),'inverse':inverse.tolist()}
    args.output.with_suffix('.algebra.txt').write_text(json.dumps(algebra,indent=2)+'\n')
    print(json.dumps({"selected_degrees": selected.tolist(),
                      "condition": float(np.linalg.cond(jacobian[:, selected]))}))


if __name__ == "__main__":
    main()
