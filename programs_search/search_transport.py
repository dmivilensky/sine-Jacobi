"""Propose storage coefficients on the common two-filter state region.

The canceled parametrization covers its boundary; both sharp conditional
moment endpoints are sampled. Lemma lem:regular-criterion supplies the
cubic conjugate and storage expression. The sampled maximum defines only
a search objective. SLSQP: Kraft, DFVLR-FB 88-28 (1988), implementation:
https://docs.scipy.org/doc/scipy-1.17.0/reference/optimize.minimize-slsqp.html
The native program must prove the inequality on the entire continuous
domain and enclose the spatial integral before a lower bound is used."""
from __future__ import annotations
from .common import cpu_slot, workers
from .search_trace import traced
import argparse
import json
from pathlib import Path
from itertools import combinations, product
import numpy as np
from numpy.polynomial import Legendre, Polynomial
from scipy.optimize import minimize
from .model import Reference, write_native_parameters
from .search_reference import K, EPS, log_cosh


def common_region(s1, s2, grid, centers=None):
    """Unit-square parametrization including all four edges and vertices.

    V=f(U)+T(U-f(U)), f(U)=(k+s1)U/(k+s2-(s2-s1)U).
    The two returned values of h/q are the sharp conditional endpoints of
    the first complementary moment. No endpoint division by U or 1-V is
    needed in these canceled formulas.
    """
    c, a, delta = K+s1, K+s2, s2-s1
    U, T = np.meshgrid(np.linspace(0, 1, grid), np.linspace(0, 1, grid))
    U, T = U.ravel(), T.ravel()
    if centers is not None:
        cu, ct = centers
        U = np.broadcast_to(U, (len(cu), len(U))).copy()
        T = np.broadcast_to(T, (len(cu), len(T))).copy()
        extra_u, extra_t = [cu], [ct]
        # The reference trajectory can lie much closer to the boundary than
        # a uniform grid resolves. Dyadic neighborhoods are generated from
        # that computed trajectory down to the square-root rounding scale.
        # These are search samples; their density proves no enclosure.
        for level in range(1, (np.finfo(float).nmant+1)//2+1):
            radius = 2.**(-level)
            for du in [-1, 0, 1]:
                for dt in [-1, 0, 1]:
                    if du == dt == 0:
                        continue
                    extra_u.append(np.clip(cu+du*radius, 0, 1))
                    extra_t.append(np.clip(ct+dt*radius, 0, 1))
        U = np.c_[U, np.array(extra_u).T]
        T = np.c_[T, np.array(extra_t).T]
    V = U*(c+delta*T*(1-U))/(a-delta*U)
    node_p = a*U*(1-T)/(a-delta*U*T)
    node_m = (c*U+a*T*(1-U))/(c+delta*T*(1-U))
    lo = K*(1-U)/(K+s1*node_p)
    hi = 1-c*U/(K+s1*node_m)
    return (np.concatenate([U, U], axis=-1), np.concatenate([V, V], axis=-1),
            np.concatenate([lo, hi], axis=-1), np.concatenate([T, T], axis=-1))


class Transport:
    def __init__(self, reference, scales, order, grid, degree, state_degree=2,
                 use_second_moment=True):
        self.ref = reference
        self.scales = np.asarray(scales, float)
        if len(scales) != 2 or not 0 < scales[0] < scales[1]:
            raise ValueError("two distinct ordered positive scales required")
        t, weights = reference.nodes(order)
        self.t = np.r_[t, t]
        self.q = np.r_[t**3, 2-t**3]
        self.Q = self.q*(2-self.q)
        n, speed = reference.local(self.t)
        # P=dx/dt=3*t^2/D. Use this positive coordinate identity to
        # compute D without subtracting K*n^2 and Q.
        self.n, self.d = n, 3*self.t**2/speed
        self.ell = self.Q/(K*n*n)
        self.weights = np.r_[3*t*t*weights, 3*t*t*weights]*n/K
        self.mass_term = 1-2*np.dot(weights, reference.local(t)[0]**2*reference.local(t)[1])
        q = self.q[:, None]
        Q = self.Q[:, None]
        amp = n[:, None]
        drift = self.d[:, None]
        vcoord = np.r_[t, 2-t]
        right, left, signals = [], [], []
        for s in scales:
            flow, H = reference.filter(s)
            right.append(flow.sol(2-vcoord)[0])
            left.append(flow.sol(vcoord)[0]/s)
            signals.append(log_cosh(np.sqrt(s))-H)
        center_u = np.clip(K*left[0]/self.q, 0, 1)
        center_v = np.clip(K*left[1]/self.q, 0, 1)
        s1, s2 = scales
        lower_v = (K+s1)*center_u/(K+s2-(s2-s1)*center_u)
        center_t = np.divide(center_v-lower_v, center_u-lower_v,
                             out=np.zeros_like(center_u), where=center_u>lower_v)
        U, V, mh, T = common_region(*scales, grid,
                                    centers=(center_u, np.clip(center_t, 0, 1)))
        self.U, self.V, self.mh, self.T = U, V, mh, T
        h = q*mh
        us = [q*U/K, q*V/K]
        linear, S = reference.filter(K, linear=True)
        right_linear = linear.sol(2-vcoord)[0, :, None]
        left_linear = linear.sol(vcoord)[0]
        self.signal = np.r_[signals, self.mass_term]
        shape = h.shape
        # There are three linear states (h,u0,u1), and total+1 monomials
        # of each state degree >=2, hence (d+1)(d+2)/2 columns per q basis.
        # Fill the final tensors directly; retaining every temporary column
        # and then stacking them would double this dominant allocation.
        columns = 3+int(use_second_moment)+degree*((state_degree+1)*(state_degree+2)//2)
        bp = np.empty((columns, *shape))
        cp = np.empty_like(bp)
        labels, reward, descriptions = [], [], []
        def append_direction(B, C, label, gain, description=None):
            index = len(labels)
            bp[index], cp[index] = B, C
            labels.append(label); reward.append(gain); descriptions.append(description)

        zeros = np.zeros(shape)
        for j, s in enumerate(scales):
            R = right[j][:, None]
            r = s*us[j]
            coefficient = -(s*Q+K*r*R)/(drift*(2+r+R))/amp
            append_direction(coefficient, zeros, "xi_"+str(j), signals[j])
        append_direction(Q/(drift*amp), zeros, "tau", -self.mass_term)
        if use_second_moment:
            A = q-h
            coefficient = K*(2*A*right_linear-Q*(A+right_linear))/(2*drift*amp)
            append_direction(coefficient, zeros, "chi", K*K/6-S)
        self.observables = len(labels)

        def append_potential(label, polynomial, Z, input_part, decay_part,
                             reference_Z, reference_Zq):
            ph = polynomial(self.q)[:, None]
            pd = polynomial.deriv()(self.q)[:, None]
            B = K*pd*Z+ph*input_part
            C = pd*drift*Z+ph*(input_part*amp*amp-decay_part)
            # Subtract the value of this potential on the local reference.
            # Its boundary values are both zero, so the exact price is
            # unchanged. The gauge derivative is computed for each basis.
            gauge = (polynomial.deriv()(self.q)*reference_Z
                     +polynomial(self.q)*reference_Zq)[:, None]
            append_direction((-B+K*gauge)/amp, (C-gauge*drift)/amp**3, label, 0.,
                             {"q_polynomial": polynomial.coef.tolist(),
                              "state_monomial": label.split(":")[0]})

        qpoly = Polynomial([0., 1.])
        # Lemma lem:regular-criterion requires (2-q)^2 to divide every
        # spatial polynomial, and an extra q for linear filter terms.
        # These factors enforce endpoint cancellation; the coefficients
        # multiplying the resulting basis functions are all optimized.
        wall = (2-qpoly)**2
        hbar = self.q-left_linear
        hbar_q = (self.q**2-2*hbar)/self.d
        left = np.asarray(left)
        left_q = np.array([(n*n-2*left[j]-scales[j]*left[j]**2)/self.d
                          for j in range(2)])
        for j in range(degree):
            # Orthogonal polynomials improve conditioning; the output gives
            # the complete ordinary monomial coefficients for hand checking.
            basis = Legendre.basis(j).convert(kind=Polynomial)(qpoly-1)
            ph = wall*basis
            append_potential("h:"+str(j), ph, h, zeros, 2*h-q*q,
                             hbar, hbar_q)
            for a in range(2):
                u = us[a]
                append_potential("u"+str(a)+":"+str(j), ph*qpoly,
                                 u, np.ones(shape), 2*u+scales[a]*u*u,
                                 left[a], left_q[a])
            for total in range(2, state_degree+1):
                for power0 in range(total+1):
                    power1 = total-power0
                    powers = (power0, power1)
                    Z = us[0]**power0*us[1]**power1
                    bv, dv = np.zeros(shape), np.zeros(shape)
                    Zbar = left[0]**power0*left[1]**power1
                    Zq = np.zeros_like(n)
                    for a in range(2):
                        if powers[a] == 0:
                            continue
                        exponents = list(powers); exponents[a] -= 1
                        derivative = powers[a]*us[0]**exponents[0]*us[1]**exponents[1]
                        bv += derivative
                        dv += derivative*(2*us[a]+scales[a]*us[a]**2)
                        Zq += (powers[a]*left[0]**exponents[0]*left[1]**exponents[1]
                               *left_q[a])
                    label = "u0^"+str(power0)+"u1^"+str(power1)+":"+str(j)
                    append_potential(label, ph, Z, bv, dv, Zbar, Zq)

        self.labels, self.descriptions = labels, descriptions
        if len(labels) != columns:
            raise ArithmeticError('storage basis dimension mismatch')
        # Column normalization depends solely on the computed design arrays.
        # Contract squares without allocating further design-sized arrays.
        # Normalizing in place keeps the denser search's memory proportional
        # to the two coefficient tensors themselves.
        column_norm = np.sqrt((np.einsum('ijk,ijk->i', bp, bp)
                               +np.einsum('ijk,ijk->i', cp, cp))/(bp.shape[1]*bp.shape[2]))
        if np.any(column_norm <= 0):
            raise ArithmeticError("zero search direction")
        self.column_scale = 1/column_norm
        bp *= self.column_scale[:, None, None]
        cp *= self.column_scale[:, None, None]
        self.bp, self.cp = bp, cp
        self.reward = np.array(reward)*self.column_scale
        self.objective_scale = 1/max(np.linalg.norm(self.reward), EPS)
        self.last = None
        self.best_value = np.inf
        self.best_x = None

    @traced
    def objective(self, x, details=False):
        if not details and self.last is not None and np.array_equal(x, self.last):
            return self.cached
        b = np.tensordot(x, self.bp, 1)
        c = np.tensordot(x, self.cp, 1)
        e = np.zeros(len(self.q))
        rows = np.arange(len(e))
        for iteration in range(np.finfo(float).nmant):
            z = b-e[:, None]
            # Lemma lem:conjugate fixes the normalized conjugate:
            # psi(z)=-z-1/2+(4/27)*(z+3/2)_+^3. Its expanded branch
            # avoids cancellation near zero; these are exact formula
            # coefficients, not proposed parameters of a free function.
            psi = np.where(z >= -1.5, z*z*(2/3+4*z/27), -z-.5)
            active = (psi-c).argmax(axis=1)
            za = z[rows, active]
            derivative = np.maximum(1+2*za/3, 0)**2-self.ell
            if np.any(derivative <= 0):
                # Outside the regular branch used by this proposal solver.
                # The rigorous verifier handles conjugate branches directly.
                penalty = self.objective_scale*(1+np.dot(x, x))
                return penalty, 2*self.objective_scale*x
            step = ((psi-c)[rows, active]-e*(1-self.ell))/derivative
            e += step
            if np.max(np.abs(step)) <= EPS**.75:
                break
        else:
            return self.objective_scale*(1+np.dot(x, x)), 2*self.objective_scale*x
        psi_prime = np.where(za >= -1.5, 4*za/3+4*za*za/9, -1.)
        partial = (self.bp[:, rows, active]*psi_prime[None, :]
                   -self.cp[:, rows, active])/derivative[None, :]
        value = self.weights@e-self.reward@x
        gradient = partial@self.weights-self.reward
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise ArithmeticError('nonfinite storage search objective')
        if value < self.best_value:
            self.best_value, self.best_x = float(value), x.copy()
        if details:
            return {"sampled_increment": float(-value), "price": e.tolist(),
                    "active_states": np.c_[self.q, self.U[rows, active], self.T[rows, active],
                                             self.mh[rows, active]].tolist(),
                    "gradient_norm": float(np.linalg.norm(gradient))}
        self.last = x.copy()
        self.cached = self.objective_scale*value, self.objective_scale*gradient
        return self.cached

    def optimize(self, iterations, initial=None, alternatives=()):
        self.best_value, self.best_x = np.inf, None
        self.last = None
        # Retain the zero potential and every admissible evaluated iterate.
        # SLSQP termination (or a new, larger basis) must not discard a
        # better seed on this very same sampling grid.
        self.objective(np.zeros(len(self.labels)))
        for seed in (initial, *alternatives):
            if seed is not None:
                self.objective(np.asarray(seed)/self.column_scale)
        if self.best_x is None:
            raise ArithmeticError('no admissible starting potential')
        x = self.best_x.copy()
        result = minimize(self.objective, x, jac=True, method="SLSQP",
                          options={"maxiter": iterations, "ftol": EPS**.625})
        self.objective(result.x)
        selected = self.best_x.copy()
        details = self.objective(selected, details=True)
        if not isinstance(details, dict):
            raise ArithmeticError("optimizer returned an inadmissible proposal")
        return {"status": "ordinary finite-grid proposal", "scales": self.scales.tolist(),
                "success": bool(result.success), "message": result.message,
                "iterations": int(result.nit), "labels": self.labels,
                "parameters": (selected*self.column_scale).tolist(),
                "retained_best_iterate": not np.array_equal(selected, result.x),
                "basis": self.descriptions, **details}


# Each spawned process owns its flow cache and complete objective trace.
# A cache hit changes no arithmetic; no optimizer or floating-point reduction
# is shared between candidates. Phase barriers preserve seed selection.
_worker_reference = None
_worker_parameters = None


def evaluate_candidate(task):
    from contextlib import redirect_stdout, redirect_stderr
    import os
    global _worker_reference, _worker_parameters
    index, parameters, work, scales, order, grid, degree, state_degree, second, iterations, starts = task
    directory = Path(work)/'candidates'/f'{index:04d}'
    directory.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get('SINEJACOBI_SEARCH_TRACE')
    os.environ['SINEJACOBI_SEARCH_TRACE'] = str(directory/'evaluations.jsonl')
    # Restarting an individual search recomputes this candidate in full.
    (directory/'evaluations.jsonl').write_text('')
    try:
        with cpu_slot(f'storage candidate {index} scales {scales} in {work}'):
            print('BEGIN storage candidate', index, 'scales', scales, flush=True)
            with (directory/'search.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
                if _worker_parameters != parameters:
                    _worker_reference = Reference(parameters)
                    _worker_parameters = parameters
                problem = Transport(_worker_reference, scales, order, grid, degree, state_degree, second)
                seeds = []
                for source in starts:
                    previous_coefficients = dict(zip(source['labels'], source['parameters']))
                    seeds.append(np.array([previous_coefficients.get(label, 0.) for label in problem.labels]))
                row = problem.optimize(iterations, alternatives=seeds)
                row.update({'order': order, 'grid': grid, 'degree': degree,
                            'state_degree': state_degree, 'trajectory_lower_bound': 'zero',
                            'second_moment_observable': second})
                print(json.dumps(row), flush=True)
            print('END storage candidate', index, 'increment', row['sampled_increment'], flush=True)
            return row
    finally:
        if previous is None:
            os.environ.pop('SINEJACOBI_SEARCH_TRACE', None)
        else:
            os.environ['SINEJACOBI_SEARCH_TRACE'] = previous


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--order", type=int, default=32)
    ap.add_argument("--grid", type=int, default=17)
    ap.add_argument("--degree", type=int, default=3,
                    help="number of Legendre polynomials in each spatial storage basis")
    ap.add_argument("--state-degree", type=int, default=2)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--scale-levels", type=int, default=6)
    ap.add_argument("--refine-scales", action="store_true",
                    help="refine neighborhoods of the two best dyadic pairs, then compare three finalists on a denser grid")
    ap.add_argument("--scales", nargs=2, type=float)
    ap.add_argument("--start", type=Path, action="append",
                    help="embed a previous potential; repeat to compare several seeds")
    ap.add_argument("--without-second-moment", action="store_true",
                    help="use determinant data only; the moment state remains available for storage")
    args = ap.parse_args()
    if min(args.order,args.grid)<2 or min(args.degree,args.state_degree,args.iterations)<1 or args.scale_levels<2:
        ap.error('positive polynomial degrees and iteration budget, order/grid/scale-levels >= 2 required')
    args.work.mkdir(parents=True, exist_ok=True)
    parameters = json.loads(args.reference.read_text())
    records, best = [], None
    initial = [json.loads(path.read_text()) for path in (args.start or [])]
    def retain(row):
        nonlocal best
        records.append(row)
        print(json.dumps({key: row[key] for key in ["scales", "order", "grid", "degree", "success", "iterations",
                                                   "sampled_increment", "gradient_norm"]}), flush=True)
        (args.work/"transport-search.json").write_text(json.dumps(records, indent=2)+"\n")
        if best is None or row["sampled_increment"] > best["sampled_increment"]:
            best = row
            (args.work/"transport.json").write_text(json.dumps(row, indent=2)+"\n")
            write_native_parameters(parameters, row["scales"], args.work/"reference.txt")
        return row

    choices = list([args.scales] if args.scales else combinations(2.**np.arange(args.scale_levels), 2))
    from concurrent.futures import ProcessPoolExecutor
    from contextlib import nullcontext
    import multiprocessing
    count = min(workers(), len(choices))
    context = (ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context('spawn'))
               if count > 1 else nullcontext(None))
    with context as pool:
        def phase(proposals, order, grid, iterations):
            tasks = [(len(records)+i, parameters, args.work, tuple(scales), order, grid,
                      args.degree, args.state_degree, not args.without_second_moment,
                      iterations, ([start] if isinstance(start, dict) else start) or [])
                     for i, (scales, start) in enumerate(proposals)]
            # Ordered map, stable ranking and strict comparisons also fix ties.
            # Scheduling and worker count cannot select different seeds.
            rows = pool.map(evaluate_candidate, tasks) if pool else map(evaluate_candidate, tasks)
            for row in rows:
                retain(row)
        phase([(scales, initial) for scales in choices], args.order, args.grid, args.iterations)
        if args.refine_scales and args.scales is None:
            # Local half-steps refine the two best coarse neighborhoods.
            # Generate all candidates before dispatch; completion order must
            # not choose a seed when these neighborhoods overlap.
            leaders = sorted(records, key=lambda row: -row['sampled_increment'])[:2]
            tried = {tuple(row['scales']) for row in records}
            proposals = []
            for leader in leaders:
                for offsets in product((-.5, 0., .5), repeat=2):
                    exponents = np.log2(leader['scales'])+offsets
                    if not (0 <= exponents[0] < exponents[1] <= args.scale_levels-1):
                        continue
                    scales = tuple(2.**exponents)
                    if scales in tried:
                        continue
                    tried.add(scales)
                    proposals.append((scales, leader))
            phase(proposals, args.order, args.grid, args.iterations)
            finalists = sorted(records, key=lambda row: -row['sampled_increment'])[:3]
            # All finalists are compared on one common finer grid.
            best = None
            phase([(row['scales'], row) for row in finalists], (3*args.order+1)//2,
                  (3*(args.grid-1)+1)//2+1, (3*args.iterations+1)//2)


if __name__ == "__main__":
    main()
