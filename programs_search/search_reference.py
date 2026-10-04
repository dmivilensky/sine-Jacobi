"""Search contact stationarity and fit a positive rational support.

With q=t^3 and z=n/t, k*z^3-3*t*(2-t^3)*z-2*p*(2-t^3)=0
(thm:contact). The BVP is fourth-order collocation with residual control:
Kierzenka and Shampine, ACM TOMS 27 (2001), 299--316,
DOI 10.1145/502800.502801. SciPy's residual normalization and singular term:
https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.integrate.solve_bvp.html
Positive rational fitting uses logarithmic coordinates and trust-region
least squares; Branch, Coleman and Li, SIAM J. Sci. Comput. 21 (1999), 1--23;
https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.optimize.least_squares.html
All search and collocation results are proposals for interval verification."""
from __future__ import annotations

from .search_trace import traced
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_simpson, quad, solve_bvp, solve_ivp
from scipy.optimize import brentq, least_squares, minimize_scalar
from scipy.special import roots_legendre

K = np.pi**2 / 4
EPS = np.finfo(float).eps


def cubic_positive(a, b):
    """Largest nonnegative root of x^3-a*x-b, for a,b>=0.

    sqrt(a) <= root <= sqrt(a)+cbrt(b). Newton from the upper endpoint
    decreases to the root: f'>0 and f''>=0 on that interval. Stopping at
    floating-point stagnation is a search criterion, never certification.
    """
    a, b = np.broadcast_arrays(a, b)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a < 0) or np.any(b < 0):
        raise ValueError("positive cubic requires a,b >= 0")
    lower = np.sqrt(a)
    x = lower + np.cbrt(b)
    for _ in range(np.finfo(float).nmant + 1):
        den = 3*x*x-a
        step = np.divide((x*x-a)*x-b, den,
                         out=np.zeros_like(x), where=den > 0)
        y = np.maximum(lower, x-step)
        if np.all(np.abs(y-x) <= EPS*np.maximum(1, np.abs(x))):
            return y
        x = y
    raise ArithmeticError("cubic iteration did not stagnate")


def shape_from_p(t, p):
    t = np.asarray(t)
    h = 2-t**3
    z = cubic_positive(3*t*h/K, 2*p*h/K)
    # The cubic equation gives K*z^2=3*t*h+2*p*h/z; this
    # positive expression agrees with the native regular coordinate.
    drift = 2*t*h+2*p*h/z
    if np.any(drift <= 0):
        raise ValueError("nonpositive coordinate drift")
    return t*z, 3/drift


@traced
def scalar_optimum(tol):
    @traced
    def mass(b):
        def integrand(t):
            n,speed=shape_from_p(t,b)
            return n*n*speed
        return 2*quad(integrand,
            0, 1, epsabs=tol, epsrel=tol)[0]
    lo = hi = 1.
    if mass(hi) > 1:
        while mass(hi) > 1:
            hi *= 2
    else:
        while mass(lo) < 1:
            lo /= 2
    # Safeguarded interpolation and bisection: Brent, Algorithms for
    # Minimization Without Derivatives (1973), SciPy implementation:
    # https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.optimize.brentq.html
    b = brentq(lambda b: mass(b)-1, lo, hi, xtol=tol)
    value = -b+2*b/K+9/K*quad(
        lambda t: t*t*shape_from_p(t, b)[0], 0, 1,
        epsabs=tol, epsrel=tol)[0]
    return b, value


def log_cosh(x):
    return np.logaddexp(x, -x)-np.log(2.)


def target_second(beta):
    if beta == 0:
        return K*K/6
    return K/beta-2*log_cosh(np.sqrt(beta*K))/beta**2


def vacuum_square(w, beta):
    if beta == 0:
        return w*w/4
    return (beta*w-2*np.log1p(beta*w/2))/beta**2


def scalar_filters(beta, b, t, tol):
    """Integrate both positive histories of the scalar equality profile."""
    def rhs(v, y):
        u = min(v, 2-v)
        n, speed = shape_from_p(u, b)
        q = u**3 if v <= 1 else 2-u**3
        w, r = y
        return speed*np.array([K*n*n-2*w-beta*w*w,
                               q*q+beta*w*w-2*r])
    sol = solve_ivp(rhs, (0, 2), [0., 0.], method="DOP853",
                    rtol=tol, atol=tol*EPS**.25, dense_output=True)
    if not sol.success:
        raise ArithmeticError(sol.message)
    r = sol.sol(t)[1]
    a = np.divide(r, t**3, out=np.zeros_like(t), where=t != 0)
    return a, sol.sol(2-t)[0]


@traced
def stationary_reference(beta, scalar_b, tol, previous=None):
    """Solve stationarity for mass and one exact spectral observation.

    Unknown multipliers (b,eta) have gamma=beta*eta for beta>0,
    and gamma=eta for the genuinely scale-free beta=0 specialization.
    Four unknown functions
    are a=(q-w)/q, the right filter v, accumulated mass, and accumulated
    squared filters. The singular term -3*a/t is given explicitly to the
    collocation solver; its boundary condition is a(0)=0.

    The result is merely a proposed reference. Positivity, contact, and all
    quantitative bounds require separate interval checks.
    """
    # The initial mesh size is derived from requested accuracy, then the
    # collocation solver refines it according to its residual estimate.
    solver_tol = max(tol, EPS**.75)
    gamma_scale = beta if beta > 0 else 1.
    exponent = max(1, int(np.ceil(-np.log2(solver_tol)/4)))
    t = np.linspace(0, 1, 2**exponent+1)

    def data(t, y, par):
        b, eta = par
        q = t**3
        h = 2-q
        Q = q*h
        a, v = y[:2]
        w = q*(1-a)
        ratio = (q*(h-v)+a*(2*v-Q))/(h*(2+beta*(w+v)))
        p = b+K*gamma_scale*eta*ratio
        # Collocation Newton steps may leave the admissible set. Extension
        # here is a proposal heuristic; final acceptance checks raw p>0.
        n, speed = shape_from_p(t, np.maximum(p, EPS))
        return q, w, v, n, speed, p

    def rhs(t, y, par):
        q, w, v, n, speed, p = data(t, y, par)
        return speed*np.array([
            q*(1+beta*(1-y[0])**2)-2*y[0],
            -K*n*n+2*v+beta*v*v,
            n*n, w*w+v*v])

    def boundary(ya, yb, par):
        return np.array([
            ya[0], yb[1]-(1-yb[0]), ya[2], yb[2]-.5,
            ya[3], yb[3]+vacuum_square(ya[1], beta)-target_second(beta)])

    if previous is None:
        a, v = scalar_filters(beta, scalar_b, t, tol)
        par = np.array([scalar_b, 0.])
        n, speed = shape_from_p(t, scalar_b)
        w = t**3*(1-a)
        y = np.vstack((a, v,
                       cumulative_simpson(n*n*speed, x=t, initial=0),
                       cumulative_simpson((w*w+v*v)*speed, x=t, initial=0)))
    else:
        y = previous.sol(t)
        par = previous.p.copy()
    singular = np.zeros((4, 4))
    singular[0, 0] = -3
    # Bound failed proposal work: tightening binary64 residual tolerances
    # can trigger refinement at the rounding floor. Exhausting this mesh
    # budget is a search failure, never evidence of mathematical infeasibility.
    sol = solve_bvp(rhs, boundary, t, y, p=par, S=singular,
                    tol=solver_tol, max_nodes=min(len(t)**2, 2**16))
    check_t = np.linspace(0, 1, 2**(exponent+2)+1)
    q, w, v, n, speed, p = data(check_t, sol.sol(check_t), sol.p)
    # Composite Gaussian quadrature respects every collocation breakpoint.
    # Agreement after order doubling is only a proposal accuracy check;
    # the independent interval calculation still supplies the proof.
    previous_integrals = None
    for quadrature_order in (4, 8, 16, 32, 64):
        nodes, weights = roots_legendre(quadrature_order)
        widths = np.diff(sol.x)/2
        qt = ((sol.x[:-1]+sol.x[1:])[:, None]/2
              +widths[:, None]*nodes).ravel()
        qw = (widths[:, None]*weights).ravel()
        _, _, _, qn, qp, _ = data(qt, sol.sol(qt), sol.p)
        integrals = 2*np.array([qw@(qn**3*qp), qw@(qn*qp)])
        agreement = (float(np.max(np.abs(integrals-previous_integrals)))
                     if previous_integrals is not None else np.inf)
        if agreement <= tol:
            break
        previous_integrals = integrals
    else:
        raise ArithmeticError('stationary quadrature failed to stabilize')
    row = {
        "status": "ordinary stationary proposal",
        "beta": float(beta), "b": float(sol.p[0]),
        "gamma": float(gamma_scale*sol.p[1]), "eta": float(sol.p[1]),
        "solver_success": bool(sol.success), "solver_message": sol.message,
        "solver_tolerance": solver_tol,
        "nodes": len(sol.x), "minimum_sampled_p": float(p.min()),
        "minimum_sampled_p_increment": float(np.diff(p).min()),
        "energy": float(integrals[0]),
        "root_mass": float(integrals[1]),
        "quadrature_order": quadrature_order,
        "quadrature_agreement": agreement,
        "boundary_residual": boundary(sol.sol(0), sol.sol(1), sol.p).tolist(),
    }
    row["admissible_proposal"] = bool(sol.success and np.min(sol.p) >= 0
                                       and np.min(p) > 0)
    return sol, row, (check_t, p, n, speed, w, v)


@traced
def fit_positive_rational(t, p, terms, tol):
    """Fit p0+sum a_j*Q/(Q+c_j), a_j,c_j,p0>0.

    Pole locations are initialized from quantiles of the observed positive
    variation dp. Optimization then varies every pole and amplitude in
    logarithmic coordinates; the exact positive profile is verified later.
    """
    Q = t**3*(2-t**3)
    variation = np.maximum(np.diff(p), 0)
    cumulative = np.r_[0., np.cumsum(variation)]
    if cumulative[-1] <= 0:
        return [float(p[0])], 0.
    fractions = np.arange(1, terms+1)/(terms+1)
    poles = np.interp(fractions*cumulative[-1], cumulative, Q)
    poles = np.maximum(poles, EPS)
    amplitudes = np.full(terms, cumulative[-1]/terms)
    x0 = np.log(np.r_[max(p[0], EPS), amplitudes, poles])
    weights = np.sqrt(np.r_[np.diff(t), t[-1]-t[-2]])

    def unpack(x):
        v = np.exp(x)
        return v[0], v[1:terms+1], v[terms+1:]

    @traced
    def residual(x):
        p0, a, c = unpack(x)
        pred = p0+np.sum(a[:, None]*Q/(Q+c[:, None]), axis=0)
        return (pred-p)*weights

    @traced
    def jacobian(x):
        # In logarithmic coordinates the three derivative blocks are
        # p0, a_j Q/(Q+c_j), and -a_j c_j Q/(Q+c_j)^2.
        p0, a, c = unpack(x)
        terms = a[:, None]*Q/(Q+c[:, None])
        return np.column_stack((np.full_like(Q, p0), terms.T,
                                (-terms*c[:, None]/(Q+c[:, None])).T))*weights[:, None]

    fit = least_squares(residual, x0, jac=jacobian, xtol=tol, ftol=tol, gtol=tol,
                        max_nfev=None)
    p0, a, c = unpack(fit.x)
    pairs = sorted(zip(a, c), key=lambda pair: pair[1])
    coefficients = [float(p0)]+[float(v) for pair in pairs for v in pair]
    return coefficients, float(np.linalg.norm(residual(fit.x)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--digits", type=int, default=8)
    ap.add_argument("--maximum-doublings", type=int, default=None)
    ap.add_argument("--canonical", action="store_true",
                    help="use only the ground state and second moment; beta is exactly zero")
    args = ap.parse_args()
    if not 6<=args.digits<=13 or (args.maximum_doublings is not None and args.maximum_doublings<1):
        ap.error('use 6..13 search digits and a positive doubling budget')
    args.work.mkdir(parents=True, exist_ok=True)
    tol = 10.**(-args.digits)
    b, value = scalar_optimum(tol)
    (args.work/"scalar-search.json").write_text(json.dumps(
        {"status": "ordinary search", "b": b, "value": value}, indent=2)+"\n")
    print("SCALAR", b, value, flush=True)
    if args.canonical:
        sol, row, arrays = stationary_reference(0., b, tol)
        if not row['admissible_proposal']:
            raise ArithmeticError('canonical stationary proposal failed: '+str(row))
        (args.work/'best-stationary-search.json').write_text(json.dumps(row, indent=2)+'\n')
        (args.work/'scale-search.json').write_text(json.dumps([row], indent=2)+'\n')
        np.savez(args.work/'best-stationary-search.npz', t=arrays[0], p=arrays[1],
                 n=arrays[2], speed=arrays[3], w=arrays[4], v=arrays[5],
                 bvp_x=sol.x, bvp_y=sol.y, parameters=sol.p)
        print(json.dumps(row), flush=True)
        return
    beta = 1.
    previous = None
    best = None
    rows = []
    solutions = []
    for level in range(args.maximum_doublings or np.finfo(float).nmant):
        try:
            sol, row, arrays = stationary_reference(beta, b, max(tol, np.sqrt(EPS)), previous)
        except (ValueError, ArithmeticError, FloatingPointError) as exc:
            row = {"beta": beta, "admissible_proposal": False,
                   "error": str(exc)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        (args.work/"scale-search.json").write_text(json.dumps(rows, indent=2)+"\n")
        if row["admissible_proposal"]:
            previous = sol
            solutions.append((beta, sol))
            if best is None or row["energy"] > best[0]["energy"]:
                best = (row, arrays, sol)
            np.savez(args.work/("stationary-"+str(level)+".npz"),
                     t=arrays[0], p=arrays[1], n=arrays[2], speed=arrays[3],
                     w=arrays[4], v=arrays[5], bvp_x=sol.x, bvp_y=sol.y,
                     parameters=sol.p)
        elif best is not None:
            break
        beta *= 2
    if best is not None:
        coarse = list(solutions)

        def objective(log_beta):
            nonlocal best
            trial = np.exp(log_beta)
            nearest = min(solutions, key=lambda item: abs(np.log(item[0]/trial)))
            try:
                sol, row, arrays = stationary_reference(trial, b, tol, nearest[1])
            except (ValueError, ArithmeticError, FloatingPointError) as exc:
                row = {'beta': float(trial), 'admissible_proposal': False, 'error': str(exc)}
                rows.append(row)
                print('REFINE', json.dumps(row), flush=True)
                return 1.
            rows.append(row)
            print("REFINE", json.dumps(row), flush=True)
            if not row["admissible_proposal"]:
                return 1.
            solutions.append((trial, sol))
            if best is None or row["energy"] > best[0]["energy"]:
                best = (row, arrays, sol)
            return -row["energy"]

        # Half-steps in log(beta) resolve the successful doubling range,
        # including the interval up to its first failed endpoint. Recompute
        # the three coarse leaders at the fine tolerance before comparing
        # them with fine candidates: a coarse integration error cannot win.
        leaders = sorted((row for row in rows if row.get('admissible_proposal')),
                         key=lambda row: -row['energy'])[:3]
        best = None
        for row in leaders:
            objective(np.log(row['beta']))
        endpoints = sorted(set([value for value, _ in coarse]+[beta]))
        for left, right in zip(endpoints[:-1], endpoints[1:]):
            objective((np.log(left)+np.log(right))/2)
        if best is not None:
            fine_points = sorted(set(row['beta'] for row in rows
                                     if row.get('admissible_proposal')))
            center = best[0]['beta']
            lower = max((v for v in fine_points if v < center), default=center)
            upper = min((v for v in fine_points if v > center), default=beta)
            # Refine around the best observed value, rather than the last
            # successful doubling. This remains a local proposal search;
            # it does not assert global optimality over beta or BVP branches.
            if lower < upper:
                minimize_scalar(objective, bounds=(np.log(lower), np.log(upper)),
                                method="bounded", options={"xatol": tol**.5})
        (args.work/"scale-search.json").write_text(json.dumps(rows, indent=2)+"\n")
    if best is not None:
        row, arrays, sol = best
        (args.work/"best-stationary-search.json").write_text(json.dumps(row, indent=2)+"\n")
        np.savez(args.work/"best-stationary-search.npz", t=arrays[0], p=arrays[1],
                 n=arrays[2], speed=arrays[3], w=arrays[4], v=arrays[5],
                 bvp_x=sol.x, bvp_y=sol.y, parameters=sol.p)
    else:
        raise ArithmeticError('no admissible stationary reference was found')


if __name__ == "__main__":
    main()
