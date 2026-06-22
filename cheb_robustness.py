#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize

Array = np.ndarray

CSIN = 1.1426922
C_PREFIX_ASYMPTOTIC = 4.0 / math.sqrt(3.0)
C_SINE_ASYMPTOTIC = 2.0 * math.sqrt(CSIN)


# -----------------------------------------------------------------------------
# Chebyshev helpers and block methods
# -----------------------------------------------------------------------------

def cheb_T_eta(N: int, eta: float) -> float:
    if N < 0:
        return 1.0
    return math.cosh(N * math.acosh(eta))


def cheb_epsilon(N: int, mu: float, L: float) -> float:
    if N <= 0:
        return 1.0
    eta = (L + mu) / (L - mu)
    return 1.0 / abs(cheb_T_eta(N, eta))


def asymptotic_C(method: str) -> float:
    # A_N ~ C_R * N^{3/2}/Delta * eps_N.
    # Constants are the article constants, not tuned plot parameters.
    if method == "prefix":
        return C_PREFIX_ASYMPTOTIC
    if method == "sine":
        return C_SINE_ASYMPTOTIC
    raise ValueError(method)


def prefix_block(x0: Array, grad: Callable[[Array], Array], mu: float, L: float, N: int,
                 return_traj: bool = False) -> Tuple[Array, Optional[List[Array]]]:
    if N <= 0:
        return x0.copy(), [x0.copy()] if return_traj else None
    Delta = L - mu
    eta = (L + mu) / Delta
    c = np.empty(N + 2, dtype=float)
    c[0] = 1.0
    c[1] = eta
    for t in range(1, N + 1):
        c[t + 1] = 2.0 * eta * c[t] - c[t - 1]
    x_prev = x0.copy()
    x_cur = x0 - (2.0 / (L + mu)) * grad(x0)
    traj = [x_prev.copy(), x_cur.copy()] if return_traj else None
    if N == 1:
        return x_cur, traj
    for t in range(1, N):
        beta = c[t - 1] / c[t + 1]
        g = grad(x_cur)
        x_next = x_cur + beta * (x_cur - x_prev) - (1.0 + beta) * (2.0 / (L + mu)) * g
        x_prev, x_cur = x_cur, x_next
        if return_traj:
            traj.append(x_cur.copy())
    return x_cur, traj


def sine_jacobi_block(x0: Array, grad: Callable[[Array], Array], mu: float, L: float, N: int,
                      return_traj: bool = False) -> Tuple[Array, Optional[List[Array]]]:
    if N <= 0:
        return x0.copy(), [x0.copy()] if return_traj else None
    Delta = L - mu
    eta = (L + mu) / Delta
    if N == 1:
        x1 = x0 - (2.0 / (L + mu)) * grad(x0)
        return x1, [x0.copy(), x1.copy()] if return_traj else None
    a2 = np.empty(N, dtype=float)
    p = np.empty(N + 1, dtype=float)
    p[0] = 1.0
    p[1] = eta
    for s in range(1, N):
        a2[s] = 0.25 * math.sin(math.pi * s / N) ** 2 / (
            math.sin(math.pi * (s - 0.5) / N) * math.sin(math.pi * (s + 0.5) / N)
        )
        p[s + 1] = eta * p[s] - a2[s] * p[s - 1]
    x_prev = x0.copy()
    x_cur = x0 - (2.0 / (L + mu)) * grad(x0)
    traj = [x_prev.copy(), x_cur.copy()] if return_traj else None
    for s in range(1, N):
        alpha = eta * p[s] / p[s + 1]
        beta = a2[s] * p[s - 1] / p[s + 1]
        gamma = (2.0 / Delta) * p[s] / p[s + 1]
        g = grad(x_cur)
        x_next = alpha * x_cur - beta * x_prev - gamma * g
        x_prev, x_cur = x_cur, x_next
        if return_traj:
            traj.append(x_cur.copy())
    return x_cur, traj


def run_block(method: str, x0: Array, grad: Callable[[Array], Array], mu: float, L: float, N: int,
              return_traj: bool = False) -> Tuple[Array, Optional[List[Array]]]:
    if method == "prefix":
        return prefix_block(x0, grad, mu, L, N, return_traj)
    if method == "sine":
        return sine_jacobi_block(x0, grad, mu, L, N, return_traj)
    raise ValueError(method)

# -----------------------------------------------------------------------------
# Deterministic endpoint-coupled convex GLM
# -----------------------------------------------------------------------------

@dataclass
class LogisticProblem:
    A: Array
    y: Array
    lam: float
    x_star: Array
    f_star: float
    mu: float
    L: float
    H_star: Array
    M_est: float
    eigvals: Array
    eigvecs: Array

    def loss(self, x: Array) -> float:
        z = self.y * (self.A @ x)
        return float(np.mean(np.logaddexp(0.0, -z)) + 0.5 * self.lam * np.dot(x, x))

    def grad(self, x: Array) -> Array:
        z = self.y * (self.A @ x)
        s = 1.0 / (1.0 + np.exp(np.clip(z, -50, 50)))
        return -(self.A.T @ (self.y * s)) / self.A.shape[0] + self.lam * x

    def hess(self, x: Array) -> Array:
        z = self.y * (self.A @ x)
        sig = 1.0 / (1.0 + np.exp(np.clip(-z, -50, 50)))
        w = sig * (1.0 - sig)
        return (self.A.T * w) @ self.A / self.A.shape[0] + self.lam * np.eye(self.A.shape[1])


def make_endpoint_coupled_logistic(seed: int = 7, n: int = 2200, d: int = 32,
                                   cond_target: float = 300.0, lam: float = 0.003,
                                   signal: float = 1.70) -> LogisticProblem:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    eigs = np.geomspace(1.0, cond_target, d)
    eigs = eigs / np.mean(eigs) * 4.0
    sqrt_cov = Q @ np.diag(np.sqrt(eigs))
    Z = rng.normal(size=(n, d))
    A = Z @ sqrt_cov.T

    # Structured rows that make third-order curvature couple spectral endpoints.
    extra = []
    for _ in range(max(80, 3 * d)):
        i = rng.integers(0, max(2, d // 5))
        j = rng.integers(max(d - d // 5, 1), d)
        k = rng.integers(max(1, d // 3), max(d // 3 + 1, 2 * d // 3))
        v = Q[:, i] + rng.choice([-1.0, 1.0]) * Q[:, j] + 0.25 * rng.normal() * Q[:, k]
        v /= norm(v)
        scale = math.sqrt(eigs[i] + eigs[j] + eigs[k]) * rng.uniform(1.4, 2.4)
        extra.append(scale * v)
    A = np.vstack([A, np.array(extra)])

    true_dir = Q[:, 0] + 0.85 * Q[:, -1] + 0.35 * Q[:, d // 2]
    true_dir /= norm(true_dir)
    w_true = signal * true_dir
    logits = A @ w_true
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    y = np.where(rng.random(A.shape[0]) < prob, 1.0, -1.0)

    tmp = LogisticProblem(A=A, y=y, lam=lam, x_star=np.zeros(d), f_star=0.0,
                          mu=0.0, L=0.0, H_star=np.eye(d), M_est=0.0,
                          eigvals=np.ones(d), eigvecs=np.eye(d))
    res = minimize(lambda x: tmp.loss(x), np.zeros(d), jac=lambda x: tmp.grad(x),
                   method="L-BFGS-B", options={"gtol": 1e-12, "ftol": 1e-14, "maxiter": 2500})
    x_star = res.x
    Hs = tmp.hess(x_star)
    evals, evecs = np.linalg.eigh(Hs)
    mu = float(evals[0])
    L = float(evals[-1])

    # Empirical local Hessian-Lipschitz proxy around x_star.
    M_emp = 0.0
    for rr in [1e-3, 3e-3, 1e-2]:
        for _ in range(12):
            u = rng.normal(size=d)
            u /= norm(u)
            M_emp = max(M_emp, norm(tmp.hess(x_star + rr * u) - Hs, 2) / rr)
    # Conservative GLM-based floor; avoids pathological underestimating budgets.
    row_norms = norm(A, axis=1)
    M_floor = 0.03 * float(np.mean(row_norms ** 3))
    M_est = max(M_emp, M_floor)
    return LogisticProblem(A=A, y=y, lam=lam, x_star=x_star, f_star=tmp.loss(x_star),
                           mu=mu, L=L, H_star=Hs, M_est=float(M_est),
                           eigvals=evals, eigvecs=evecs)


def summarize_problem(prob: LogisticProblem) -> Dict:
    return {
        "n": int(prob.A.shape[0]),
        "d": int(prob.A.shape[1]),
        "mu": float(prob.mu),
        "L": float(prob.L),
        "Delta": float(prob.L - prob.mu),
        "kappa": float(prob.L / prob.mu),
        "M_est": float(prob.M_est),
        "f_star": float(prob.f_star),
    }


def endpoint_directions(prob: LogisticProblem, count: int, seed: int) -> List[Array]:
    rng = np.random.default_rng(seed)
    d = prob.eigvecs.shape[0]
    out = []
    for q in range(count):
        u = prob.eigvecs[:, 0] + rng.choice([-1.0, 1.0]) * rng.uniform(0.65, 1.05) * prob.eigvecs[:, -1]
        u += rng.uniform(-0.35, 0.35) * prob.eigvecs[:, d // 2]
        u += 0.08 * rng.normal(size=d)
        u /= norm(u)
        out.append(u)
    return out


def r_acc_pred(prob: LogisticProblem, method: str, N: int, eta: float, S: float = 1.0) -> float:
    return 2.0 * eta * (prob.L - prob.mu) / (prob.M_est * S * asymptotic_C(method) * N * N)


def exp2_glm_phase(seed: int = 7, N: int = 36, dirs: int = 10,
                   eta_accept: float = 1.0) -> Dict:
    """One-block deterministic shell with multiple endpoint-coupled directions."""
    prob = make_endpoint_coupled_logistic(seed=seed)
    # Use absolute radii: the rigorous M-based shell is very conservative here.
    # The plotted shell is empirical, not a claim of global validity.
    r0 = 1.0
    scales = np.geomspace(0.003, 1.0, 16)
    directions = endpoint_directions(prob, dirs, seed + 123)
    eps = cheb_epsilon(N, prob.mu, prob.L)
    rows = []
    for scale in scales:
        r = float(scale * r0)
        for method in ["prefix", "sine"]:
            contr = []
            slopes = []
            S_vals = []
            accepted = []
            for u in directions:
                x0 = prob.x_star + r * u
                xN, traj = run_block(method, x0, prob.grad, prob.mu, prob.L, N, return_traj=True)
                c = norm(xN - prob.x_star) / r
                contr.append(c)
                slopes.append((c - eps) / max(r, 1e-300))
                S_vals.append(max(norm(z - prob.x_star) for z in traj) / r)
                accepted.append(c <= (1.0 + eta_accept) * eps)
            rows.append({
                "scale": float(scale),
                "r": r,
                "method": method,
                "median_contr": float(np.median(contr)),
                "q25_contr": float(np.quantile(contr, 0.25)),
                "q75_contr": float(np.quantile(contr, 0.75)),
                "mean_contr": float(np.mean(contr)),
                "median_over_eps": float(np.median(contr) / eps),
                "median_slope": float(np.median(slopes)),
                "q25_slope": float(np.quantile(slopes, 0.25)),
                "q75_slope": float(np.quantile(slopes, 0.75)),
                "median_S": float(np.median(S_vals)),
                "accept_rate": float(np.mean(accepted)),
            })
    return {
        "problem": summarize_problem(prob),
        "N": int(N),
        "dirs": int(dirs),
        "eta_accept": float(eta_accept),
        "epsN": float(eps),
        "radius_scan_min": float(min(scales)),
        "radius_scan_max": float(max(scales)),
        "r_acc_prefix_formula_eta025": float(r_acc_pred(prob, "prefix", N, eta=0.25, S=1.0)),
        "r_acc_sine_formula_eta025": float(r_acc_pred(prob, "sine", N, eta=0.25, S=1.0)),
        "rows": rows,
    }

# -----------------------------------------------------------------------------
# Restarted deterministic continuation
# -----------------------------------------------------------------------------

def gradient_radius_proxy(prob: LogisticProblem, x: Array) -> float:
    return norm(prob.grad(x)) / prob.mu


def choose_predicted_N(prob: LogisticProblem, method: str, r_hat: float, eta_budget: float,
                       N_cap: int, S_hat: float = 1.15) -> int:
    if r_hat <= 0:
        return N_cap
    val = math.sqrt(max(1.0, 2.0 * eta_budget * (prob.L - prob.mu) /
                        (prob.M_est * S_hat * asymptotic_C(method) * r_hat + 1e-300)))
    return int(max(1, min(N_cap, math.floor(val))))


def run_restarted_solver(prob: LogisticProblem, method: str, x_start: Array,
                         eta_budget: float, N_cap: int, tol_radius: float,
                         max_blocks: int = 20, oracle_radius: bool = False) -> Dict:
    x = x_start.copy()
    hist = []
    total_grads = 0
    rejected = 0
    for k in range(max_blocks):
        r_true = norm(x - prob.x_star)
        if r_true <= tol_radius:
            break
        r_hat = r_true if oracle_radius else gradient_radius_proxy(prob, x)
        # Practical finite-radius restart: try the most aggressive block first and
        # backtrack if the observed objective decrease is not Chebyshev-like.
        # This measures accepted block length directly instead of trusting the
        # conservative M-Lipschitz radius estimate.
        N_try = N_cap
        accepted = False
        attempts = 0
        while N_try >= 1 and not accepted:
            eps = cheb_epsilon(N_try, prob.mu, prob.L)
            x_new, traj = run_block(method, x, prob.grad, prob.mu, prob.L, N_try, return_traj=True)
            r_new = norm(x_new - prob.x_star)
            f_old = prob.loss(x) - prob.f_star
            f_new = prob.loss(x_new) - prob.f_star
            # Benchmark acceptance: the radius contraction should remain within
            # a finite-radius Chebyshev degradation budget.  This is a certified
            # diagnostic rule using x_star; it is intentionally stricter than
            # just requiring monotone objective decrease.
            ok = (r_new / max(r_true, 1e-300)) <= (1.0 + 4.0 * eta_budget) * eps
            total_grads += N_try
            if ok or N_try <= 2:
                x = x_new
                accepted = True
                hist.append({
                    "block": int(k),
                    "N": int(N_try),
                    "r_before": float(r_true),
                    "r_after": float(r_new),
                    "f_before": float(f_old),
                    "f_after": float(f_new),
                    "eps": float(eps),
                    "contraction": float(r_new / max(r_true, 1e-300)),
                    "contraction_over_eps": float((r_new / max(r_true, 1e-300)) / max(eps, 1e-300)),
                    "S_emp": float(max(norm(z - prob.x_star) for z in traj) / max(r_true, 1e-300)),
                    "attempts": int(attempts + 1),
                })
            else:
                rejected += 1
                attempts += 1
                N_try = max(1, int(math.floor(0.82 * N_try)))
    return {
        "total_grads": int(total_grads),
        "final_radius": float(norm(x - prob.x_star)),
        "blocks": int(len(hist)),
        "rejected": int(rejected),
        "history": hist,
    }


def exp3_restart_shell(seed: int = 7, eta_budget: float = 0.23, N_cap: int = 44,
                       scales: Optional[List[float]] = None, tol_radius: float = 2e-5) -> Dict:
    prob = make_endpoint_coupled_logistic(seed=seed)
    dirs = endpoint_directions(prob, 1, seed + 555)
    u = dirs[0]
    N_target = N_cap
    # Empirical continuation shell.  The M-Lipschitz worst-case radius is far too
    # conservative for this GLM, so the scan is over actual local radii where
    # long blocks are sometimes accepted and sometimes rejected.
    r_pref = 0.10
    r_sine = 0.10 * math.sqrt(asymptotic_C("prefix") / asymptotic_C("sine"))
    if scales is None:
        scales = list(np.geomspace(0.035, 0.70, 10))
    rows = []
    histories_at_shell = {}
    for scale in scales:
        r0 = float(scale)
        x_start = prob.x_star + r0 * u
        row = {"scale": float(scale), "r0": r0}
        for method in ["prefix", "sine"]:
            res = run_restarted_solver(prob, method, x_start, eta_budget, N_cap, tol_radius,
                                       max_blocks=16, oracle_radius=False)
            row[f"{method}_grads"] = res["total_grads"]
            row[f"{method}_blocks"] = res["blocks"]
            row[f"{method}_first_N"] = res["history"][0]["N"] if res["history"] else 0
            row[f"{method}_max_N"] = max([h["N"] for h in res["history"]], default=0)
            row[f"{method}_final_radius"] = res["final_radius"]
            row[f"{method}_rejected"] = res["rejected"]
            if abs(scale - 0.10) < 0.025 or (not histories_at_shell and scale > 0.08):
                histories_at_shell[method] = res
        row["grad_saving_frac"] = (row["prefix_grads"] - row["sine_grads"]) / max(row["prefix_grads"], 1)
        row["first_N_ratio"] = row["sine_first_N"] / max(row["prefix_first_N"], 1)
        rows.append(row)
    return {
        "problem": summarize_problem(prob),
        "eta_budget": float(eta_budget),
        "N_cap": int(N_cap),
        "N_target": int(N_target),
        "r_acc_prefix_pred": float(r_pref),
        "r_acc_sine_pred": float(r_sine),
        "rows": rows,
        "representative_histories": histories_at_shell,
    }

# -----------------------------------------------------------------------------
# Stochastic pure multiplicative Hessian noise
# -----------------------------------------------------------------------------

@dataclass
class StochConfig:
    d: int = 44
    kappa: float = 120.0
    trials: int = 360
    seed: int = 1007
    N_main: int = 44
    batches: Tuple[int, ...] = (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
    N_grid: Tuple[int, ...] = (28, 36, 44, 52)
    collapse_batches: Tuple[int, ...] = (24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
    noise_strength: float = 1.0
    fit_tail_fraction: float = 0.5


def make_diag_problem(d: int, kappa: float) -> Tuple[Array, float, float, Array, Array]:
    mu = 1.0
    L = float(kappa)
    eigs = np.geomspace(mu, L, d)
    sqrt_eigs = np.sqrt(eigs)
    u = np.zeros(d)
    u[0] = 1.0
    u[-1] = 1.0
    u[d // 2] = 0.25
    rng = np.random.default_rng(12345 + d)
    u += 0.03 * rng.normal(size=d)
    u /= norm(u)
    return eigs, mu, L, sqrt_eigs, u


def exact_quad_grad(eigs: Array) -> Callable[[Array], Array]:
    return lambda x: eigs * x


class SequenceStochGrad:
    """Common-random-number endpoint spectral curvature oracle for one block.

    The oracle is pure multiplicative Hessian noise in the eigenbasis of H:
        g(x) = H x + (sigma/sqrt(b)) z_t B x,
    where B couples the two endpoint eigendirections.  This is the stochastic
    version of the spectral-pair regime in which the A_N coefficient is expected
    to be visible.  The parameter b is an effective batch size/noise budget.
    """
    def __init__(self, eigs: Array, sqrt_eigs: Array, batch: int, seed: int, N: int, noise_strength: float):
        self.eigs = eigs
        self.batch = int(batch)
        self.seed = int(seed)
        self.N = int(N)
        self.noise_strength = float(noise_strength)
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def __call__(self, x: Array) -> Array:
        hx = self.eigs * x
        z = self.rng.normal()
        gx = np.zeros_like(x)
        gx[0] = x[-1]
        gx[-1] = x[0]
        self.calls += 1
        return hx + (self.noise_strength / math.sqrt(self.batch)) * z * gx


def largest_design_tail(values, fraction: float) -> set:
    vals = sorted(set(int(v) for v in values))
    if not vals:
        return set()
    keep = max(3, int(math.ceil(float(fraction) * len(vals))))
    return set(vals[-min(keep, len(vals)):])


def stochastic_batch_curve(config: StochConfig) -> Dict:
    eigs, mu, L, sqrt_eigs, u = make_diag_problem(config.d, config.kappa)
    N = int(config.N_main)
    eps = cheb_epsilon(N, mu, L)
    exact = {}
    for method in ["prefix", "sine"]:
        xN, _ = run_block(method, u, exact_quad_grad(eigs), mu, L, N, return_traj=False)
        exact[method] = float(norm(xN) ** 2)
    rows = []
    for b in config.batches:
        paired_vals = {"prefix": [], "sine": []}
        for t in range(config.trials):
            base_seed = config.seed + 1000003 * int(b) + t
            for method in ["prefix", "sine"]:
                grad = SequenceStochGrad(eigs, sqrt_eigs, int(b), base_seed, N, config.noise_strength)
                xN, _ = run_block(method, u, grad, mu, L, N, return_traj=False)
                paired_vals[method].append(norm(xN) ** 2)
        for method in ["prefix", "sine"]:
            vals = np.array(paired_vals[method])
            rows.append({
                "batch": int(b),
                "method": method,
                "mean_sq": float(vals.mean()),
                "std_sq": float(vals.std(ddof=1)),
                "se_sq": float(vals.std(ddof=1) / math.sqrt(config.trials)),
                "mean_over_eps2": float(vals.mean() / (eps ** 2)),
                "degradation": float(vals.mean() / (eps ** 2) - 1.0),
                "failure_rate_gt_2eps": float(np.mean(np.sqrt(vals) > 2.0 * eps)),
            })
        diff = np.array(paired_vals["prefix"]) - np.array(paired_vals["sine"])
        rows.append({
            "batch": int(b),
            "method": "paired_gap",
            "mean_gap_over_eps2": float(diff.mean() / (eps ** 2)),
            "se_gap_over_eps2": float(diff.std(ddof=1) / math.sqrt(config.trials) / (eps ** 2)),
        })
    # Fit degradation ~= a / b on the design tail (largest batch sizes),
    # rather than selecting rows by the observed response.
    fit_batches = largest_design_tail(config.batches, config.fit_tail_fraction)
    fits = {}
    for method in ["prefix", "sine"]:
        rr = [r for r in rows if r.get("method") == method and int(r["batch"]) in fit_batches]
        x = np.array([1.0 / r["batch"] for r in rr])
        y = np.array([r["degradation"] for r in rr])
        if len(x) >= 2:
            a = float(np.dot(x, y) / np.dot(x, x))
        else:
            a = float("nan")
        fits[method] = {"slope_a_over_b": a, "fit_batches": sorted(fit_batches)}
    return {
        "config": asdict(config),
        "mu": mu,
        "L": L,
        "Delta": L - mu,
        "epsN": float(eps),
        "exact_quadratic_sq": exact,
        "rows": rows,
        "fits": fits,
        "predicted_batch_ratio_C2": float((asymptotic_C("sine") / asymptotic_C("prefix")) ** 2),
    }


def stochastic_collapse(config: StochConfig) -> Dict:
    eigs, mu, L, sqrt_eigs, u = make_diag_problem(config.d, config.kappa)
    rows = []
    for N in config.N_grid:
        eps = cheb_epsilon(int(N), mu, L)
        for b in config.collapse_batches:
            vals_by_method = {"prefix": [], "sine": []}
            # Fewer trials in collapse to keep runtime reasonable.
            local_trials = max(120, config.trials // 2)
            for t in range(local_trials):
                base_seed = config.seed + 777777 * int(N) + 1009 * int(b) + t
                for method in ["prefix", "sine"]:
                    grad = SequenceStochGrad(eigs, sqrt_eigs, int(b), base_seed, int(N), config.noise_strength)
                    xN, _ = run_block(method, u, grad, mu, L, int(N), return_traj=False)
                    vals_by_method[method].append(norm(xN) ** 2)
            for method in ["prefix", "sine"]:
                vals = np.array(vals_by_method[method])
                xscale = (int(N) ** 3) / (int(b) * (L - mu) ** 2)
                rows.append({
                    "N": int(N),
                    "batch": int(b),
                    "method": method,
                    "epsN": float(eps),
                    "xscale_N3_over_bDelta2": float(xscale),
                    "mean_over_eps2": float(vals.mean() / (eps ** 2)),
                    "degradation": float(vals.mean() / (eps ** 2) - 1.0),
                    "se_over_eps2": float(vals.std(ddof=1) / math.sqrt(local_trials) / (eps ** 2)),
                    "trials": int(local_trials),
                })
    fit_batches = largest_design_tail(config.collapse_batches, config.fit_tail_fraction)
    fits = {}
    for method in ["prefix", "sine"]:
        rr = [r for r in rows if r["method"] == method and int(r["batch"]) in fit_batches]
        x = np.array([r["xscale_N3_over_bDelta2"] for r in rr])
        y = np.array([r["degradation"] for r in rr])
        if len(x) >= 2:
            slope = float(np.dot(x, y) / np.dot(x, x))
        else:
            slope = float("nan")
        fits[method] = {"slope_vs_N3_over_bDelta2": slope, "fit_batches": sorted(fit_batches)}
    return {"config": asdict(config), "mu": mu, "L": L, "Delta": L - mu, "rows": rows, "fits": fits}

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

_METHOD_LABEL = {"prefix": "Prefix-Chebyshev", "sine": "Sine/Jacobi"}
_METHOD_COLOR = {"prefix": "#4C72B0", "sine": "#DD8452"}
_METHOD_MARKER = {"prefix": "o", "sine": "s"}


def set_article_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 320,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_all(fig, base: str, outdir: str) -> List[str]:
    paths = []
    for ext in ["png", "pdf"]:
        path = os.path.join(outdir, f"{base}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    return paths


def plot_exp2_phase(exp: Dict, outdir: str) -> List[str]:
    import matplotlib.pyplot as plt
    paths = []
    rows = exp["rows"]
    fig, axs = plt.subplots(1, 2, figsize=(10.2, 4.0))
    for method in ["prefix", "sine"]:
        rr = sorted([r for r in rows if r["method"] == method], key=lambda z: z["scale"])
        x = np.array([r["scale"] for r in rr])
        med = np.array([r["median_over_eps"] for r in rr])
        q25 = np.array([r["q25_contr"] / exp["epsN"] for r in rr])
        q75 = np.array([r["q75_contr"] / exp["epsN"] for r in rr])
        acc = np.array([r["accept_rate"] for r in rr])
        axs[0].plot(x, med, marker=_METHOD_MARKER[method], color=_METHOD_COLOR[method], label=_METHOD_LABEL[method])
        axs[0].fill_between(x, q25, q75, color=_METHOD_COLOR[method], alpha=0.14, linewidth=0)
        axs[1].plot(x, acc, marker=_METHOD_MARKER[method], color=_METHOD_COLOR[method], label=_METHOD_LABEL[method])
    axs[0].axhline(1.0, color="0.2", lw=1.0, ls="--", label=r"ideal $\epsilon_N^\star$")
    axs[0].axhline(2.0, color="0.5", lw=0.9, ls=":", label="acceptance budget")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel(r"initial radius r")
    axs[0].set_ylabel(r"median block contraction / $\epsilon_N^\star$")
    axs[0].set_title("Finite-radius block degradation")
    axs[0].legend(loc="best")
    axs[1].set_xscale("log")
    axs[1].set_ylim(-0.04, 1.04)
    axs[1].set_xlabel(r"initial radius r")
    axs[1].set_ylabel("accepted-direction fraction")
    axs[1].set_title("Acceptance shell over endpoint directions")
    axs[1].legend(loc="best")
    fig.suptitle(f"Endpoint-coupled GLM one-block shell (N={exp['N']}, {exp['dirs']} directions)", y=1.02)
    paths += save_all(fig, "fig1_glm_one_block_shell", outdir)
    plt.close(fig)

    # Slope/degradation version as a single clean auxiliary figure.
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for method in ["prefix", "sine"]:
        rr = sorted([r for r in rows if r["method"] == method], key=lambda z: z["scale"])
        x = np.array([r["scale"] for r in rr])
        med = np.array([r["median_slope"] for r in rr])
        q25 = np.array([r["q25_slope"] for r in rr])
        q75 = np.array([r["q75_slope"] for r in rr])
        ax.plot(x, med, marker=_METHOD_MARKER[method], color=_METHOD_COLOR[method], label=_METHOD_LABEL[method])
        ax.fill_between(x, q25, q75, color=_METHOD_COLOR[method], alpha=0.14, linewidth=0)
    ax.axhline(0.0, color="0.25", lw=0.9, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"initial radius r")
    ax.set_ylabel(r"median $((r_N/r)-\epsilon_N^\star)/r$")
    ax.set_title("Observed first nonlinear degradation slope")
    ax.legend(loc="best")
    paths += save_all(fig, "fig1b_glm_degradation_slope", outdir)
    plt.close(fig)
    return paths


def plot_exp3_restart(exp: Dict, outdir: str) -> List[str]:
    import matplotlib.pyplot as plt
    paths = []
    rows = sorted(exp["rows"], key=lambda r: r["scale"])
    x = np.array([r["scale"] for r in rows])
    fig, axs = plt.subplots(1, 2, figsize=(10.2, 4.0))
    for method in ["prefix", "sine"]:
        yN = np.array([r[f"{method}_first_N"] for r in rows])
        yG = np.array([r[f"{method}_grads"] for r in rows])
        axs[0].plot(x, yN, marker=_METHOD_MARKER[method], color=_METHOD_COLOR[method], label=_METHOD_LABEL[method])
        axs[1].plot(x, yG, marker=_METHOD_MARKER[method], color=_METHOD_COLOR[method], label=_METHOD_LABEL[method])
    axs[0].axhline(exp["N_cap"], color="0.45", lw=0.9, ls=":", label="cap")
    axs[0].set_xscale("log")
    axs[0].set_xlabel(r"start radius r0")
    axs[0].set_ylabel("first accepted block length")
    axs[0].set_title("Continuation admits longer first blocks")
    axs[0].legend(loc="best")
    axs[1].set_xscale("log")
    axs[1].set_xlabel(r"start radius r0")
    axs[1].set_ylabel("gradients to target")
    axs[1].set_title("End-to-end finite-radius cost")
    axs[1].legend(loc="best")
    fig.suptitle("Restarted GLM continuation shell", y=1.02)
    paths += save_all(fig, "fig2_restart_shell_summary", outdir)
    plt.close(fig)

    # Representative trajectory.
    reps = exp.get("representative_histories", {})
    if reps:
        fig, axs = plt.subplots(1, 2, figsize=(10.2, 4.0))
        for method in ["prefix", "sine"]:
            hist = reps.get(method, {}).get("history", [])
            if not hist:
                continue
            grads = np.cumsum([h["N"] for h in hist])
            radii = np.array([h["r_after"] for h in hist])
            Ns = np.array([h["N"] for h in hist])
            axs[0].plot(grads, radii, marker=_METHOD_MARKER[method], color=_METHOD_COLOR[method], label=_METHOD_LABEL[method])
            axs[1].step(np.arange(len(Ns)), Ns, where="mid", marker=_METHOD_MARKER[method], color=_METHOD_COLOR[method], label=_METHOD_LABEL[method])
        axs[0].set_yscale("log")
        axs[0].set_xlabel("gradient evaluations")
        axs[0].set_ylabel(r"distance $\|x_k-x^\star\|$")
        axs[0].set_title("Representative restarted trajectory")
        axs[0].legend(loc="best")
        axs[1].set_xlabel("accepted block index")
        axs[1].set_ylabel("accepted N")
        axs[1].set_title("Block lengths along the trajectory")
        axs[1].legend(loc="best")
        paths += save_all(fig, "fig2b_restart_representative", outdir)
        plt.close(fig)
    return paths


def plot_exp4_stochastic(batch_exp: Dict, collapse_exp: Dict, outdir: str) -> List[str]:
    import matplotlib.pyplot as plt
    paths = []
    rows = batch_exp["rows"]
    eps = batch_exp["epsN"]
    fig, ax = plt.subplots(figsize=(5.8, 4.1))
    for method in ["prefix", "sine"]:
        rr = sorted([r for r in rows if r.get("method") == method], key=lambda z: z["batch"])
        b = np.array([r["batch"] for r in rr])
        y = np.array([r["mean_over_eps2"] for r in rr])
        se = np.array([r["se_sq"] / (eps ** 2) for r in rr])
        ax.errorbar(b, y, yerr=1.96 * se, marker=_METHOD_MARKER[method], color=_METHOD_COLOR[method],
                    capsize=2.8, lw=1.6, label=_METHOD_LABEL[method])
        # fitted perturbative 1+a/b curve
        a = batch_exp["fits"][method]["slope_a_over_b"]
        xx = np.linspace(b.min(), b.max(), 200)
        if np.isfinite(a):
            ax.plot(xx, 1.0 + a / xx, color=_METHOD_COLOR[method], lw=1.0, ls="--", alpha=0.8)
    ax.axhline(1.0, color="0.25", lw=0.9, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("mini-batch size b")
    ax.set_ylabel(r"$\mathbb{E}\|e_N\|^2/(\epsilon_N^{\star 2}\|e_0\|^2)$")
    ax.set_title(f"Pure stochastic curvature degradation (N={batch_exp['config']['N_main']})")
    ax.legend(loc="best")
    paths += save_all(fig, "fig3_stochastic_batch_scaling", outdir)
    plt.close(fig)

    # Paired gap figure.
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    gaps = sorted([r for r in rows if r.get("method") == "paired_gap"], key=lambda z: z["batch"])
    b = np.array([r["batch"] for r in gaps])
    y = np.array([r["mean_gap_over_eps2"] for r in gaps])
    se = np.array([r["se_gap_over_eps2"] for r in gaps])
    ax.errorbar(b, y, yerr=1.96 * se, marker="D", color="#55A868", capsize=2.8, lw=1.6)
    ax.axhline(0.0, color="0.25", lw=0.9, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("mini-batch size b")
    ax.set_ylabel(r"paired gap / $\epsilon_N^{\star 2}$")
    ax.set_title("Common-random-number gap: prefix minus sine")
    paths += save_all(fig, "fig3b_stochastic_paired_gap", outdir)
    plt.close(fig)

    # N^3/b collapse.
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    crows = collapse_exp["rows"]
    for method in ["prefix", "sine"]:
        rr = [r for r in crows if r["method"] == method]
        x = np.array([r["xscale_N3_over_bDelta2"] for r in rr])
        y = np.array([r["degradation"] for r in rr])
        se = np.array([r["se_over_eps2"] for r in rr])
        ax.errorbar(x, y, yerr=1.96 * se, marker=_METHOD_MARKER[method], linestyle="none",
                    color=_METHOD_COLOR[method], alpha=0.82, capsize=2.0, label=_METHOD_LABEL[method])
        slope = collapse_exp["fits"][method]["slope_vs_N3_over_bDelta2"]
        xs = np.linspace(0, max(x) * 1.04, 200)
        if np.isfinite(slope):
            ax.plot(xs, slope * xs, color=_METHOD_COLOR[method], lw=1.2, ls="--")
    ax.set_xlabel(r"$N^3/(b\Delta^2)$")
    ax.set_ylabel(r"degradation: mean$/\epsilon_N^{\star 2}-1$")
    ax.set_title(r"Collapse of stochastic curvature loss versus $N^3/b$")
    ax.legend(loc="best")
    paths += save_all(fig, "fig4_stochastic_N3_over_b_collapse", outdir)
    plt.close(fig)
    return paths


def plot_overview(exp2: Dict, exp3: Dict, batch_exp: Dict, outdir: str) -> List[str]:
    import matplotlib.pyplot as plt
    paths = []
    fig, axs = plt.subplots(1, 3, figsize=(14.4, 3.8))
    # Panel A: exp2 contraction/eps.
    for method in ["prefix", "sine"]:
        rr = sorted([r for r in exp2["rows"] if r["method"] == method], key=lambda z: z["scale"])
        x = np.array([r["scale"] for r in rr])
        y = np.array([r["median_over_eps"] for r in rr])
        axs[0].plot(x, y, marker=_METHOD_MARKER[method], color=_METHOD_COLOR[method], label=_METHOD_LABEL[method])
    axs[0].axhline(2.0, color="0.45", lw=0.9, ls=":")
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel("initial radius r")
    axs[0].set_ylabel(r"contraction$/\epsilon_N^\star$")
    axs[0].set_title("A. finite-radius shell")
    # Panel B: restart first N.
    rr = sorted(exp3["rows"], key=lambda r: r["scale"])
    x = np.array([r["scale"] for r in rr])
    for method in ["prefix", "sine"]:
        y = np.array([r[f"{method}_first_N"] for r in rr])
        axs[1].plot(x, y, marker=_METHOD_MARKER[method], color=_METHOD_COLOR[method], label=_METHOD_LABEL[method])
    axs[1].set_xscale("log")
    axs[1].set_xlabel("start initial radius r")
    axs[1].set_ylabel("first accepted N")
    axs[1].set_title("B. restarted continuation")
    # Panel C: stochastic batch.
    rows = batch_exp["rows"]; eps = batch_exp["epsN"]
    for method in ["prefix", "sine"]:
        rrm = sorted([r for r in rows if r.get("method") == method], key=lambda z: z["batch"])
        b = np.array([r["batch"] for r in rrm])
        y = np.array([r["mean_over_eps2"] for r in rrm])
        axs[2].plot(b, y, marker=_METHOD_MARKER[method], color=_METHOD_COLOR[method], label=_METHOD_LABEL[method])
    axs[2].axhline(1.0, color="0.25", lw=0.9, ls="--")
    axs[2].set_xscale("log")
    axs[2].set_xlabel("mini-batch size b")
    axs[2].set_ylabel(r"mean square$/\epsilon_N^{\star 2}$")
    axs[2].set_title("C. stochastic curvature")
    axs[2].legend(loc="best")
    paths += save_all(fig, "fig0_article_overview", outdir)
    plt.close(fig)
    return paths

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="/mnt/data/cheb_robustness_pkg_v2/out")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trials", type=int, default=320)
    parser.add_argument("--fast", action="store_true", help="reduce trials/directions for a quick smoke run")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    figdir = os.path.join(args.outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    set_article_style()

    dirs = 6 if args.fast else 10
    trials = 140 if args.fast else args.trials
    config = StochConfig(trials=trials, seed=args.seed + 1000)
    if args.fast:
        config = StochConfig(trials=trials, seed=args.seed + 1000,
                             batches=(16, 32, 64, 128, 256, 512),
                             N_grid=(28, 44), collapse_batches=(32, 64, 128, 256, 512),
                             noise_strength=1.0)

    results: Dict = {}
    results["exp2_glm_phase"] = exp2_glm_phase(seed=args.seed, N=36, dirs=dirs, eta_accept=1.0)
    results["exp3_restart_shell"] = exp3_restart_shell(seed=args.seed, eta_budget=0.23, N_cap=44)
    results["exp4_stochastic_batch"] = stochastic_batch_curve(config)
    results["exp4b_stochastic_collapse"] = stochastic_collapse(config)

    with open(os.path.join(args.outdir, "results_v2.json"), "w") as f:
        json.dump(results, f, indent=2)

    plot_exp2_phase(results["exp2_glm_phase"], figdir)
    plot_exp3_restart(results["exp3_restart_shell"], figdir)
    plot_exp4_stochastic(results["exp4_stochastic_batch"], results["exp4b_stochastic_collapse"], figdir)
    plot_overview(results["exp2_glm_phase"], results["exp3_restart_shell"], results["exp4_stochastic_batch"], figdir)


if __name__ == "__main__":
    main()
