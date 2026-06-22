#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import zipfile
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path('/mnt/data/cheb_sine_article_exact_layer')
OUT = ROOT / 'outputs'
TABLES = OUT / 'tables'
FIGS = OUT / 'figures'
for p in [TABLES, FIGS]:
    p.mkdir(parents=True, exist_ok=True)


def z_of_lambda(lam: np.ndarray | float, mu: float, L: float) -> np.ndarray | float:
    Delta = L - mu
    eta = (L + mu) / Delta
    return eta - 2.0 * np.asarray(lam) / Delta


def cheb_T(n: int, x: np.ndarray | float) -> np.ndarray | float:
    """Chebyshev T_n(x), stable enough for tested ranges."""
    xa = np.asarray(x, dtype=float)
    out = np.empty_like(xa, dtype=float)
    mask_hi = xa > 1.0
    mask_lo = xa < -1.0
    mask_mid = ~(mask_hi | mask_lo)
    out[mask_hi] = np.cosh(n * np.arccosh(xa[mask_hi]))
    out[mask_lo] = ((-1.0) ** n) * np.cosh(n * np.arccosh(-xa[mask_lo]))
    out[mask_mid] = np.cos(n * np.arccos(np.clip(xa[mask_mid], -1.0, 1.0)))
    if np.isscalar(x):
        return float(out)
    return out


def cheb_U(n: int, x: np.ndarray | float) -> np.ndarray | float:
    """Chebyshev U_n(x) by recurrence; U_{-1}=0 is allowed."""
    if n < 0:
        xa = np.asarray(x, dtype=float)
        out = np.zeros_like(xa, dtype=float)
        return float(out) if np.isscalar(x) else out
    xa = np.asarray(x, dtype=float)
    if n == 0:
        out = np.ones_like(xa, dtype=float)
    elif n == 1:
        out = 2.0 * xa
    else:
        u0 = np.ones_like(xa, dtype=float)
        u1 = 2.0 * xa
        for _ in range(1, n):
            u0, u1 = u1, 2.0 * xa * u1 - u0
        out = u1
    return float(out) if np.isscalar(x) else out


def cheb_residual(N: int, lam: np.ndarray | float, mu: float, L: float) -> np.ndarray | float:
    eta = (L + mu) / (L - mu)
    z = z_of_lambda(lam, mu, L)
    return cheb_T(N, z) / cheb_T(N, eta)


def cheb_T_eta_sequence(N: int, mu: float, L: float) -> np.ndarray:
    eta = (L + mu) / (L - mu)
    c = np.empty(N + 2, dtype=float)
    c[0] = 1.0
    if N + 1 >= 1:
        c[1] = eta
    for t in range(1, N + 1):
        c[t + 1] = 2.0 * eta * c[t] - c[t - 1]
    return c


@dataclass(frozen=True)
class PrefixChebyshev:
    mu: float
    L: float
    N: int

    @property
    def Delta(self) -> float:
        return self.L - self.mu

    @property
    def eta(self) -> float:
        return (self.L + self.mu) / (self.L - self.mu)

    def beta(self) -> np.ndarray:
        """beta_t = T_{t-1}(eta)/T_{t+1}(eta), t=1,...,N-1."""
        if self.N <= 1:
            return np.zeros(0)
        c = cheb_T_eta_sequence(self.N, self.mu, self.L)
        return np.array([c[t - 1] / c[t + 1] for t in range(1, self.N)], dtype=float)

    def scalar_residual_sequence(self, lam: float) -> np.ndarray:
        r = np.zeros(self.N + 1, dtype=float)
        r[0] = 1.0
        if self.N == 0:
            return r
        r[1] = 1.0 - 2.0 * lam / (self.L + self.mu)
        betas = self.beta()
        for t in range(1, self.N):
            b = betas[t - 1]
            r[t + 1] = r[t] + b * (r[t] - r[t - 1]) - (1.0 + b) * 2.0 * lam / (self.L + self.mu) * r[t]
        return r

    def run_time_varying_linear(self, Hs: List[np.ndarray], x0: np.ndarray) -> np.ndarray:
        assert len(Hs) == self.N
        if self.N == 0:
            return x0.copy()
        x_prev = x0.copy()
        x_cur = x_prev - (2.0 / (self.L + self.mu)) * (Hs[0] @ x_prev)
        betas = self.beta()
        for t in range(1, self.N):
            b = betas[t - 1]
            g = Hs[t] @ x_cur
            x_next = x_cur + b * (x_cur - x_prev) - (1.0 + b) * (2.0 / (self.L + self.mu)) * g
            x_prev, x_cur = x_cur, x_next
        return x_cur

    def run_gradient(self, grad: Callable[[np.ndarray], np.ndarray], x0: np.ndarray) -> np.ndarray:
        if self.N == 0:
            return x0.copy()
        x_prev = x0.copy()
        x_cur = x_prev - (2.0 / (self.L + self.mu)) * grad(x_prev)
        betas = self.beta()
        for t in range(1, self.N):
            b = betas[t - 1]
            x_next = x_cur + b * (x_cur - x_prev) - (1.0 + b) * (2.0 / (self.L + self.mu)) * grad(x_cur)
            x_prev, x_cur = x_cur, x_next
        return x_cur


@dataclass(frozen=True)
class SineJacobi:
    mu: float
    L: float
    N: int

    @property
    def Delta(self) -> float:
        return self.L - self.mu

    @property
    def eta(self) -> float:
        return (self.L + self.mu) / (self.L - self.mu)

    def a2(self) -> np.ndarray:
        """a_{s,N}^2, s=1,...,N-1, exactly as in the article."""
        if self.N <= 1:
            return np.zeros(0)
        N = self.N
        vals = []
        for s in range(1, N):
            val = 0.25 * (math.sin(math.pi * s / N) ** 2) / (
                math.sin(math.pi * (s - 0.5) / N) * math.sin(math.pi * (s + 0.5) / N)
            )
            vals.append(val)
        return np.array(vals, dtype=float)

    def p_eta(self) -> np.ndarray:
        """p_s=P_s(eta), s=0,...,N."""
        p = np.zeros(self.N + 1, dtype=float)
        p[0] = 1.0
        if self.N == 0:
            return p
        p[1] = self.eta
        a2 = self.a2()
        for s in range(1, self.N):
            p[s + 1] = self.eta * p[s] - a2[s - 1] * p[s - 1]
        return p

    def coefficients(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """alpha_s, beta_s, gamma_s for s=1,...,N-1."""
        if self.N <= 1:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        p = self.p_eta()
        a2 = self.a2()
        alpha = np.zeros(self.N - 1, dtype=float)
        beta = np.zeros(self.N - 1, dtype=float)
        gamma = np.zeros(self.N - 1, dtype=float)
        for s in range(1, self.N):
            alpha[s - 1] = self.eta * p[s] / p[s + 1]
            beta[s - 1] = a2[s - 1] * p[s - 1] / p[s + 1]
            gamma[s - 1] = (2.0 / self.Delta) * p[s] / p[s + 1]
        return alpha, beta, gamma

    def P_values(self, z: float) -> np.ndarray:
        """Monic leading polynomials P_s(z), s=0,...,N."""
        P = np.zeros(self.N + 1, dtype=float)
        P[0] = 1.0
        if self.N == 0:
            return P
        P[1] = z
        a2 = self.a2()
        for s in range(1, self.N):
            P[s + 1] = z * P[s] - a2[s - 1] * P[s - 1]
        return P

    def Q_tail_value(self, s: int, x: float) -> float:
        """Q_{N-s-1}^{(s+1)}(x), right-tail characteristic polynomial."""
        m = self.N - s - 1
        if m <= 0:
            return 1.0
        if m == 1:
            return float(x)
        a2 = self.a2()
        q0 = 1.0
        q1 = float(x)
        # q_{r+1}=x q_r - a_{s+1+r}^2 q_{r-1}, r=1,...,m-1
        for r in range(1, m):
            edge_index = s + 1 + r  # a_{edge_index}; 1-based
            q0, q1 = q1, x * q1 - a2[edge_index - 1] * q0
        return float(q1)

    def scalar_residual_sequence(self, lam: float) -> np.ndarray:
        r = np.zeros(self.N + 1, dtype=float)
        r[0] = 1.0
        if self.N == 0:
            return r
        r[1] = 1.0 - 2.0 * lam / (self.L + self.mu)
        alpha, beta, gamma = self.coefficients()
        for s in range(1, self.N):
            r[s + 1] = (alpha[s - 1] - gamma[s - 1] * lam) * r[s] - beta[s - 1] * r[s - 1]
        return r

    def run_time_varying_linear(self, Hs: List[np.ndarray], x0: np.ndarray) -> np.ndarray:
        assert len(Hs) == self.N
        if self.N == 0:
            return x0.copy()
        x_prev = x0.copy()
        x_cur = x_prev - (2.0 / (self.L + self.mu)) * (Hs[0] @ x_prev)
        alpha, beta, gamma = self.coefficients()
        for s in range(1, self.N):
            g = Hs[s] @ x_cur
            x_next = alpha[s - 1] * x_cur - beta[s - 1] * x_prev - gamma[s - 1] * g
            x_prev, x_cur = x_cur, x_next
        return x_cur

    def run_gradient(self, grad: Callable[[np.ndarray], np.ndarray], x0: np.ndarray) -> np.ndarray:
        if self.N == 0:
            return x0.copy()
        x_prev = x0.copy()
        x_cur = x_prev - (2.0 / (self.L + self.mu)) * grad(x_prev)
        alpha, beta, gamma = self.coefficients()
        for s in range(1, self.N):
            x_next = alpha[s - 1] * x_cur - beta[s - 1] * x_prev - gamma[s - 1] * grad(x_cur)
            x_prev, x_cur = x_cur, x_next
        return x_cur


def prefix_kernel_formula(N: int, mu: float, L: float, lam: float, nu: float, s: int) -> float:
    Delta = L - mu
    eta = (L + mu) / Delta
    x = float(z_of_lambda(lam, mu, L))
    y = float(z_of_lambda(nu, mu, L))
    denom = Delta * cheb_T(N, eta)
    if s == 0:
        return -2.0 / denom * cheb_U(N - 1, x)
    return -4.0 / denom * cheb_U(N - s - 1, x) * cheb_T(s, y)


def sine_kernel_formula(N: int, mu: float, L: float, lam: float, nu: float, s: int) -> float:
    sj = SineJacobi(mu, L, N)
    Delta = L - mu
    x = float(z_of_lambda(lam, mu, L))
    y = float(z_of_lambda(nu, mu, L))
    P_y = sj.P_values(y)[s]
    Q_x = sj.Q_tail_value(s, x)
    P_eta_N = sj.p_eta()[N]
    return -2.0 / (Delta * P_eta_N) * Q_x * P_y


def finite_difference_kernel(method, lam: float, nu: float, s: int, delta: float = 1e-7) -> float:
    H = np.diag([lam, nu]).astype(float)
    B = np.array([[0.0, 1.0], [0.0, 0.0]])
    Hs0 = [H.copy() for _ in range(method.N)]
    Hs1 = [H.copy() for _ in range(method.N)]
    Hs1[s] = Hs1[s] + delta * B
    x0 = np.array([0.0, 1.0])
    y0 = method.run_time_varying_linear(Hs0, x0)
    y1 = method.run_time_varying_linear(Hs1, x0)
    return float((y1[0] - y0[0]) / delta)


def jacobi_matrix_sine(N: int) -> np.ndarray:
    if N == 0:
        return np.zeros((0, 0))
    sj = SineJacobi(1.0, 2.0, N)  # mu,L irrelevant for a2
    a = np.sqrt(sj.a2())
    J = np.zeros((N, N), dtype=float)
    for s in range(1, N):
        J[s - 1, s] = a[s - 1]
        J[s, s - 1] = a[s - 1]
    return J


def normalized_poly_apply(eigs: np.ndarray, coeffs: np.ndarray, N: int, mu: float, L: float) -> np.ndarray:
    return cheb_residual(N, eigs, mu, L) * coeffs


def run_verifications() -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    summary = {}
    rng = np.random.default_rng(12345)
    Ns = [1, 2, 3, 5, 8, 16, 32]
    mu = 1.0
    L = 100.0
    lam_grid = np.r_[np.linspace(mu, L, 17), rng.uniform(mu, L, 23)]

    def add(name: str, N: int, err: float, tol: float):
        rows.append({'test': name, 'N': N, 'error': err, 'tolerance': tol, 'pass': bool(err <= tol)})

    for N in Ns:
        pref = PrefixChebyshev(mu, L, N)
        sine = SineJacobi(mu, L, N)
        # Prefix exactness at every prefix.
        max_err = 0.0
        for lam in lam_grid:
            seq = pref.scalar_residual_sequence(float(lam))
            for t in range(N + 1):
                max_err = max(max_err, abs(seq[t] - cheb_residual(t, float(lam), mu, L)))
        add('prefix_every_step_chebyshev_residual', N, max_err, 5e-11)

        # Sine residual equals P_s(z)/P_s(eta) at every step, final equals Chebyshev.
        max_err = 0.0
        final_err = 0.0
        for lam in lam_grid:
            z = float(z_of_lambda(float(lam), mu, L))
            seq = sine.scalar_residual_sequence(float(lam))
            P = sine.P_values(z)
            p = sine.p_eta()
            max_err = max(max_err, float(np.max(np.abs(seq - P / p))))
            final_err = max(final_err, abs(seq[N] - cheb_residual(N, float(lam), mu, L)))
        add('sine_jacobi_step_residual_equals_Ps_over_Ps_eta', N, max_err, 5e-10)
        add('sine_jacobi_terminal_chebyshev_residual', N, final_err, 5e-10)

        # Sine Jacobi eigenvalues exactly Chebyshev midpoint nodes.
        if N >= 1:
            J = jacobi_matrix_sine(N)
            eig = np.sort(np.linalg.eigvalsh(J))
            nodes = np.sort(np.cos((2 * np.arange(1, N + 1) - 1) * math.pi / (2 * N)))
            err = float(np.max(np.abs(eig - nodes))) if N > 0 else 0.0
            add('sine_jacobi_matrix_spectrum_midpoint_chebyshev_nodes', N, err, 5e-12)

        # Fixed quadratic vector test.
        d = 7
        eigs = np.linspace(mu, L, d)
        H = np.diag(eigs)
        x0 = rng.normal(size=d)
        Hs = [H.copy() for _ in range(N)]
        if N == 0:
            y_pref = x0.copy()
            y_sine = x0.copy()
        else:
            y_pref = pref.run_time_varying_linear(Hs, x0)
            y_sine = sine.run_time_varying_linear(Hs, x0)
        y_star = normalized_poly_apply(eigs, x0, N, mu, L)
        add('prefix_fixed_quadratic_vector_equals_cheb_polynomial', N, float(np.linalg.norm(y_pref - y_star, ord=np.inf)), 5e-10)
        add('sine_fixed_quadratic_vector_equals_cheb_polynomial', N, float(np.linalg.norm(y_sine - y_star, ord=np.inf)), 5e-10)
        add('prefix_and_sine_terminal_match_on_fixed_quadratic', N, float(np.linalg.norm(y_pref - y_sine, ord=np.inf)), 5e-10)

    # First-variation finite-difference kernel tests for representative Ns.
    for N in [3, 5, 8, 12]:
        pref = PrefixChebyshev(mu, L, N)
        sine = SineJacobi(mu, L, N)
        errs_pref = []
        errs_sine = []
        for _ in range(24):
            lam = float(rng.uniform(mu, L))
            nu = float(rng.uniform(mu, L))
            s = int(rng.integers(0, N))
            fd = finite_difference_kernel(pref, lam, nu, s, delta=1e-7)
            ff = prefix_kernel_formula(N, mu, L, lam, nu, s)
            errs_pref.append(abs(fd - ff))
            fd2 = finite_difference_kernel(sine, lam, nu, s, delta=1e-7)
            ff2 = sine_kernel_formula(N, mu, L, lam, nu, s)
            errs_sine.append(abs(fd2 - ff2))
        add('prefix_kernel_matches_time_varying_finite_difference', N, float(max(errs_pref)), 5e-8)
        add('sine_jacobi_kernel_matches_time_varying_finite_difference', N, float(max(errs_sine)), 5e-8)

    df = pd.DataFrame(rows)
    summary['all_tests_pass'] = bool(df['pass'].all())
    summary['num_tests'] = int(len(df))
    summary['num_failed'] = int((~df['pass']).sum())
    summary['max_error_over_tolerance'] = float((df['error'] / df['tolerance']).max())
    return df, summary

def main() -> None:
    df, summary = run_verifications()
    df.to_csv(TABLES / 'article_exact_verification_tests.csv', index=False)
    (TABLES / 'article_exact_verification_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    if not summary['all_tests_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
