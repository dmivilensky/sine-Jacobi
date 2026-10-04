"""Evaluate rational contact references for numerical search.

The formulas are thm:contact and lem:regular-criterion. Evolving q-w directly
preserves its small leading-edge scale. Filters use DOP853, the explicit
8(5,3) Runge--Kutta method with continuous output described by Hairer,
Norsett and Wanner, Solving ODEs I (1993), and the author's implementation:
https://www.unige.ch/~hairer/software.html
Solver tolerances and Gaussian quadrature agreement are proposal criteria;
reference.hpp independently proves whole-interval enclosures."""
from __future__ import annotations
from .search_trace import traced
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.special import roots_legendre
from .search_reference import K, EPS, cubic_positive, target_second, vacuum_square


class Reference:
    def __init__(self, parameters, digits=12):
        self.exact = parameters
        self.beta, self.b, self.gamma = (float(parameters[name])
                                         for name in ["beta", "b", "gamma"])
        self.coefficients = np.array(parameters["support"], float)
        if not np.isfinite([self.beta,self.b,self.gamma,*self.coefficients]).all() or not (
                self.beta>=0 and self.b>0 and self.gamma>=0 and len(self.coefficients)%2==1
                and np.all(self.coefficients>0)):
            raise ValueError('finite admissible reference parameters required')
        self.tolerance = max(10.**(-digits), 100*EPS)
        self.flows = {}
        self.base_grids = {}
        def rhs(v, y):
            n, P = self.support(v)
            t = min(v, 2-v)
            q = t**3 if v <= 1 else 2-t**3
            # Evolve the residual itself, so its O(t^7) leading edge is not
            # formed by subtracting two independently integrated O(t^3) terms.
            r, w = y
            return P*np.array([q*q+self.beta*(q-r)**2-2*r,
                               K*n*n-2*w-self.beta*w*w])
        self.residual = solve_ivp(rhs, (0, 2), [0., 0.], method="DOP853",
                                  rtol=self.tolerance,
                                  atol=self.tolerance*EPS**.5, dense_output=True)
        if not self.residual.success:
            raise ArithmeticError(self.residual.message)

    def support(self, v):
        t = np.minimum(v, 2-np.asarray(v))
        h = 2-t**3
        Q = t**3*h
        c = self.coefficients
        p = c[0]+sum(c[j]*Q/(Q+c[j+1]) for j in range(1, len(c), 2))
        z = cubic_positive(3*t*h/K, 2*p*h/K)
        return t*z, 3/(2*t*h+2*p*h/z)

    def contact_ratio(self, t):
        t = np.asarray(t)
        q = t**3
        h = 2-q
        Q = q*h
        r = self.residual.sol(t)[0]
        wl = q-r
        wr = self.residual.sol(2-t)[1]
        ratio = np.divide(r, q, out=np.zeros_like(q), where=q != 0)
        return (q*(h-wr)+ratio*(2*wr-Q))/(h*(2+self.beta*(wl+wr)))

    def local(self, t):
        t = np.asarray(t)
        h = 2-t**3
        p = self.b+K*self.gamma*self.contact_ratio(t)
        z = cubic_positive(3*t*h/K, 2*p*h/K)
        return t*z, 3/(2*t*h+2*p*h/z)

    def filter(self, s, linear=False):
        key = (float(s), bool(linear))
        if key not in self.flows:
            def rhs(v, y):
                n, P = self.local(min(v, 2-v))
                w = y[0]
                return P*np.array([s*n*n-2*w-(0 if linear else w*w),
                                   w*w if linear else w])
            sol = solve_ivp(rhs, (0, 2), [0., 0.], method="DOP853",
                            rtol=self.tolerance, atol=self.tolerance*EPS**.5,
                            dense_output=True)
            if not sol.success:
                raise ArithmeticError(sol.message)
            w, acc = sol.sol(1)
            val = sol.y[1, -1]+sol.y[0, -1]**2/4 if linear else 2*acc+np.log1p(w)
            self.flows[key] = (sol, float(val))
        return self.flows[key]

    @staticmethod
    @lru_cache(maxsize=None)
    def nodes(order):
        if not isinstance(order,(int,np.integer)) or order<2:
            raise ValueError('quadrature order must be an integer at least two')
        t, weights = roots_legendre(order)
        nodes,weights=(t+1)/2,weights/2
        nodes.flags.writeable=False;weights.flags.writeable=False
        return nodes,weights

    @traced
    def base_value(self, order):
        # The support and its filter are fixed while b and gamma vary.
        # Cache their quadrature arrays, never the multiplier-dependent n.
        if order not in self.base_grids:
            t, weights = self.nodes(order)
            q = t**3
            h = 2-q
            r = self.residual.sol(t)[0]
            wl, wr = q-r, self.residual.sol(2-t)[1]
            ratio = self.contact_ratio(t)
            J = weights@((2*q*h*ratio-wl*wl-wr*wr)*self.support(t)[1])
            J -= vacuum_square(self.residual.sol(2)[1], self.beta)
            self.base_grids[order] = t, weights, h, ratio, J+target_second(self.beta)
        t, weights, h, ratio, contact_cost = self.base_grids[order]
        p = self.b+K*self.gamma*ratio
        n = t*cubic_positive(3*t*h/K, 2*p*h/K)
        return float(-self.b+2*self.b/K+9/K*(weights@(t*t*n))
                     -self.gamma*contact_cost)

    def polish_multipliers(self, order, tolerance):
        start = np.log([self.b, self.gamma])
        original = np.array([self.b, self.gamma])
        def objective(x):
            self.b, self.gamma = np.exp(x)
            return -self.base_value(order)
        result = minimize(objective, start, method="Nelder-Mead",
                          options={"xatol": tolerance**.5, "fatol": tolerance})
        self.b, self.gamma = np.exp(result.x)
        self.flows.clear()
        self.exact["b"] = repr(float(self.b))
        self.exact["gamma"] = repr(float(self.gamma))
        return {"success": bool(result.success), "message": result.message,
                "before": original.tolist(), "after": [self.b, self.gamma]}

    @traced
    def statistics(self, order):
        t, weights = self.nodes(order)
        n, P = self.local(t)
        return {"status": "ordinary evaluation", "base": self.base_value(order),
                "mass": float(2*(weights@(n*n*P))),
                "root_mass": float(2*(weights@(n*P))),
                "energy": float(2*(weights@(n**3*P)))}


def write_native_parameters(parameters, scales, path):
    lines = ["CONTACT_REFERENCE 1"]
    lines += [str(parameters[name]) for name in ["beta", "b", "gamma"]]
    lines += [str(len(parameters["support"]))]
    lines += [str(c) for c in parameters["support"]]
    lines += [str(len(scales))]+[str(s) for s in scales]
    Path(path).write_text("\n".join(lines)+"\n")
