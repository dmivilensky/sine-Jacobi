"""Evaluate the ground-exact polynomial family and its sensitivities.

For q=t^3, set D=t^2*exp(p), P=3*exp(-p), n^2=(D+Q)/k.
The variational ODE differentiates this identity and each Riccati filter;
prop:primal-jacobian gives the equivalent reflected integral formulas.
DOP853 is used for proposals (Hairer, Norsett and Wanner, Solving ODEs I,
1993; https://www.unige.ch/~hairer/software.html). No solver error estimate
is used as a rigorous enclosure."""
from __future__ import annotations
from .search_trace import traced
import numpy as np
from numpy.polynomial.chebyshev import chebval, chebvander
from scipy.integrate import solve_ivp
from scipy.special import roots_legendre
from .search_reference import K, EPS, log_cosh


class Primal:
    def __init__(self, scales, degree, order, second=False, tolerance=EPS**.8):
        self.scales = np.array(scales, float)
        self.degree = degree
        self.second = second
        self.tolerance = max(tolerance, 100*EPS)
        x, weights = roots_legendre(order)
        self.nodes, self.weights = (x+1)/2, weights/2
        self.basis = chebvander(x, degree)
        self.target = np.r_[1., 1., log_cosh(np.sqrt(self.scales)),
                            [K*K/6] if second else []]
        self.cache = None

    @staticmethod
    def shape(t, coefficients):
        a = np.exp(chebval(2*np.asarray(t)-1, coefficients))
        h = 2-np.asarray(t)**3
        z = np.sqrt((a+t*h)/K)
        return np.asarray(t)*z, 3/a, a

    @traced
    def evaluate(self, coefficients):
        coefficients = np.asarray(coefficients)
        if self.cache is not None and np.array_equal(coefficients, self.cache[0]):
            return self.cache[1]
        n, P, a = self.shape(self.nodes, coefficients)
        t = self.nodes
        h = 2-t**3
        weighted = self.weights*P
        values = [2*np.dot(weighted, n**j) for j in [2, 1, 3]]
        gradients = [(-2*weighted*t**3*h/K)@self.basis,
                     (-weighted*t*(a+2*t*h)/(K*np.sqrt((a+t*h)/K)))@self.basis,
                     (weighted*n**3*(a-2*t*h)/(a+t*h))@self.basis]
        observation_count = len(self.scales)+int(self.second)
        scales = np.r_[self.scales, [K] if self.second else []]
        quadratic = np.r_[np.ones(len(self.scales)), [0.] if self.second else []]
        dimension = self.degree+1

        def rhs(t, state):
            n, P, a = self.shape(t, coefficients)
            Q = t**3*(2-t**3)
            basis = chebvander(2*t-1, self.degree)[0]
            y = state.reshape(observation_count, dimension+1)
            r, sensitivity = y[:, 0], y[:, 1:]
            flow = P*(scales*n*n-2*r-quadratic*r*r)
            forcing = P*(-scales*Q/K+2*r+quadratic*r*r)
            derivative = -2*P*(1+quadratic*r)
            return np.c_[flow, derivative[:, None]*sensitivity
                         +forcing[:, None]*basis].ravel()

        flow = solve_ivp(rhs, (0., 1.), np.zeros(observation_count*(dimension+1)),
                         method="DOP853", dense_output=True,
                         rtol=self.tolerance, atol=self.tolerance*EPS**.25)
        if not flow.success:
            raise ArithmeticError(flow.message)
        nodes = flow.sol(t).reshape(observation_count, dimension+1, len(t))
        endpoint = flow.y[:, -1].reshape(observation_count, dimension+1)
        for j in range(observation_count):
            r, sensitivity = nodes[j, 0], nodes[j, 1:]
            r1, sensitivity1 = endpoint[j, 0], endpoint[j, 1:]
            if quadratic[j]:
                value = 2*np.dot(weighted, r)+np.log1p(r1)
                gradient = 2*(sensitivity@weighted-(weighted*r)@self.basis)
                gradient += sensitivity1/(1+r1)
            else:
                # Reflection gives the exact identity S=2 int_left A^2+A(0)^2.
                value = 2*np.dot(weighted, r*r)+r1*r1
                gradient = 4*(sensitivity@(weighted*r))-2*(weighted*r*r)@self.basis
                gradient += 2*r1*sensitivity1
            values.append(value)
            gradients.append(gradient)
        result = {"energy": values[2], "energy_gradient": gradients[2],
                  "observations": np.r_[values[:2], values[3:]],
                  "jacobian": np.array(gradients[:2]+gradients[3:]),
                  "residual": np.r_[values[:2], values[3:]]-self.target}
        self.cache = (coefficients.copy(), result)
        return result
