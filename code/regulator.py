"""The regulator's subsidy problem.

The regulator chooses non-negative per-unit subsidies c to maximize solar generation
    G(c) = sum_i sum_j alpha_ij s_ij(c)
subject to the budget
    sum_i sum_j c_ij L_i s_ij(c) <= B.

The subsidy structure is given by an integer array `index` (N, J): the subsidy applied to size j
in market i is c[index[i, j]] (e.g. index[i, j] = j gives one subsidy per size). Solved with SLSQP
and exact gradients, started at c = 0 or at a warm start.
"""

import numpy as np
from scipy.optimize import minimize

from clogit import log_denominator


def quantile_groups(v, q):
    """Quantile group (0..q-1) of each element (breaks at the empirical quantiles, intervals closed
    on the left)."""
    breaks = np.quantile(v, np.arange(1, q) / q)
    return np.searchsorted(breaks, v, side="right")


def as_column(beta_p):
    """Scalar price coefficient as is; per-market coefficients as an (N, 1) column."""
    return beta_p if np.ndim(beta_p) == 0 else np.asarray(beta_p, float).reshape(-1, 1)


class SubsidyProblem:
    """Generation objective and budget (with gradients) for a given subsidy structure.

    beta_p: scalar, or one price coefficient per market (N,) when price sensitivity varies.
    """

    def __init__(self, V, L, beta_p, index, n_params, alpha):
        self.V, self.L, self.beta_p = V, L, as_column(beta_p)
        self.index, self.flat, self.n_params = index, index.ravel(), n_params
        self.alpha = alpha
        self._c = None

    def _evaluate(self, c):
        if self._c is not None and np.array_equal(c, self._c):
            return
        self._c = c.copy()
        cc = c[self.index]                                   # subsidy of each (i, j) cell
        U = self.V - self.beta_p * cc
        s = np.exp(U - log_denominator(U)[:, None])

        # objective and its derivative with respect to each cell subsidy c_ij
        a_s = self.alpha * s
        self.F = a_s.sum()
        dF = -self.beta_p * s * (self.alpha - a_s.sum(axis=1, keepdims=True))

        # government spending and its derivative
        cbar = (cc * s).sum(axis=1)
        self.S = self.L @ cbar
        dS = self.L[:, None] * s * (1.0 - self.beta_p * (cc - cbar[:, None]))

        # aggregate cell derivatives into derivatives with respect to the parameters
        self.dF = np.bincount(self.flat, dF.ravel(), minlength=self.n_params)
        self.dS = np.bincount(self.flat, dS.ravel(), minlength=self.n_params)

    def objective_value(self, c):
        self._evaluate(np.asarray(c, float))
        return self.F

    def solve(self, budget, c0=None):
        c0 = np.zeros(self.n_params) if c0 is None else np.asarray(c0, float)
        F0 = abs(self.objective_value(np.zeros(self.n_params))) or 1.0   # scaling

        def fun(c):
            self._evaluate(c)
            return -self.F / F0

        def jac(c):
            self._evaluate(c)
            return -self.dF / F0

        def budget_slack(c):
            self._evaluate(c)
            return np.array([(budget - self.S) / budget])

        def budget_jac(c):
            self._evaluate(c)
            return (-self.dS / budget)[None, :]

        res = minimize(fun, c0, jac=jac, method="SLSQP",
                       bounds=[(0.0, None)] * self.n_params,
                       constraints=[dict(type="ineq", fun=budget_slack, jac=budget_jac)],
                       options=dict(maxiter=5000, ftol=1e-10))
        c = np.where(res.x < 1e-10, 0.0, res.x)             # clean up numerical zeros
        if not res.success:
            print(f"WARNING (budget {budget}): {res.message}")
        return c, self.objective_value(c)
