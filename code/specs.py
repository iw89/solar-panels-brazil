"""Demand specifications shared by the estimation (estimate.py) and the counterfactuals.

The stock of PVs in the data files is cumulative *including* the current quarter's
adoptions (stock_t = stock_{t-1} + adoptions_t), so using it as a regressor puts the
dependent variable on the right-hand side. All specifications use
the lagged stock, stock_{t-1} = stock_t - adoptions_t, entered as log(1 + stock_{t-1}).

Specifications
    nofe     : log lagged stock, no time effects
    main     : log lagged stock + quarter fixed effects (common to all sizes)
    het      : main + one price coefficient per income quartile of the sector

With quarter fixed effects the electricity tariff (which varies only over time) is absorbed
by the fixed effects and dropped. Alternative-specific regressors are ordered so that the
price coefficient is always last.
"""

import numpy as np

SIZES = ["2kW", "4kW", "6kW", "8kW"]
CAPACITY = np.array([2.0, 4.0, 6.0, 8.0])
PRICES = ["p2q", "p4q", "p6q", "p8q"]
Y_NAMES = ["pv2q", "pv4q", "pv6q", "pv8q"]
ALL_ADOPTIONS = ["pv2q", "pv4q", "pv6q", "pv8q", "pv10q"]
DEMOGRAPHICS = ["stock", "dens", "inc_med", "inc_high", "bath", "age", "educ", "houses", "tariffq"]
T = 14

# Heterogeneous price sensitivity: one price coefficient per quartile of the share of households
# with per capita income above one minimum wage (inc_med + inc_high). Cut-offs are the quartiles
# over all 419,188 sector-quarters of the population, so that estimation and counterfactuals use
# the same groups. The highest-income quartile is the base ("price").
INCOME_CUTOFFS = [0.1679, 0.3111, 0.4980]


def income_quartile(d):
    """Income quartile of each observation, 1 (poorest) to 4."""
    return 1 + np.searchsorted(INCOME_CUTOFFS, (d["inc_med"] + d["inc_high"]).to_numpy(float), side="right")

SPECS = {
    "nofe": dict(fe=False, het=False),
    "main": dict(fe=True, het=False),
    "het": dict(fe=True, het=True),
}


def lagged_stock(d):
    """Cumulative installations in the sector before the quarter."""
    return d["stock"].to_numpy(float) - d[ALL_ADOPTIONS].to_numpy(float).sum(axis=1)


def build(d, spec):
    """Arrays for a specification.

    Returns dict with X (N, J, K) alternative-specific regressors (price last), W (N, H-1)
    sector regressors without constant, counts (N, J+1), L (N,), and the names x_names, w_names.
    """
    cfg = SPECS[spec]
    N, J = len(d), len(SIZES)
    w = {v: d[v].to_numpy(float) for v in DEMOGRAPHICS}
    w["stock"] = np.log1p(lagged_stock(d))
    w_names = [v for v in DEMOGRAPHICS if not (cfg["fe"] and v == "tariffq")]

    p = d[PRICES].to_numpy(float)
    xs, x_names = [], []
    if cfg["fe"]:
        tt = d["tt"].to_numpy()
        for t in range(2, T + 1):
            xs.append(np.repeat((tt == t)[:, None], J, axis=1).astype(float))
            x_names.append(f"quarter{t}")
    if cfg["het"]:
        q = income_quartile(d)
        for k in [1, 2, 3]:
            xs.append(p * (q == k)[:, None])
            x_names.append(f"price_x_income_q{k}")
    xs.append(p)
    x_names.append("price")

    Y = d[Y_NAMES].to_numpy(float)
    L = np.maximum(d["L"].to_numpy(float), 1.0)
    return dict(
        X=np.stack(xs, axis=2),
        W=np.column_stack([w[v] for v in w_names]),
        counts=np.column_stack([Y, L - Y.sum(axis=1)]),
        L=L,
        x_names=x_names,
        w_names=w_names,
    )


def price_coefficient(beta, x_names, d):
    """Price coefficient of each observation, shape (N,) (constant unless spec is het)."""
    b = dict(zip(x_names, beta))
    out = np.full(len(d), b["price"])
    if "price_x_income_q1" in b:
        q = income_quartile(d)
        for k in [1, 2, 3]:
            out = out + b[f"price_x_income_q{k}"] * (q == k)
    return out
