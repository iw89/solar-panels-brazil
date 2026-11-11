"""Conditional logit on aggregate purchase counts, estimated with pyfixest.

Model, for market-period i and alternative j = 1..J (outside good j = 0, V_i0 = 0):
    V_ij = x_ij' beta + gamma_j' w_i,      s_ij = exp(V_ij) / (1 + sum_k exp(V_ik))
where w_i contains a constant (so gamma_j includes an alternative-specific constant).

Estimation uses the multinomial-Poisson equivalence (Baker 1994): the Poisson
regression
    E[y_ij] = exp(alpha_i + V_ij),   one fixed effect alpha_i per market-period,
on the long data (one row per market-period x alternative, outside good included
with zero regressors) gives exactly the conditional logit MLE of (beta, gamma), and
its (unadjusted) iid covariance is the inverse information of the logit likelihood.
pyfixest absorbs the ~200k fixed effects efficiently and also gives clustered SEs.
"""

import numpy as np
import pandas as pd
import pyfixest as pf


def long_format(counts, X, W, x_names, w_names, alt_names, cluster=None):
    """Reshapes to one row per (market-period, alternative).

    counts : (N, J+1) purchases, inside goods first, outside good last
    X      : (N, J, K) alternative-specific regressors (coefficient beta, common to all j)
    W      : (N, H) market regressors without constant (coefficients gamma_j, one set per j)
    Returns the long DataFrame and the regressor columns in the order
    [x_names..., const_alt1, w_names_alt1..., const_alt2, ...], i.e. theta = [beta; vec(gamma')].
    """
    N, J, K = X.shape
    alt = np.tile(np.arange(J + 1), N)
    long = pd.DataFrame({"mkt": np.repeat(np.arange(N), J + 1), "count": counts.ravel()})
    if cluster is not None:
        long["cluster"] = np.repeat(np.asarray(cluster), J + 1)

    for k, name in enumerate(x_names):                       # outside good: x = 0
        long[name] = np.column_stack([X[:, :, k], np.zeros(N)]).ravel()

    columns = list(x_names)
    W1 = np.column_stack([np.ones(N), W])
    for j, a in enumerate(alt_names):                        # interactions w_i x 1{alt = j}
        dummy = (alt == j).astype(float)
        for h, v in enumerate(["const"] + list(w_names)):
            col = f"{v}_{a}"
            long[col] = dummy * np.repeat(W1[:, h], J + 1)
            columns.append(col)
    return long, columns


def fit(long, columns, cluster=False):
    """Estimates the model. Returns theta, the MLE (iid) covariance and, if
    cluster=True, the covariance clustered by the `cluster` column."""
    fml = "count ~ " + " + ".join(columns) + " | mkt"
    model = pf.fepois(fml, data=long, vcov="iid", ssc=pf.ssc(k_adj=False))
    theta = model.coef()[columns].to_numpy()
    vcov = pd.DataFrame(model._vcov, index=model._coefnames, columns=model._coefnames)
    vcov_iid = vcov.loc[columns, columns].to_numpy()
    if not cluster:
        return theta, vcov_iid, None
    model.vcov({"CRV1": "cluster"})
    vcov = pd.DataFrame(model._vcov, index=model._coefnames, columns=model._coefnames)
    return theta, vcov_iid, vcov.loc[columns, columns].to_numpy()


def split_theta(theta, K, J):
    """theta -> (beta (K,), gamma (J, H)) with gamma rows = alternatives, cols = [const, w...]."""
    theta = np.asarray(theta, float)
    return theta[:K], theta[K:].reshape(J, -1)


def log_denominator(V):
    """log(1 + sum_j exp(V_ij)) for each row, computed stably."""
    u = np.maximum(V.max(axis=1), 0.0)
    return u + np.log(np.exp(-u) + np.exp(V - u[:, None]).sum(axis=1))


def log_shares(V):
    """log choice probabilities (N, J+1) from utilities V (N, J); outside good last."""
    return np.column_stack([V, np.zeros(len(V))]) - log_denominator(V)[:, None]
