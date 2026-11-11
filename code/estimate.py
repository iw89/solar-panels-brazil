"""Estimates the conditional logit model for solar panel (PV) adoption.

    python code/estimate.py              (all specifications)
    python code/estimate.py main het     (selected specifications)

For each specification in specs.py writes to output/
    estimation_results_{spec}.csv, vcov_{spec}.csv, vcov_cluster_{spec}.csv
and a summary fit_statistics.csv (log-likelihood, hold-out fit). For the main specification
also writes the marginal effects, elasticities and relative effects of sector characteristics
(output/) and the data-fit figures
(draft/figs/datafit_in.png, datafit_out.png).
The estimation itself is done by pyfixest (see clogit.py).
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paths
from clogit import fit, log_shares, long_format, split_theta
from specs import SIZES, SPECS, T, build, price_coefficient

specs = sys.argv[1:] or list(SPECS)

d_in = pd.read_parquet(paths.data("data_sample.parquet"))          # estimation sample
d_out = pd.read_parquet(paths.data("data_offsample.parquet"))      # hold-out sectors


def stars(t):
    return np.select([t > 2.576, t > 1.960, t > 1.645], ["(***)", "(**)", "(*)"], "")


def predicted_shares(d, spec, theta):
    A = build(d, spec)
    beta, gamma = split_theta(theta, A["X"].shape[2], len(SIZES))
    V = A["X"] @ beta + np.column_stack([np.ones(len(d)), A["W"]]) @ gamma.T
    log_s = log_shares(V)
    return np.exp(log_s[:, :-1]), log_s, A, beta, gamma


## ESTIMATION

fit_rows = []
for spec in specs:
    A = build(d_in, spec)
    J, K = len(SIZES), A["X"].shape[2]
    long, columns = long_format(A["counts"], A["X"], A["W"], A["x_names"], A["w_names"], SIZES,
                                cluster=d_in["ii"])
    theta, vcov, vcov_cl = fit(long, columns, cluster=True)
    se, se_cl = np.sqrt(np.diag(vcov)), np.sqrt(np.diag(vcov_cl))
    t_stat, t_cl = np.abs(theta / se), np.abs(theta / se_cl)

    names = A["x_names"] + (["const"] + A["w_names"]) * J
    choice = [""] * K + [s for s in SIZES for _ in range(len(A["w_names"]) + 1)]
    result = pd.DataFrame(dict(name=names, choice=choice, theta=theta, SE=se, t_statistic=t_stat,
                               star=stars(t_stat), SE_cluster=se_cl, t_cluster=t_cl,
                               star_cluster=stars(t_cl)))
    result.to_csv(paths.output(f"estimation_results_{spec}.csv"), index=False)
    labels = [f"{n} ({c})" if c else n for n, c in zip(names, choice)]
    pd.DataFrame(vcov, index=labels, columns=labels).to_csv(paths.output(f"vcov_{spec}.csv"))
    pd.DataFrame(vcov_cl, index=labels, columns=labels).to_csv(paths.output(f"vcov_cluster_{spec}.csv"))

    # fit: log-likelihood, McFadden pseudo R2 (constants-only null), hold-out predictions
    _, log_s, _, _, _ = predicted_shares(d_in, spec, theta)
    ll = np.sum(A["counts"] * log_s)
    shares0 = A["counts"].sum(axis=0) / A["counts"].sum()
    ll0 = np.sum(A["counts"] * np.log(shares0))
    s_out, _, A_out, _, _ = predicted_shares(d_out, spec, theta)
    pred_out = (A_out["L"][:, None] * s_out).sum(axis=0)
    act_out = A_out["counts"][:, :J].sum(axis=0)
    fit_rows.append(dict(spec=spec, loglik=ll, pseudoR2=1 - ll / ll0, parameters=len(theta),
                         holdout_predicted=pred_out.sum(), holdout_observed=act_out.sum(),
                         **{f"holdout_pred_{s}": p for s, p in zip(SIZES, pred_out)},
                         **{f"holdout_obs_{s}": a for s, a in zip(SIZES, act_out)},
                         max_share_holdout=s_out.sum(axis=1).max()))
    price_rows = result[result.name.str.startswith("price")]
    print(f"\n{spec}: log-likelihood {ll:,.1f}; hold-out predicted {pred_out.sum():,.0f} "
          f"vs observed {act_out.sum():,.0f}")
    print(price_rows[["name", "theta", "SE", "SE_cluster"]].to_string(index=False))

fits = pd.DataFrame(fit_rows)
if os.path.exists(paths.output("fit_statistics.csv")):
    old = pd.read_csv(paths.output("fit_statistics.csv"))
    fits = pd.concat([old[~old.spec.isin(fits.spec)], fits])
fits.to_csv(paths.output("fit_statistics.csv"), index=False)

## MARGINAL EFFECTS, ELASTICITIES AND FIT FIGURES FOR THE MAIN SPECIFICATION

if "main" in specs:
    theta = pd.read_csv(paths.output("estimation_results_main.csv"))["theta"].to_numpy()
    s, _, A, beta, gamma = predicted_shares(d_in, "main", theta)
    J = len(SIZES)
    b = price_coefficient(beta, A["x_names"], d_in)[:, None, None]
    price = A["X"][:, :, -1]

    # ds_j/dp_k = beta * s_j (1{j=k} - s_k);  e_jk = ds_j/dp_k * p_k / s_j
    ds_dp = b * s[:, :, None] * (np.eye(J)[None] - s[:, None, :])
    elasticity = ds_dp / s[:, :, None] * price[:, None, :]
    pd.DataFrame(elasticity.mean(axis=0), columns=SIZES).to_csv(paths.output("mean_price_elasticity.csv"), index=False)

    # ds_j/dw_h = s_j (gamma_jh - sum_k s_k gamma_kh), constant dropped
    g = gamma[:, 1:]
    ds_dw = s[:, :, None] * (g[None] - (s @ g)[:, None, :])
    me = pd.DataFrame(ds_dw.mean(axis=0), columns=A["w_names"])
    me.insert(0, "Choice", SIZES)
    me.to_csv(paths.output("marginal_effects_demographics.csv"), index=False)
    pd.DataFrame({"size": SIZES, "mean_probability": s.mean(axis=0)}).to_csv(paths.output("mean_probability.csv"), index=False)

    # relative effects of discrete changes in sector characteristics (quoted in the text): change in the
    # mean adoption probability of each size, recomputing the probabilities exactly. The standard
    # deviation of density is the one across all sectors (Table 1).
    sd_dens = pd.read_parquet(paths.data("data_population.parquet"), columns=["tt", "dens"]).query("tt == 1")["dens"].std()
    rel = []
    for var, delta in [("inc_med", 0.1), ("educ", 0.1), ("dens", sd_dens)]:
        d_new = d_in.copy()
        d_new[var] = d_new[var] + delta
        s_new = predicted_shares(d_new, "main", theta)[0]
        rel.append(dict(variable=var, change=delta, **dict(zip(SIZES, s_new.mean(axis=0) / s.mean(axis=0) - 1))))
    pd.DataFrame(rel).to_csv(paths.output("relative_effects.csv"), index=False)

    # predicted vs observed adoptions per quarter
    for flag, d in [("in", d_in), ("out", d_out)]:
        s_d, _, A_d, _, _ = predicted_shares(d, "main", theta)
        tt = d["tt"].to_numpy() - 1
        pred = np.column_stack([np.bincount(tt, A_d["L"] * s_d[:, j], minlength=T) for j in range(J)])
        obs = np.column_stack([np.bincount(tt, A_d["counts"][:, j], minlength=T) for j in range(J)])
        quarters = d.drop_duplicates("tt").sort_values("tt")["tq"].tolist()
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for j, color in enumerate(["tab:blue", "tab:red", "tab:green", "black"]):
            ax.plot(range(T), obs[:, j], "-", color=color, alpha=0.7, label=f"{SIZES[j]} observed")
            ax.plot(range(T), pred[:, j], ":", color=color, lw=2, label=f"{SIZES[j]} predicted")
        ax.set_xticks(range(T))
        ax.set_xticklabels(quarters, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("New adoptions")
        ax.set_title("Estimation sample" if flag == "in" else "Hold-out sample", fontsize=10)
        ax.legend(fontsize=7, ncol=2, loc="upper left", frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(paths.FIGS, f"datafit_{flag}.png"), dpi=200)
        plt.close(fig)

print("\n", fits.round(3).to_string(index=False))
