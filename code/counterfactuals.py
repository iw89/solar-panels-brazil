"""Counterfactual subsidy designs used in the paper, for one demand specification.

    python code/counterfactuals.py main        (also: het, nofe)

Designs (subsidy c_mj for size j in sector m):
    uniform       one subsidy for all sectors and sizes
    size          one subsidy per size
    size_irr      per size x irradiation quartile                         (16 subsidies)
    size_demog    per size x quartiles of density, medium income, bathrooms (256 subsidies)
    proportional  c_mj = a * kappa_j * I_m              (subsidy per unit of expected generation)
    rule          c_mj = max(0, a * kappa_j * I_m - 1/|beta_m|)  (approximation to the optimum)
The scalar a of the two rules is set so that the budget is exactly spent.

Reads output/estimation_results_{spec}.csv and writes output/results_{spec}.csv (one row per
design x budget), output/subsidies_size_{spec}.csv and output/subsidies_irr_{spec}.csv.
"""

import sys
import time

import numpy as np
import pandas as pd

import paths
from clogit import log_denominator, split_theta
from regulator import SubsidyProblem, quantile_groups
from specs import CAPACITY, SIZES, build, price_coefficient

spec = sys.argv[1] if len(sys.argv) > 1 else "main"
BUDGETS = [1000, 2000, 5000, 10000, 20000]                         # 10^4 BRL
QUARTILES = 4

## DATA AND PARAMETERS

d = pd.read_parquet(paths.data("data_population.parquet"))
A = build(d, spec)
theta = pd.read_csv(paths.output(f"estimation_results_{spec}.csv"))["theta"].to_numpy()
beta, gamma = split_theta(theta, A["X"].shape[2], len(SIZES))
N, J = len(d), len(SIZES)

# baseline utility without the price term changes; the subsidy enters as -beta_m * c
V = A["X"] @ beta + np.column_stack([np.ones(N), A["W"]]) @ gamma.T
beta_p = price_coefficient(beta, A["x_names"], d)                  # (N,)
L = A["L"]
I = d["pv_output"].to_numpy(float)                                 # MWh per kW per year
alpha = (I * L)[:, None] * CAPACITY[None, :]                       # generation weights
print(f"spec {spec}: price coefficient mean {beta_p.mean():.3f}, range [{beta_p.min():.3f}, {beta_p.max():.3f}]")


def evaluate(cc):
    """Generation, spending and adoptions by size for subsidies cc (scalar or (N, J))."""
    U = V - beta_p[:, None] * cc
    s = np.exp(U - log_denominator(U)[:, None])
    return (alpha * s).sum(), (L[:, None] * cc * s).sum(), (L[:, None] * s).sum(axis=0)


def row(design, budget, cc, n_subsidies, mean_subsidy):
    G, S, adopt = evaluate(cc)
    return dict(spec=spec, design=design, budget=budget, subsidies=n_subsidies, mean_subsidy=mean_subsidy,
                spending=S, generation=G, **{f"adoptions_{s}": a for s, a in zip(SIZES, adopt)})


rows = [row("none", 0, 0.0, 0, 0.0)]

## GROUPED DESIGNS (SOLVED WITH SLSQP)

q_irr = quantile_groups(I, QUARTILES)
q_demog = ((quantile_groups(d["dens"].to_numpy(float), QUARTILES) * QUARTILES
            + quantile_groups(d["inc_med"].to_numpy(float), QUARTILES)) * QUARTILES
           + quantile_groups(d["bath"].to_numpy(float), QUARTILES))
jj = np.arange(J)[None, :]
designs = {
    "uniform": (np.zeros((N, J), dtype=int), 1),
    "size": (np.tile(np.arange(J), (N, 1)), J),
    "size_irr": (jj * QUARTILES + q_irr[:, None], J * QUARTILES),
    "size_demog": (jj * QUARTILES**3 + q_demog[:, None], J * QUARTILES**3),
}
saved = {}
for design, (index, n) in designs.items():
    t0 = time.time()
    problem = SubsidyProblem(V, L, beta_p, index, n, alpha)
    c = None
    for B in BUDGETS:
        c, _ = problem.solve(B, c0=c)                               # warm start from smaller budget
        cc = c[index]
        rows.append(row(design, B, cc, n, np.average(cc, weights=L[:, None] * np.ones((1, J)))))
        saved[(design, B)] = c
    print(f"  {design:<12} solved in {time.time() - t0:5.1f}s")

## SIMPLE RULES (ONE PARAMETER, SET BY BISECTION TO EXHAUST THE BUDGET)

expected_generation = CAPACITY[None, :] * I[:, None]               # MWh/year per installation
deductible = 1.0 / np.abs(beta_p)[:, None]                         # 10^4 BRL


def rule_subsidy(a, design):
    if design == "proportional":
        return a * expected_generation
    return np.maximum(0.0, a * expected_generation - deductible)


for design in ["proportional", "rule"]:
    for B in BUDGETS:
        lo, hi = 0.0, 1.0
        while evaluate(rule_subsidy(hi, design))[1] < B:
            hi *= 2
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if evaluate(rule_subsidy(mid, design))[1] < B else (lo, mid)
        cc = rule_subsidy(hi, design)
        rows.append(row(design, B, cc, 1, np.average(cc, weights=L[:, None] * np.ones((1, J)))) | {"rule_a": hi})

## SAVE

results = pd.DataFrame(rows)
results.to_csv(paths.output(f"results_{spec}.csv"), index=False)

size = pd.DataFrame([saved[("size", B)] for B in BUDGETS], columns=SIZES)
size.insert(0, "budget", BUDGETS)
size.to_csv(paths.output(f"subsidies_size_{spec}.csv"), index=False)

irr = pd.DataFrame([[B, s, *saved[("size_irr", B)].reshape(J, QUARTILES)[j]]
                    for B in BUDGETS for j, s in enumerate(SIZES)],
                   columns=["budget", "size", "Q1", "Q2", "Q3", "Q4"])
irr.to_csv(paths.output(f"subsidies_irr_{spec}.csv"), index=False)

G0 = results.loc[results.design == "none", "generation"].iloc[0]
summary = results[results.design != "none"].pivot(index="budget", columns="design", values="generation")
print("\nIncrease in generation relative to no subsidy (%):")
print((100 * (summary / G0 - 1)).round(2).to_string())
