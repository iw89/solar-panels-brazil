"""Generates the LaTeX results tables of the paper.

    python code/tables.py

Reads the data (summary statistics), the estimates and the counterfactual results in output/
and writes draft/tables/*.tex, which draft/main.tex includes with input commands. Also writes
output/irradiation_quartiles.csv (adoption and income by irradiation quartile, quoted in the text).
"""

import os

import numpy as np
import pandas as pd

import paths
from regulator import quantile_groups

SIZES = ["2 kW", "4 kW", "6 kW", "8 kW"]
SHOW_BUDGETS = [1000, 5000, 20000]          # budgets shown in the irradiation table (10^4 BRL)

LABELS = {
    "stock": "Log(1 + stock of PVs)",
    "dens": "Density ($10^4$ hab/km$^2$)",
    "inc_med": "Medium income (share)",
    "inc_high": "High income (share)",
    "bath": "Bathrooms",
    "age": "Age (decades)",
    "educ": "Literacy (share)",
    "houses": "Houses (share)",
    "tariffq": "Electricity tariff (BRL/kWh)",
    "const": "Constant",
}
DESIGNS = [("uniform", "Uniform"), ("size", "Size"), ("size_irr", "Size \\& irr."),
           ("size_demog", "Size \\& dem."), ("proportional", "Prop."), ("rule", "Rule")]


def demand(name):
    return pd.read_csv(paths.output(name))


def reg(name):
    return pd.read_csv(paths.output(name))


def write(name, text):
    with open(os.path.join(paths.TABLES, name), "w", encoding="utf-8") as f:
        f.write(text)
    print("wrote", name)


def stars(t):
    return "***" if t > 2.576 else "**" if t > 1.960 else "*" if t > 1.645 else ""


def f(x, d=3):
    return f"{x:,.{d}f}"


def bold_row(cells):
    return " & ".join(f"\\textbf{{{c}}}" for c in cells)


def table(caption, label, colspec, body, notes):
    return (
        "\\begin{table}[htbp]\n\\centering\n\\small\n\\begin{threeparttable}\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{colspec}}}\n\\hline \\hline \\noalign{{\\vskip 1mm}}\n"
        f"{body}"
        "\\noalign{\\vskip 1mm} \\hline\n\\end{tabular}\n"
        f"\\begin{{tablenotes}}\n\\footnotesize\n\\item {notes}\n\\end{{tablenotes}}\n"
        "\\end{threeparttable}\n\\end{table}\n"
    )


## SUMMARY STATISTICS AND IRRADIATION QUARTILES (FROM THE DATA)

pop = pd.read_parquet(paths.data("data_population.parquet"))
first = pop[pop.tt == 1]                                  # one row per sector, start of the period
by_quarter = pop.drop_duplicates("tt")                    # prices and tariffs vary by quarter only
lagged_stock = pop["stock"] - pop[["pv2q", "pv4q", "pv6q", "pv8q", "pv10q"]].sum(axis=1)


def sum_row(label, x, d=3):
    x = pd.Series(x, dtype=float)
    stats = [x.mean(), x.std(), x.min(), x.median(), x.max()]
    cells = [f"{v:,.0f}" if float(v).is_integer() else f"{v:,.{d}f}" for v in stats]
    return label + " & " + " & ".join(cells) + f" & {len(x):,} \\\\\n"


body = ("\\textbf{Variable} & \\textbf{Mean} & \\textbf{SD} & \\textbf{Min} & \\textbf{Median} & \\textbf{Max} "
        "& \\textbf{N} \\\\ \\hline \\noalign{\\vskip 1mm}\n"
        "\\multicolumn{7}{l}{\\textit{Adoptions per sector-quarter}} \\\\\n")
for s, y in zip(SIZES, ["pv2q", "pv4q", "pv6q", "pv8q"]):
    body += sum_row(s, pop[y])
body += "\\multicolumn{7}{l}{\\textit{Investment price ($10^4$ BRL)}} \\\\\n"
for s, p in zip(SIZES, ["p2q", "p4q", "p6q", "p8q"]):
    body += sum_row(s, by_quarter[p])
body += "\\multicolumn{7}{l}{\\textit{Sector characteristics}} \\\\\n"
body += sum_row("Households (2016Q3)", first["L"], 0)
for var in ["dens", "inc_med", "inc_high", "bath", "age", "educ", "houses"]:
    body += sum_row(LABELS[var], first[var])
body += sum_row("Stock of PVs before the quarter", lagged_stock)
body += sum_row("Solar generation (MWh/kW/year)", first["pv_output"])
body += sum_row(LABELS["tariffq"], by_quarter["tariffq"])
write("tab_sum.tex", table(
    "Summary Statistics", "tab:sum", "lcccccc", body,
    f"All {len(first):,} sectors over {pop.tt.nunique()} quarters (2016Q3--2019Q4). Households are those that "
    "had not adopted at the start of the period. Prices and tariffs are common to the whole area. "
    "December 2019 BRL."))

# adoption and income by irradiation quartile of the sector (quoted in Section 2)
q_irr = quantile_groups(first["pv_output"].to_numpy(float), 4)
adopt = pop.groupby("ii")[["pv2q", "pv4q", "pv6q", "pv8q"]].sum().sum(axis=1).reindex(first["ii"]).to_numpy()
irr = pd.DataFrame({"quartile": q_irr + 1, "adoptions": adopt, "households": first["L"].to_numpy(),
                    "inc_above_mw": (first["inc_med"] + first["inc_high"]).to_numpy(),
                    "pv_output": first["pv_output"].to_numpy()}).groupby("quartile")
irr = pd.DataFrame({"sectors": irr.size(), "adoptions": irr.adoptions.sum(), "households": irr.households.sum(),
                    "inc_above_mw": irr.inc_above_mw.mean(), "pv_output": irr.pv_output.mean()})
irr["adoptions_per_1000_households"] = 1000 * irr.adoptions / irr.households
irr.to_csv(paths.output("irradiation_quartiles.csv"))
print(irr.round(3).to_string())

fits = demand("fit_statistics.csv").set_index("spec")

## ESTIMATION RESULTS (MAIN SPECIFICATION)

est = demand("estimation_results_main.csv")
price = est[est.name == "price"].iloc[0]
body = (f"Investment price ($10^4$ BRL) & {f(price.theta)}{stars(price.t_cluster)} & & & \\\\\n"
        f" & ({f(price.SE_cluster)}) & & & \\\\\n & [{f(price.SE)}] & & & \\\\\n"
        " & " + bold_row(SIZES) + " \\\\ \\cline{2-5} \\noalign{\\vskip 1mm}\n")
g = est[est.choice.notna()]
for var in ["const", "stock", "dens", "inc_med", "inc_high", "bath", "age", "educ", "houses"]:
    rows = list(g[g["name"] == var].itertuples())
    body += LABELS[var] + " & " + " & ".join(f"{f(r.theta)}{stars(r.t_cluster)}" for r in rows) + " \\\\\n"
    body += " & " + " & ".join(f"({f(r.SE_cluster)})" for r in rows) + " \\\\\n"
    body += " & " + " & ".join(f"[{f(r.SE)}]" for r in rows) + " \\\\\n"
body += (f"\\hline \\noalign{{\\vskip 1mm}}\nQuarter fixed effects & \\multicolumn{{4}}{{l}}{{Yes}} \\\\\n"
         f"Log-likelihood & \\multicolumn{{4}}{{l}}{{{fits.loc['main', 'loglik']:,.1f}}} \\\\\n"
         "Sector-quarters & \\multicolumn{4}{l}{210,910} \\\\\n")
write("tab_results.tex", table(
    "Estimation Results", "tab:results", "lllll", body,
    "Maximum likelihood estimates on the estimation sample (15,065 sectors, 14 quarters), with quarter fixed "
    "effects common to all sizes (which absorb the electricity tariff). Standard errors clustered by sector in "
    "parentheses, conventional ones in brackets; stars refer to the clustered ones (*** $p<0.01$, ** $p<0.05$, "
    "* $p<0.1$). No adoption is the reference alternative."))

## SPECIFICATION COMPARISON

cols = [("nofe", "(1)"), ("main", "(2)"), ("het", "(3)")]
coef = {s: demand(f"estimation_results_{s}.csv").set_index("name") for s, _ in cols}
lines = [("price", "Price"), ("price_x_income_q1", "Price $\\times$ income Q1"),
         ("price_x_income_q2", "Price $\\times$ income Q2"), ("price_x_income_q3", "Price $\\times$ income Q3")]
body = " & " + " & ".join(c for _, c in cols) + " \\\\ \\hline \\noalign{\\vskip 1mm}\n"
for key, label in lines:
    body += label + " & " + " & ".join(
        f"{f(coef[s].loc[key, 'theta'])}{stars(coef[s].loc[key, 't_cluster'])}" if key in coef[s].index else ""
        for s, _ in cols) + " \\\\\n"
    body += " & " + " & ".join(f"({f(coef[s].loc[key, 'SE_cluster'])})" if key in coef[s].index else ""
                               for s, _ in cols) + " \\\\\n"
body += "\\noalign{\\vskip 1mm}\n"
body += "Quarter fixed effects & No & Yes & Yes \\\\\n"
body += "Log-likelihood & " + " & ".join(f"{fits.loc[s, 'loglik']:,.1f}" for s, _ in cols) + " \\\\\n"
body += ("Hold-out adoptions, predicted & " + " & ".join(f"{fits.loc[s, 'holdout_predicted']:,.0f}" for s, _ in cols)
         + " \\\\\n")
body += f"Hold-out adoptions, observed & \\multicolumn{{3}}{{c}}{{{fits.loc['main', 'holdout_observed']:,.0f}}} \\\\\n"
write("tab_specs.tex", table(
    "Price Coefficients Across Specifications", "tab:specs", "lccc", body,
    "All specifications include the sector characteristics of Table \\ref{tab:results} with size-specific "
    "coefficients; (1) also includes the electricity tariff. In (3) the price coefficient differs across quartiles "
    "of the share of households with per capita income above one minimum wage; ``Price'' is the coefficient of "
    "the richest quartile and the interactions are differences relative to it. Clustered standard errors in "
    "parentheses. The hold-out sample consists of the 14,877 sectors not used in estimation."))

## ELASTICITIES AND DEMOGRAPHIC MARGINAL EFFECTS

el = demand("mean_price_elasticity.csv").to_numpy()
body = " & " + bold_row(SIZES) + " \\\\ \\hline \\noalign{\\vskip 1mm}\n"
for s, row in zip(SIZES, el):
    body += f"\\textbf{{{s}}} & " + " & ".join(f"{x:.4f}" for x in row) + " \\\\\n"
write("tab_elasticity.tex", table(
    "Mean Price Elasticities", "tab:mean_price_elasticity", "lcccc", body,
    "Elasticity of the probability of adopting the row size with respect to the price of the column size, "
    "averaged across sector-quarters of the estimation sample (specification of Table \\ref{tab:results})."))

me_w = demand("marginal_effects_demographics.csv").set_index("Choice")
body = " & " + bold_row(SIZES) + " \\\\ \\hline \\noalign{\\vskip 1mm}\n"
for var in me_w.columns:
    body += LABELS[var] + " & " + " & ".join(f"{100 * x:.4f}" for x in me_w[var]) + " \\\\\n"
write("tab_me_demographics.tex", table(
    "Marginal Effects of Sector Characteristics", "tab:marginal_effects_demographic_characteristics",
    "lcccc", body,
    "Mean across sector-quarters of Equation (\\ref{eq:marginal_effect_demographics}): change in the quarterly "
    "probability of adopting each size, in percentage points, for a one-unit increase in each variable "
    "(for shares, from 0 to 1)."))

## COUNTERFACTUALS

res = {s: reg(f"results_{s}.csv") for s in ["main", "het", "nofe"]}


def by_design(r, value):
    return r[r.design != "none"].pivot(index="budget", columns="design", values=value)


# size-only subsidies and adoptions (main)
r = res["main"]
sub = reg("subsidies_size_main.csv").set_index("budget")
gen_head = "\\textbf{\\begin{tabular}[c]{@{}c@{}}Generation\\\\ (MWh/year)\\end{tabular}}"
body = (" & \\multicolumn{4}{c}{\\textit{Subsidy ($10^4$ BRL)}} & \\multicolumn{4}{c}{\\textit{Adoptions}} & \\\\ "
        "\\cline{2-5} \\cline{6-9}\n\\textbf{Budget} & " + bold_row(SIZES) + " & " + bold_row(SIZES)
        + f" & {gen_head} \\\\ \\hline \\noalign{{\\vskip 1mm}}\n")
for _, x in r[r.design.isin(["none", "size"])].iterrows():
    c = sub.loc[x.budget].to_numpy() if x.budget > 0 else np.zeros(4)
    body += (f"{x.budget / 100:,.0f} & " + " & ".join(f"{v:.2f}" for v in c) + " & "
             + " & ".join(f"{x[f'adoptions_{s}']:,.0f}" for s in ['2kW', '4kW', '6kW', '8kW'])
             + f" & {x.generation:,.0f} \\\\\n")
write("tab_cf1.tex", table(
    "Optimal Subsidies by Installation Size", "tab:cf1", "cccccccccc", body,
    "Budgets in millions and subsidies in $10^4$ BRL per installation (December 2019 BRL). Budget $=0$ is the "
    "model prediction without subsidies (observed adoptions: 3,430, 7,624, 7,861 and 1,515). Adoptions are "
    "totals over the 14 quarters in all 29,942 sectors; generation is their yearly output."))

# subsidies by size x irradiation quartile (main)
irr = reg("subsidies_irr_main.csv")
irr = irr[irr.budget.isin(SHOW_BUDGETS)]
body = "\\textbf{Budget} & \\textbf{Size} & \\textbf{Q1} & \\textbf{Q2} & \\textbf{Q3} & \\textbf{Q4} \\\\ \\hline \\noalign{\\vskip 1mm}\n"
for k, (B, grp) in enumerate(irr.groupby("budget", sort=False)):
    grp = grp[grp[["Q1", "Q2", "Q3", "Q4"]].sum(axis=1) > 0]
    for i, x in enumerate(grp.itertuples()):
        lead = f"\\multirow{{{len(grp)}}}{{*}}{{{B / 100:,.0f}}}" if i == 0 else ""
        body += f"{lead} & {x.size.replace('kW', ' kW')} & {x.Q1:.2f} & {x.Q2:.2f} & {x.Q3:.2f} & {x.Q4:.2f} \\\\\n"
    if k < len(SHOW_BUDGETS) - 1:
        body += "\\noalign{\\vskip 1mm}\n"
write("tab_cf2.tex", table(
    "Optimal Subsidies by Installation Size and Irradiation Quartile", "tab:cf2", "cccccc", body,
    "Subsidies in $10^4$ BRL per installation by quartile of the sector's solar generation potential "
    "(Q1 = least sunny). Sizes that receive no subsidy in any quartile are omitted. Budgets in millions of BRL."))


# comparison of designs (main and heterogeneous price sensitivity)
def comparison_panel(r, title):
    G0 = r.loc[r.design == "none", "generation"].iloc[0]
    gen = by_design(r, "generation")
    out = (f"\\multicolumn{{7}}{{l}}{{\\textit{{{title}}}}} \\\\\n"
           "\\textbf{Budget} & " + bold_row([lab for _, lab in DESIGNS]) + " \\\\ \\hline \\noalign{\\vskip 1mm}\n")
    out += "\\multicolumn{7}{l}{Increase in yearly generation relative to no subsidy (\\%)} \\\\\n"
    for B, row in gen.iterrows():
        out += f"{B / 100:,.0f} & " + " & ".join(f"{100 * (row[d] / G0 - 1):.1f}" for d, _ in DESIGNS) + " \\\\\n"
    out += "\\multicolumn{7}{l}{Public cost per additional MWh/year (BRL)} \\\\\n"
    for B, row in gen.iterrows():
        out += f"{B / 100:,.0f} & " + " & ".join(f"{B * 1e4 / (row[d] - G0):,.0f}" for d, _ in DESIGNS) + " \\\\\n"
    return out


body = (comparison_panel(res["main"], "Panel A: common price sensitivity (Table \\ref{tab:results})")
        + "\\noalign{\\vskip 2mm}\n"
        + comparison_panel(res["het"], "Panel B: price sensitivity by income quartile (Table \\ref{tab:specs}, column 3)"))
write("tab_comparison.tex", table(
    "Generation Under Alternative Subsidy Designs", "tab:comparison", "ccccccc", body,
    "Uniform: one subsidy for all. Size: one subsidy per size. Size \\& irr.: per size and irradiation quartile "
    "(16 subsidies). Size \\& dem.: per size and quartiles of density, medium-income share and bathrooms (256 "
    "subsidies). Prop.: subsidy proportional to the expected yearly generation of the installation, "
    "$a\\,\\kappa_j I_m$. Rule: $\\max\\{0, a\\,\\kappa_j I_m - 1/|\\beta_m|\\}$, Equation (\\ref{eq:approx}). In the "
    "last two, $a$ is set to exhaust the budget. Budgets in millions of December 2019 BRL. Generation without "
    f"subsidies: {res['main'].generation.iloc[0]:,.0f} MWh/year (Panel A) and "
    f"{res['het'].generation.iloc[0]:,.0f} MWh/year (Panel B)."))

# appendix: no fixed effects
write("tab_comparison_nofe.tex", table(
    "Generation Under Alternative Subsidy Designs, No Quarter Fixed Effects", "tab:comparison_nofe", "ccccccc",
    comparison_panel(res["nofe"], "Specification of Table \\ref{tab:specs}, column 1"),
    "See notes to Table \\ref{tab:comparison}. Generation without subsidies: "
    f"{res['nofe'].generation.iloc[0]:,.0f} MWh/year."))
