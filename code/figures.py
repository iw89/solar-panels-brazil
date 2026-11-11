"""Descriptive figures of the paper: new adoptions and installation prices per quarter, by size.

    python code/figures.py

Writes draft/figs/newpv.pdf and draft/figs/prices.pdf from data/data_population.parquet.
(The data-fit figures are written by estimate.py; the study-area map draft/figs/MG.png is a
fixed input.)
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import paths
from specs import PRICES, Y_NAMES

LABELS = ["2 kW", "4 kW", "6 kW", "8 kW"]
COLORS = ["tab:blue", "tab:red", "tab:green", "tab:orange"]

d = pd.read_parquet(paths.data("data_population.parquet"))
adoptions = d.groupby("tq")[Y_NAMES].sum()
prices = d.groupby("tq")[PRICES].mean()        # prices vary by size and quarter only


def plot(table, title, ylabel, name):
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    for col, label, color in zip(table.columns, LABELS, COLORS):
        ax.plot(range(len(table)), table[col], color=color, lw=1.6, label=label)
    ax.set_xticks(range(len(table)))
    ax.set_xticklabels(table.index, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(paths.FIGS, name))
    plt.close(fig)
    print("wrote", name)


plot(adoptions, "New PV adoptions per quarter", "Adoptions", "newpv.pdf")
plot(prices, "Installation prices", "$10^4$ BRL (Dec. 2019)", "prices.pdf")
