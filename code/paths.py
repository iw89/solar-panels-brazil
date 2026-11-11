"""Folder locations used by every script."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")        # input data
OUTPUT = os.path.join(ROOT, "output")    # intermediate results (estimates, counterfactuals)
DRAFT = os.path.join(ROOT, "draft")      # the paper
FIGS = os.path.join(DRAFT, "figs")
TABLES = os.path.join(DRAFT, "tables")

for folder in (OUTPUT, FIGS, TABLES):
    os.makedirs(folder, exist_ok=True)


def data(name):
    return os.path.join(DATA, name)


def output(name):
    return os.path.join(OUTPUT, name)
