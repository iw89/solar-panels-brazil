"""Replication package: generates every table and figure of the paper.

    python main.py                  run everything (about 20 minutes)
    python main.py tables figures   run selected steps only

Steps, in order:
    estimate         demand model, all specifications          -> output/, draft/figs/datafit_*.png
    counterfactuals  subsidy designs for main, het, nofe        -> output/results_*.csv, subsidies_*.csv
    figures          descriptive figures                       -> draft/figs/newpv.pdf, prices.pdf
    tables           LaTeX tables                              -> draft/tables/*.tex
    pdf              compiles draft/main.tex (needs tectonic or latexmk on the PATH; skipped otherwise)
"""

import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(ROOT, "code")
DRAFT = os.path.join(ROOT, "draft")


def run_script(script, *args):
    cmd = [sys.executable, os.path.join(CODE, script), *args]
    print(f"\n>>> {' '.join(['python', 'code/' + script, *args])}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"})


def compile_pdf():
    if shutil.which("tectonic"):
        cmd = ["tectonic", "main.tex"]
    elif shutil.which("latexmk"):
        cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"]
    else:
        print("\n>>> pdf: no tectonic or latexmk found; compile draft/main.tex yourself")
        return
    print(f"\n>>> {' '.join(cmd)}  (in draft/)", flush=True)
    subprocess.run(cmd, check=True, cwd=DRAFT)


STEPS = {
    "estimate": lambda: run_script("estimate.py"),
    "counterfactuals": lambda: [run_script("counterfactuals.py", s) for s in ["main", "het", "nofe"]],
    "figures": lambda: run_script("figures.py"),
    "tables": lambda: run_script("tables.py"),
    "pdf": compile_pdf,
}

if __name__ == "__main__":
    steps = sys.argv[1:] or list(STEPS)
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        sys.exit(f"unknown step(s) {unknown}; choose from {list(STEPS)}")
    t0 = time.time()
    for step in STEPS:                       # always in pipeline order
        if step in steps:
            STEPS[step]()
    print(f"\nDone in {(time.time() - t0) / 60:.1f} minutes.")
