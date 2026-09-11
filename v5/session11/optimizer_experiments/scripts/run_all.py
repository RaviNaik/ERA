"""Reproduce everything: (re)build the notebooks, execute them top-to-bottom into
the Aim repo, then print the run summary.

    uv run python scripts/run_all.py            # all five notebooks
    uv run python scripts/run_all.py 4 5        # just notebooks 4 and 5

Notebook 5 (the LR sweep) is the long one (~20 min on a laptop GPU).
"""
from __future__ import annotations

import subprocess
import sys
import os

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
NB_DIR = os.path.join(ROOT, "notebooks")

NOTEBOOKS = {
    "1": "01_adam_by_hand.ipynb",
    "2": "02_bias_correction.ipynb",
    "3": "03_update_weight_ratio_warmup.ipynb",
    "4": "04_cosine_vs_wsd.ipynb",
    "5": "05_lr_sweep_width.ipynb",
}
TIMEOUTS = {"1": 600, "2": 600, "3": 1200, "4": 1800, "5": 3600}


def main(which):
    subprocess.run([sys.executable, os.path.join(HERE, "build_notebooks.py")], check=True)
    for key in which:
        nb = NOTEBOOKS[key]
        print(f"\n=== executing {nb} ===", flush=True)
        subprocess.run(
            [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
             "--inplace", f"--ExecutePreprocessor.timeout={TIMEOUTS[key]}", nb],
            cwd=NB_DIR, check=True,
        )
    print("\n=== run summary ===", flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, "aim_summary.py")], check=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a in NOTEBOOKS]
    main(args or list(NOTEBOOKS))
