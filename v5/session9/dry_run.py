"""Fast end-to-end correctness pass over the notebook's code cells.

Sets S9_SMOKE=1 (tiny model, WikiText-2 subset, 20 steps) and hides the GPU so
the peak-memory section takes its CPU path -- the whole thing runs in ~2 min on
a laptop. Use this to confirm the notebook executes top to bottom before
committing to the real nbconvert run on a GPU server.

    python dry_run.py
"""
import os
import sys

os.environ["S9_SMOKE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force the CPU path in the memory section

sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")  # non-interactive: plt.show() must not block outside a real kernel
import build_notebook as bn

code_cells = [src for ctype, src in bn.CELLS if ctype == "code"]
full_source = "\n\n".join(code_cells)

g = {"__name__": "__main__"}
exec(compile(full_source, "notebook_dryrun", "exec"), g)
print("\n\n=== DRY RUN COMPLETED WITHOUT ERROR ===")
