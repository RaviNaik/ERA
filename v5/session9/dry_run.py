"""Concatenate all code cells from build_notebook.py's CELLS list and exec
them as one script, with N_STEPS patched down to 5 for a fast correctness
pass before committing to the full nbconvert run.
"""
import sys
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")  # non-interactive: plt.show() must not block outside a real kernel
import build_notebook as bn

code_cells = [src for ctype, src in bn.CELLS if ctype == "code"]
full_source = "\n\n".join(code_cells)
full_source = full_source.replace("N_STEPS = 300", "N_STEPS = 5")

g = {"__name__": "__main__"}
exec(compile(full_source, "notebook_dryrun", "exec"), g)
print("\n\n=== DRY RUN COMPLETED WITHOUT ERROR ===")
