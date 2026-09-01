"""Session 10 — The Training Loop.

Just the plumbing lives here; every experiment is done by hand in the notebook
`session10_training_loop.ipynb`.

    model   - a compact nanoGPT-style decoder
    data    - char-level Tiny Shakespeare + fixed and variable-length batches
    utils   - seeding, device pick, the S10_SMOKE switch
"""

from . import data, model, utils

__all__ = ["data", "model", "utils"]
