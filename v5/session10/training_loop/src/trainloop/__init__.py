"""Session 10 — The Training Loop.

A small GPT and a real training loop, instrumented so the loop tells you the
truth about itself. See the notebook `session10_training_loop.ipynb` and the
per-topic modules:

    model      - a compact nanoGPT-style decoder
    data       - char-level Tiny Shakespeare + variable-length micro-batches
    shapes     - one step with every tensor shape printed and named
    gradcheck  - one gradient verified by finite differences vs. backward()
    accum      - gradient accumulation done right vs. the average-of-averages bug
    gradnorm   - per-step grad-norm logging + a spike that leads the loss
    mfu        - Model FLOPs Utilisation, computed honestly
    floats     - 0.1 written out in fp32, bf16 and fp8 E4M3, bit by bit
"""

from . import accum, data, floats, gradcheck, gradnorm, mfu, model, shapes, utils

__all__ = [
    "accum",
    "data",
    "floats",
    "gradcheck",
    "gradnorm",
    "mfu",
    "model",
    "shapes",
    "utils",
]
