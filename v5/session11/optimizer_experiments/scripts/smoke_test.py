"""Fast correctness check — NOT a training run.

Exercises every code path (model, data, hand-Adam, schedules, trainer, Aim
tracking, update/weight-ratio logging) with a tiny model and 2-3 steps so it
finishes in seconds. Use this to verify the codebase before running the real
notebooks (which use the full step counts / sweeps) on bigger hardware.

    uv run python scripts/smoke_test.py
"""
from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from s11.data import load_char_dataset
from s11.model import GPT, GPTConfig, param_count_report
from s11.optim import run_adam_by_hand, HandAdamW, lr_cosine, lr_wsd, make_schedule
from s11.trainer import TrainConfig, train, loss_at_step
from s11.utils import get_device, set_seed

OK = "  ✓"


def check(label, cond):
    print(f"{OK if cond else '  FAIL'}  {label}")
    if not cond:
        raise SystemExit(f"smoke test failed: {label}")


def main():
    set_seed(0)
    device = get_device()
    print(f"device = {device}\n")

    # --- 1. data -----------------------------------------------------------
    data = load_char_dataset()
    check("dataset loads", data.vocab_size > 0 and len(data.train_ids) > 0)
    x, y = data.get_batch("train", batch_size=4, block_size=16, device=device)
    check("get_batch shapes correct", tuple(x.shape) == (4, 16) and tuple(y.shape) == (4, 16))

    # --- 2. model ------------------------------------------------------------
    cfg = GPTConfig(vocab_size=data.vocab_size, block_size=16, n_layer=2, n_head=2,
                    n_embd=32, dropout=0.0, bias=True)
    model = GPT(cfg).to(device)
    logits, loss = model(x, y)
    check("forward shapes correct", tuple(logits.shape) == (4, 16, data.vocab_size))
    check("loss is finite scalar", torch.isfinite(loss).item())
    report = param_count_report(GPTConfig(vocab_size=65, n_embd=384, n_layer=6, n_head=6, bias=True))
    check(f"10.77M base config reproduces (got {report['total']/1e6:.3f}M)",
          abs(report["total"] - 10_770_816) < 1)

    # --- 3. hand Adam vs torch (task 1 logic) ---------------------------------
    w0, grads = 0.5, [0.1, -0.2, 0.05, 0.3, -0.1]
    trace = run_adam_by_hand(w0, grads, lr=1e-3, beta1=0.9, beta2=0.999)
    p = torch.tensor([w0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([p], lr=1e-3, betas=(0.9, 0.999))
    for g in grads:
        opt.zero_grad(); p.grad = torch.tensor([g], dtype=torch.float64); opt.step()
    diff = abs(trace[-1]["param_after"] - p.item())
    check(f"hand Adam matches torch.optim.Adam (diff={diff:.2e})", diff < 1e-10)

    # --- 4. schedules ----------------------------------------------------------
    check("cosine warmup ramps linearly", math.isclose(lr_cosine(4, base_lr=1e-3, warmup=10, total=100), 5e-4, rel_tol=1e-9))
    check("wsd holds plateau", lr_wsd(50, base_lr=1e-3, warmup=10, total=100, decay_frac=0.2) == 1e-3)
    sched = make_schedule("wsd", base_lr=1e-3, warmup=5, total=40, decay_frac=0.25)
    check("wsd decays near the end", sched(39) < sched(30))

    # --- 5. HandAdamW bias-correction switch ------------------------------------
    lin = torch.nn.Linear(4, 4).to(device)
    o1 = HandAdamW(lin.parameters(), lr=1e-2, bias_correction=True)
    o2 = HandAdamW(lin.parameters(), lr=1e-2, bias_correction=False)
    check("HandAdamW constructs with both bias_correction settings", True)

    # --- 6. trainer.train end-to-end, with Aim, in a throwaway repo ------------
    tmp_repo = tempfile.mkdtemp(prefix="s11_smoke_aim_")
    try:
        tcfg = TrainConfig(
            block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0, bias=True,
            base_lr=1e-3, weight_decay=0.1, schedule="cosine", warmup=1, total_steps=3,
            batch_size=4, eval_interval=1, eval_iters=2, ratio_log_interval=1,
            device=device, label="smoke",
        )
        hist = train(data, tcfg, aim_repo=tmp_repo, experiment="smoke_test")
        check("train() returns loss history", len(hist["train_loss_step"]) == 3)
        check("val loss recorded", len(hist["val_loss"]) >= 1)
        check("update/weight ratio logged per layer",
              "L0.attn" in hist["ratio_by_layer"] and "layernorm" in hist["ratio_by_layer"])
        check("loss_at_step interpolates", isinstance(loss_at_step(hist, 1), float))
        check("Aim run directory created", os.path.isdir(os.path.join(tmp_repo, ".aim")))
        check("results/runs.jsonl written", os.path.exists(os.path.join(tmp_repo, "results", "runs.jsonl")))

        # WSD schedule path through the real trainer too
        tcfg2 = TrainConfig(**{**tcfg.__dict__, "schedule": "wsd", "wsd_decay_frac": 0.34, "label": "smoke_wsd"})
        hist2 = train(data, tcfg2, aim_repo=tmp_repo, experiment="smoke_test")
        check("WSD schedule runs end-to-end", len(hist2["train_loss_step"]) == 3)
    finally:
        shutil.rmtree(tmp_repo, ignore_errors=True)

    # --- 7. matplotlib figure path (what savefig() exercises) ------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    tmp_png = tempfile.mktemp(suffix=".png")
    fig.savefig(tmp_png)
    check("matplotlib figure saves", os.path.exists(tmp_png))
    os.remove(tmp_png)
    plt.close(fig)

    print("\nAll smoke checks passed. This validates code correctness with a tiny "
          "model (32-d, 2 layers, 3 steps) — it does NOT run the real experiments. "
          "Run `uv run python scripts/run_all.py` for the real notebooks.")


if __name__ == "__main__":
    main()
