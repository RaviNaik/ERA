"""Log the grad norm at every step, then find a step where it moved first.

The claim from the session: the grad norm reacts to trouble *before* the loss
does. This module runs a short training loop that logs, every single step:

  - train loss (raw) and an EMA-smoothed loss
  - the global grad norm (sqrt of the sum of squares of every gradient)

At one chosen step it feeds a single corrupted batch (random targets). That
batch produces a large gradient -> the grad-norm trace spikes immediately. The
optimizer then takes one bad step, and the *loss* only rises on the following
normal batch. `find_lead` locates that offset.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch

from .data import CharDataset, get_batch


@dataclass
class GradNormConfig:
    batch_size: int = 16
    seq_len: int = 128
    n_steps: int = 80
    lr: float = 3e-3
    ema: float = 0.25
    inject_step: int = 40
    clip: float | None = None  # set a float to also demo clipping


@dataclass
class GradNormTrace:
    step: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    loss_ema: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    injected: int = -1


def global_grad_norm(model) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum().item())
    return sq ** 0.5


def run(model, ds: CharDataset, cfg: GradNormConfig, device) -> GradNormTrace:
    model = copy.deepcopy(model).to(device).train()  # don't mutate the caller's model
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    tr = GradNormTrace(injected=cfg.inject_step)
    gen = torch.Generator().manual_seed(7)
    ema = None

    for step in range(cfg.n_steps):
        x, y = get_batch(ds, "train", cfg.batch_size, cfg.seq_len, device, generator=gen)
        if step == cfg.inject_step:
            # one corrupted batch: targets replaced with noise
            y = torch.randint(0, ds.vocab_size, y.shape, generator=gen).to(device)

        opt.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y)
        loss.backward()
        gnorm = global_grad_norm(model)
        if cfg.clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step()

        lval = float(loss.item())
        ema = lval if ema is None else (1 - cfg.ema) * ema + cfg.ema * lval
        tr.step.append(step)
        tr.loss.append(lval)
        tr.loss_ema.append(ema)
        tr.grad_norm.append(gnorm)

    return tr


def find_lead(tr: GradNormTrace, window: int = 10, loss_rise: float = 0.05) -> dict:
    """When did the grad norm spike, and when did the smoothed loss actually rise?

    The injected batch has a large gradient, so the grad-norm trace spikes on the
    injection step itself. The optimizer then takes one bad step; the smoothed
    loss only climbs on the following normal batches. We locate:

      grad_spike : argmax of the grad norm in a window around the injection
      loss_rise  : first step at/after the injection where the EMA loss exceeds
                   its pre-injection value by `loss_rise` (fraction)
    """
    import numpy as np

    gn = np.array(tr.grad_norm)
    ls = np.array(tr.loss_ema)
    inj = tr.injected
    if inj < 1 or inj >= len(gn):
        return {"error": "injection step out of range"}

    lo, hi = max(inj - 3, 0), min(inj + 3, len(gn))
    grad_spike = lo + int(np.argmax(gn[lo:hi]))

    # The injected step's own forward loss is high by construction (corrupted
    # targets). The honest question is when the loss on *subsequent normal
    # batches* degrades -- so the search starts one step after the injection.
    ref = ls[inj - 1]
    loss_rise_step = next(
        (i for i in range(inj + 1, min(inj + window, len(ls))) if ls[i] > ref * (1 + loss_rise)),
        None,
    )

    return {
        "injection step": inj,
        "grad norm at injection-1 / injection": f"{gn[inj-1]:.3f} -> {gn[inj]:.3f}",
        "loss_ema at injection-1": f"{ref:.3f}",
        "grad_norm spike at step": grad_spike,
        "loss_ema rose (>{:.0%}) at step".format(loss_rise): loss_rise_step,
        "grad norm led the loss by (steps)": (
            None if loss_rise_step is None else loss_rise_step - grad_spike
        ),
    }


def plot(tr: GradNormTrace, lead: dict, path: str) -> str:
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(tr.step, tr.loss, color="#8250df", alpha=0.35, label="train loss (raw)")
    ax1.plot(tr.step, tr.loss_ema, color="#8250df", lw=2, label="train loss (EMA)")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss", color="#8250df")
    ax2 = ax1.twinx()
    ax2.plot(tr.step, tr.grad_norm, color="#cf222e", lw=2, label="grad norm")
    ax2.set_ylabel("grad norm", color="#cf222e")

    if tr.injected >= 0:
        ax1.axvline(tr.injected, color="k", ls=":", label=f"injected batch @ {tr.injected}")
    gs = lead.get("grad_norm spike at step")
    lsp = next((v for k, v in lead.items() if k.startswith("loss_ema rose")), None)
    if gs is not None:
        ax2.axvline(gs, color="#cf222e", ls="--", alpha=0.7)
    if lsp is not None:
        ax1.axvline(lsp, color="#8250df", ls="--", alpha=0.7)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=8)
    ax1.set_title("grad norm spikes at step {}, smoothed loss only at step {}".format(gs, lsp))
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path
