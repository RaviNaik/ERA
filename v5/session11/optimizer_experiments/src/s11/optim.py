"""Hand-rolled Adam / AdamW and the learning-rate schedules used in the tasks.

Everything here is deliberately explicit so the notebooks can print every
intermediate quantity (m, v, m_hat, v_hat, the step) and diff it against
``torch.optim.AdamW``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


# ---------------------------------------------------------------------------
# 1. Adam by hand — pure-python scalar/tensor reference
# ---------------------------------------------------------------------------
@dataclass
class AdamState:
    m: float = 0.0
    v: float = 0.0
    t: int = 0


def adam_step(
    param: float,
    grad: float,
    state: AdamState,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    bias_correction: bool = True,
    decoupled: bool = True,
):
    """One Adam/AdamW update on a *scalar* parameter, returning every intermediate.

    ``decoupled=True`` + ``weight_decay>0`` == AdamW (decoupled weight decay).
    ``decoupled=False`` == classic Adam with L2 regularisation folded into the grad.
    """
    state.t += 1
    t = state.t

    g = grad
    if weight_decay > 0 and not decoupled:
        g = g + weight_decay * param

    state.m = beta1 * state.m + (1 - beta1) * g
    state.v = beta2 * state.v + (1 - beta2) * (g * g)

    if bias_correction:
        m_hat = state.m / (1 - beta1 ** t)
        v_hat = state.v / (1 - beta2 ** t)
    else:
        m_hat = state.m
        v_hat = state.v

    step = lr * m_hat / (math.sqrt(v_hat) + eps)

    new_param = param - step
    if weight_decay > 0 and decoupled:
        new_param = new_param - lr * weight_decay * param

    return {
        "t": t,
        "grad_used": g,
        "m": state.m,
        "v": state.v,
        "m_hat": m_hat,
        "v_hat": v_hat,
        "step": step,
        "param_before": param,
        "param_after": new_param,
    }


def run_adam_by_hand(
    w0: float,
    grads: list[float],
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    bias_correction: bool = True,
    decoupled: bool = True,
) -> list[dict]:
    """Apply a sequence of gradients to a single weight; return the per-step trace."""
    state = AdamState()
    w = w0
    trace = []
    for g in grads:
        rec = adam_step(w, g, state, lr, beta1, beta2, eps,
                        weight_decay, bias_correction, decoupled)
        w = rec["param_after"]
        trace.append(rec)
    return trace


# ---------------------------------------------------------------------------
# 2. A minimal tensor AdamW that exposes bias-correction as a flag
# ---------------------------------------------------------------------------
class HandAdamW(torch.optim.Optimizer):
    """Same math as ``torch.optim.AdamW`` but with a ``bias_correction`` switch
    and per-param-group logging of the update-to-weight ratio."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, bias_correction=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        bias_correction=bias_correction)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            bc = group["bias_correction"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                st["step"] += 1
                t = st["step"]
                m, v = st["exp_avg"], st["exp_avg_sq"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                if bc:
                    mh = m / (1 - b1 ** t)
                    vh = v / (1 - b2 ** t)
                else:
                    mh, vh = m, v
                update = lr * mh / (vh.sqrt().add_(eps))
                p.add_(-update)
                if wd > 0:
                    p.add_(p, alpha=-lr * wd)
        return loss


# ---------------------------------------------------------------------------
# 3. Learning-rate schedules
# ---------------------------------------------------------------------------
def lr_cosine(step: int, *, base_lr: float, warmup: int, total: int,
              min_lr_frac: float = 0.1) -> float:
    """Linear warmup then cosine decay to ``min_lr_frac * base_lr``."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= total:
        return base_lr * min_lr_frac
    progress = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_frac + (1 - min_lr_frac) * coeff)


def lr_wsd(step: int, *, base_lr: float, warmup: int, total: int,
           decay_frac: float = 0.2, min_lr_frac: float = 0.0,
           decay_shape: str = "linear") -> float:
    """Warmup-Stable-Decay: linear warmup, flat plateau at ``base_lr``, then a
    short decay over the last ``decay_frac`` of training."""
    decay_steps = max(1, int(decay_frac * total))
    decay_start = total - decay_steps
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step < decay_start:
        return base_lr
    if step >= total:
        return base_lr * min_lr_frac
    frac = (step - decay_start) / decay_steps  # 0 -> 1
    if decay_shape == "linear":
        mult = 1.0 - frac
    elif decay_shape == "cosine":
        mult = 0.5 * (1 + math.cos(math.pi * frac))
    elif decay_shape == "1-sqrt":
        mult = 1.0 - math.sqrt(frac)
    else:
        raise ValueError(decay_shape)
    return base_lr * (min_lr_frac + (1 - min_lr_frac) * mult)


def make_schedule(kind: str, **kw):
    if kind == "cosine":
        return lambda s: lr_cosine(s, **kw)
    if kind == "wsd":
        return lambda s: lr_wsd(s, **kw)
    if kind == "constant":
        base = kw["base_lr"]
        warmup = kw.get("warmup", 0)
        return lambda s: base * (s + 1) / warmup if (warmup and s < warmup) else base
    raise ValueError(kind)
