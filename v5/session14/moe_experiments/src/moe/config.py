"""Experiment presets.

``base`` is the real GPU-server configuration; ``small`` fits a laptop GPU;
``smoke`` is tiny and only exists to exercise every code path in seconds.

Select one with the ``MOE_PRESET`` environment variable (default ``base``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict


@dataclass
class ExpConfig:
    name: str = "base"
    # ---- model (dense) ----
    vocab_size: int = 50257          # GPT-2 BPE; models are built with the dataset's own vocab size (50257)
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = True
    # ---- MoE ----
    n_experts: int = 8
    top_k: int = 2
    moe_every: int = 2               # every 2nd block gets an MoE MLP, the rest stay dense
    score_fn: str = "softmax"        # router score function: softmax | sigmoid
    lb_coef: float = 1e-4            # tiny Switch-style balance loss (a backstop; the bias does the work)
    bias_update: float = 1e-3        # loss-free balancing: per-step step size of the selection bias (0 = off)
    z_coef: float = 0.001            # router z-loss weight
    explore_steps: int = 300         # after conversion: sample experts probabilistically for this many steps
    explore_noise: float = 1.0       # Gumbel temperature during that window (1.0 = sample ~ scores)
    # ---- phase 1: dense pre-training (also: MoE-from-scratch uses the same budget) ----
    batch_size: int = 32
    dense_steps: int = 6000
    dense_lr: float = 6e-4
    dense_warmup: int = 300
    # ---- phase 2: continue after the dense -> MoE conversion (or keep dense, as a control) ----
    cont_steps: int = 3000
    cont_lr: float = 3e-4            # lower peak LR than phase 1: a gentle re-warm
    cont_warmup: int = 100
    min_lr_frac: float = 0.1
    weight_decay: float = 0.1
    # ---- loop ----
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 10
    seed: int = 1337
    # ---- data ----
    dataset: str = "wikitext103"     # wikitext103 | synthetic

    def to_dict(self):
        return asdict(self)


PRESETS = {
    "base": ExpConfig(),
    "small": ExpConfig(name="small", block_size=256, n_layer=6, n_head=6, n_embd=384,
                       batch_size=16, dense_steps=1500, cont_steps=750, explore_steps=75,
                       eval_interval=100, eval_iters=25),
    "smoke": ExpConfig(name="smoke", vocab_size=512, block_size=32, n_layer=4, n_head=2, n_embd=64,
                       n_experts=4, top_k=2, moe_every=2, batch_size=8,
                       dense_steps=12, dense_warmup=3, cont_steps=8, cont_warmup=2, explore_steps=3, bias_update=1e-2,
                       eval_interval=4, eval_iters=3, log_interval=1, dataset="synthetic"),
}


def get_config(name: str | None = None) -> ExpConfig:
    name = name or os.environ.get("MOE_PRESET", "base")
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {list(PRESETS)}")
    return PRESETS[name]
