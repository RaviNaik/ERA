"""Sparse upcycling: turn a trained dense GPT into an MoE GPT.

Every selected MLP becomes ``n_experts`` identical copies of itself.  The router is
freshly initialised (small normal).  Because gates are renormalised over the top-k
experts and all experts are the same function, the converted network computes
*exactly* the same logits as the dense one — training continues from the same loss.
"""
from __future__ import annotations

from dataclasses import replace

import torch

from .model import GPT, GPTConfig, is_moe_layer


@torch.no_grad()
def upcycle(dense: GPT, n_experts: int = 8, top_k: int = 2, moe_every: int = 2,
            expert_noise: float = 0.0, router_std: float = 0.02, drop_r: float = 0.0,
            score_fn: str | None = None, seed: int = 0) -> GPT:
    """Return a new MoE GPT initialised from ``dense``.

    expert_noise: std of Gaussian noise added to each expert copy, *relative to that
        weight matrix's own std* (0 = exact copies).  Breaks the symmetry between
        experts at the price of a small initial loss bump.
    drop_r: "drop-upcycling" - fraction of each expert's hidden neurons re-drawn from a fresh
        random init (a different random subset per expert).  0 = plain copies (lossless);
        larger values make experts diverge faster but cost some loss at the moment of conversion.
    """
    assert dense.cfg.n_experts == 1, "source model must be dense"
    g = torch.Generator().manual_seed(seed)
    cfg = replace(dense.cfg, n_experts=n_experts, top_k=top_k, moe_every=moe_every,
                  score_fn=score_fn or dense.cfg.score_fn)
    moe = GPT(cfg).to(next(dense.parameters()).device)
    src = dict(dense.named_parameters())

    copied, converted = [], []
    for name, p in moe.named_parameters():
        if name in src and src[name].shape == p.shape:        # embeddings, attention, norms, non-MoE MLPs
            p.copy_(src[name])
            copied.append(name)

    def noisy(w: torch.Tensor) -> torch.Tensor:
        if expert_noise <= 0:
            return w.clone()
        n = torch.randn(w.shape, generator=g).to(w.device, w.dtype)
        return w + expert_noise * w.std() * n

    for i, blk in enumerate(moe.transformer.h):
        if not is_moe_layer(cfg, i):
            continue
        d = dense.transformer.h[i].mlp
        m = blk.mlp
        for e in range(n_experts):
            m.w1[e].copy_(noisy(d.c_fc.weight))
            m.w2[e].copy_(noisy(d.c_proj.weight))
            if m.b1 is not None:
                m.b1[e].copy_(d.c_fc.bias)
                m.b2[e].copy_(d.c_proj.bias)
        if drop_r > 0:                                             # redraw a random subset of neurons per expert
            h = m.w1.shape[1]
            for e in range(n_experts):
                idx = torch.randperm(h, generator=g)[: int(round(drop_r * h))].to(m.w1.device)
                fresh1 = torch.randn(len(idx), m.w1.shape[2], generator=g).to(m.w1) * d.c_fc.weight.std()
                fresh2 = torch.randn(m.w2.shape[1], len(idx), generator=g).to(m.w2) * d.c_proj.weight.std()
                m.w1[e, idx] = fresh1
                m.w2[e][:, idx] = fresh2
                if m.b1 is not None:
                    m.b1[e, idx] = 0.0
        m.router.weight.copy_(torch.randn(m.router.weight.shape, generator=g).to(m.router.weight) * router_std)
        converted.append(i)
    moe.upcycle_info = {"copied_tensors": len(copied), "converted_layers": converted}
    return moe


@torch.no_grad()
def max_logit_diff(a: GPT, b: GPT, idx: torch.Tensor) -> float:
    """Largest |logit_a - logit_b| over a batch (both in eval mode, fp32)."""
    a.eval(); b.eval()
    return (a(idx)[0] - b(idx)[0]).abs().max().item()
