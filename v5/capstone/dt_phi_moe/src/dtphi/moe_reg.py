"""Regulariser terms for a mixture-of-experts model under the official Deletion Tuning code (files unmodified).

Official `sum_sparse_weight_norms` adds, for every SparseLinear, the term stored by its *last forward*. In a dense model
every layer runs in every step. In a MoE model an expert that gets no tokens in a step does not run, so its stored term
is from an earlier step: its graph is already freed and the next backward fails with "Trying to backward through the
graph a second time". The official code has a window for this (`open_/close_sparse_weight_norm_window`, written for the
GRPO actor; the SFT script does not use it). Here:
  - a forward pre-hook on the model opens a new window for each training forward, so each stored term is tagged with it;
  - the replacement sum uses the tagged (fresh) terms, and for layers that did not run computes the same term,
    ||W * mask||^2 with the layer's current mask, as SparseLinear.forward does (so every layer is in the regulariser in
    every step, as in a dense model);
  - it then closes the window, so the recomputation during backward (gradient checkpointing) stores nothing.
"""
from __future__ import annotations

import torch

from learnable_sparsity import sparse_linear as sl
from learnable_sparsity.sparse_linear import SparseLinear


def install(model: torch.nn.Module, train_sft, loss_device: torch.device):
    """Replace train_sft.sum_sparse_weight_norms; returns (restore function, stats dict)."""
    original = train_sft.sum_sparse_weight_norms
    stats = {"sums": 0, "idle_terms": 0, "layers": 0}

    def open_window(module, args, kwargs):
        if module.training and torch.is_grad_enabled():
            sl.open_sparse_weight_norm_window()

    handle = model.register_forward_pre_hook(open_window, with_kwargs=True)

    def fresh_sum(model: torch.nn.Module) -> torch.Tensor:
        token = sl._ACTIVE_FORWARD_TOKEN[0]
        total = torch.zeros((), device=loss_device)
        idle = layers = 0
        for m in model.modules():
            if not isinstance(m, SparseLinear):
                continue
            layers += 1
            if token is not None and hasattr(m, "sparse_weight_norm") and getattr(m, "_sparse_weight_norm_token", None) is token:
                term = m.sparse_weight_norm
            else:  # this layer did not run in the current forward (an expert without tokens)
                mask = m.diff_mask(target_shape=m.weight.shape, dtype=m.weight.dtype).to(m.weight.device)
                term = (m.weight.detach() * mask).pow(2).sum()
                idle += 1
            total = total + term.float().to(loss_device)
        sl.close_sparse_weight_norm_window()
        stats["sums"] += 1
        stats["idle_terms"] += idle
        stats["layers"] = layers
        return total

    train_sft.sum_sparse_weight_norms = fresh_sum

    def restore() -> None:
        handle.remove()
        train_sft.sum_sparse_weight_norms = original
        sl._ACTIVE_FORWARD_TOKEN[0] = sl._STORE_ALWAYS

    return restore, stats
