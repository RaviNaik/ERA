"""Chunked cross-entropy so the (batch*seq, vocab) logits tensor is never
materialized in full.

For this assignment's 20M-parameter model, `n_embd=256` but `vocab_size ==
50257`; a single un-chunked logits tensor for a modest batch already exceeds
the memory used by the entire 9-layer transformer body. Left un-chunked,
that shared, per-variant-identical cost would dominate peak memory for
*every* variant and hide the actual memory savings the reversible
architectures are supposed to demonstrate. Chunking the final projection +
loss over the token dimension keeps the large (chunk, vocab) tensor
transient (only one chunk alive at a time) instead of caching the full
(batch*seq, vocab) tensor for backward.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class _ChunkedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, targets, weight, chunk_size):
        n = hidden.shape[0]
        total_loss = hidden.new_zeros((), dtype=torch.float32)
        correct = hidden.new_zeros((), dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                logits_chunk = hidden[start:end].float() @ weight.float().t()
                target_chunk = targets[start:end]
                total_loss += F.cross_entropy(
                    logits_chunk, target_chunk, reduction="sum"
                )
                correct += (logits_chunk.argmax(dim=-1) == target_chunk).sum()
        ctx.save_for_backward(hidden, targets, weight)
        ctx.chunk_size = chunk_size
        ctx.n = n
        accuracy = correct / n
        ctx.mark_non_differentiable(accuracy)
        return total_loss / n, accuracy

    @staticmethod
    def backward(ctx, grad_loss, _grad_accuracy):
        hidden, targets, weight = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        n = ctx.n
        grad_hidden = torch.zeros_like(hidden)
        grad_weight = torch.zeros(
            weight.shape, dtype=torch.float32, device=weight.device
        )
        scale = grad_loss / n
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            with torch.enable_grad():
                h_chunk = hidden[start:end].detach().float().requires_grad_(True)
                w_req = weight.detach().float().requires_grad_(True)
                logits_chunk = h_chunk @ w_req.t()
                loss_chunk = F.cross_entropy(
                    logits_chunk, targets[start:end], reduction="sum"
                )
                grad_h, grad_w = torch.autograd.grad(loss_chunk, (h_chunk, w_req))
            grad_hidden[start:end] = (grad_h * scale).to(hidden.dtype)
            grad_weight += grad_w * scale
        return grad_hidden, None, grad_weight.to(weight.dtype), None


def chunked_cross_entropy(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (mean loss, top-1 next-token accuracy) without full logits."""
    hidden_flat = hidden.reshape(-1, hidden.size(-1))
    targets_flat = targets.reshape(-1)
    return _ChunkedCrossEntropy.apply(hidden_flat, targets_flat, weight, chunk_size)
