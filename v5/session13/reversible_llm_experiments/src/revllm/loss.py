"""Fused, chunked lm_head + cross-entropy.

For this assignment's 20M-parameter model, `n_embd=256` but `vocab_size ==
50257`. Two consequences:

* Memory: an un-chunked (batch*seq, vocab) fp32 logits tensor is ~6.4 GB at
  batch 64 x 512 tokens. Counting its softmax/grad copies, the vocab
  projection alone uses more memory than the whole 9-layer backbone, and the
  cost is identical for every variant. Left in place, it would hide the
  difference that the reversible architectures actually make.
* Compute: the vocab projection is ~2 * 256 * 50257 = 25.7 MFLOP per token
  forward, more than half of the model's forward FLOPs.

So the loss is computed over token chunks, with only one (chunk, vocab)
logits tensor alive at a time. The gradient w.r.t. the hidden states and the
tied embedding/lm_head weight is computed *in the same pass* as the loss
(the "Liger kernel" trick): softmax(logits) - onehot(target) is the logits
gradient, so backward just rescales two stored tensors. The vocab projection
is never recomputed, and nothing of size O(tokens x vocab) is stored.
"""

from __future__ import annotations

import torch


class _FusedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, targets, weight, chunk_size, compute_dtype):
        n = hidden.shape[0]
        # Function.forward always runs with grad mode off, so read the outer
        # mode from requires_grad (False under torch.no_grad() evaluation).
        need_grad = hidden.requires_grad or weight.requires_grad
        # Softmax / loss / weight-grad accumulation in at least fp32.
        acc_dtype = torch.promote_types(compute_dtype, torch.float32)
        w = weight.to(compute_dtype)
        loss_sum = torch.zeros((), dtype=acc_dtype, device=hidden.device)
        correct = torch.zeros((), dtype=acc_dtype, device=hidden.device)
        grad_hidden = torch.empty_like(hidden) if need_grad else None
        grad_weight = (
            torch.zeros(weight.shape, dtype=acc_dtype, device=weight.device)
            if need_grad
            else None
        )
        inv_n = 1.0 / n
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                h = hidden[start:end].to(compute_dtype)
                t = targets[start:end]
                logits = h @ w.t()  # (c, V) in compute dtype
                correct += (logits.argmax(dim=-1) == t).sum()
                # One fused kernel; softmax maths in >= fp32 even for bf16 logits.
                logp = torch.log_softmax(logits, dim=-1, dtype=acc_dtype)
                del logits
                loss_sum -= logp.gather(1, t[:, None]).sum()
                if need_grad:
                    # d(mean CE)/d logits = (softmax - onehot) / n, in place.
                    probs = logp.exp_()
                    probs[torch.arange(end - start, device=t.device), t] -= 1.0
                    g = probs.mul_(inv_n).to(compute_dtype)
                    del probs, logp
                    grad_hidden[start:end] = (g @ w).to(hidden.dtype)
                    grad_weight += (g.t() @ h).to(acc_dtype)
                    del g
                else:
                    del logp
        if need_grad:
            ctx.save_for_backward(grad_hidden, grad_weight)
        ctx.weight_dtype = weight.dtype
        accuracy = correct * inv_n
        ctx.mark_non_differentiable(accuracy)
        return loss_sum * inv_n, accuracy

    @staticmethod
    def backward(ctx, grad_loss, _grad_accuracy):
        grad_hidden, grad_weight = ctx.saved_tensors
        return (
            grad_hidden * grad_loss.to(grad_hidden.dtype),
            None,
            (grad_weight * grad_loss).to(ctx.weight_dtype),
            None,
            None,
        )


def chunked_cross_entropy(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (mean loss, top-1 next-token accuracy) without full logits.

    The matmuls run in the active autocast dtype (bf16 on CUDA during
    training), and log-softmax and the loss are accumulated in fp32.
    """
    hidden_flat = hidden.reshape(-1, hidden.size(-1))
    targets_flat = targets.reshape(-1)
    device_type = hidden.device.type
    if torch.is_autocast_enabled(device_type):
        compute_dtype = torch.get_autocast_dtype(device_type)
    else:
        compute_dtype = hidden.dtype
    return _FusedLinearCrossEntropy.apply(
        hidden_flat, targets_flat, weight, chunk_size, compute_dtype
    )
