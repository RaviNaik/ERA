"""Loss-harness utilities: the ordinary (materialize-everything) cross
entropy, a hand-written chunked version, perplexity, and a peak-memory
measurement helper.

The chunked implementation is Section-10-style "chunk it": run the head
matmul + cross entropy over a slice of `chunk_size` positions at a time,
discard that slice's logits, move to the next slice. Mathematically
identical loss (sum of per-token NLL, divided by the true count of
contributing tokens) to the naive path; the only thing that changes is
what has to be resident in memory at once.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ordinary_cross_entropy(
    hidden: torch.Tensor,      # [B, T, D]
    head_weight: torch.Tensor,  # [V, D]
    targets: torch.Tensor,     # [B, T] int64, may contain ignore_index
    ignore_index: int = -100,
) -> torch.Tensor:
    """Materialize the full [B, T, V] logits tensor, then cross-entropy."""
    B, T, D = hidden.shape
    logits = hidden.reshape(-1, D) @ head_weight.t()          # [B*T, V]
    loss = F.cross_entropy(logits, targets.reshape(-1), ignore_index=ignore_index)
    return loss


def chunked_cross_entropy(
    hidden: torch.Tensor,       # [B, T, D]
    head_weight: torch.Tensor,  # [V, D]
    targets: torch.Tensor,      # [B, T] int64, may contain ignore_index
    chunk_size: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Same objective, computed chunk_size tokens at a time so at most
    [chunk_size, V] logits ever exist simultaneously. The final scalar
    loss agrees with `ordinary_cross_entropy` to float precision.
    """
    B, T, D = hidden.shape
    flat_hidden = hidden.reshape(-1, D)     # [N, D]
    flat_targets = targets.reshape(-1)      # [N]
    N = flat_hidden.shape[0]

    total_loss = flat_hidden.new_zeros(())
    total_count = 0
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        h_chunk = flat_hidden[start:end]                  # [c, D]
        t_chunk = flat_targets[start:end]                 # [c]
        logits_chunk = h_chunk @ head_weight.t()           # [c, V] -- the only part ever materialized
        chunk_loss = F.cross_entropy(logits_chunk, t_chunk, ignore_index=ignore_index, reduction="sum")
        total_loss = total_loss + chunk_loss
        total_count += (t_chunk != ignore_index).sum().item()

    return total_loss / max(total_count, 1)


def chunked_cross_entropy_backward(
    hidden: torch.Tensor,       # [B, T, D], requires_grad
    head_weight: torch.Tensor,  # [V, D], requires_grad
    targets: torch.Tensor,      # [B, T] int64, may contain ignore_index
    chunk_size: int,
    ignore_index: int = -100,
) -> float:
    """Like `chunked_cross_entropy`, but also performs backward -- one
    chunk at a time.

    This is the detail that actually matters for peak memory. Computing
    `chunked_cross_entropy(...)` and then calling `.backward()` once on the
    result keeps *every* chunk's logits (and their saved-for-backward
    activations) alive simultaneously, because autograd needs the whole
    graph to walk backward through it -- that defeats chunking entirely.
    Here, each chunk's loss (pre-scaled by 1/total_count so the chunks'
    gradients sum to the true mean-loss gradient) is backpropagated
    immediately, which lets autograd free that chunk's graph before the
    next chunk is even computed. Returns the total loss as a plain float
    (there is no single tensor left to return -- the backward already
    happened chunk-by-chunk).
    """
    B, T, D = hidden.shape
    flat_hidden = hidden.reshape(-1, D)
    flat_targets = targets.reshape(-1)
    N = flat_hidden.shape[0]

    total_count = max(int((flat_targets != ignore_index).sum().item()), 1)
    total_loss_value = 0.0

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        h_chunk = flat_hidden[start:end]
        t_chunk = flat_targets[start:end]
        logits_chunk = h_chunk @ head_weight.t()  # [c, V] -- the only part ever materialized
        chunk_loss = F.cross_entropy(
            logits_chunk, t_chunk, ignore_index=ignore_index, reduction="sum"
        ) / total_count
        chunk_loss.backward()  # accumulates into hidden.grad / head_weight.grad, then frees this chunk's graph
        total_loss_value += chunk_loss.item()

    return total_loss_value


def perplexity(mean_loss: torch.Tensor | float) -> float:
    if isinstance(mean_loss, torch.Tensor):
        mean_loss = mean_loss.item()
    return float(torch.exp(torch.tensor(mean_loss)))


def measure_peak_memory_bytes(fn, device: torch.device) -> int:
    """Run fn() (forward + backward already performed inside fn) after
    resetting CUDA's peak-memory counter, return the peak bytes allocated
    during the call. CPU fallback reports 0 (no reliable equivalent).
    """
    if device.type != "cuda":
        fn()
        return 0
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    fn()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device)
