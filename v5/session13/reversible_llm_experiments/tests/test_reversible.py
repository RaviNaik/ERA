from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from revllm.loss import chunked_cross_entropy
from revllm.model import GPT, GPTConfig, flops_per_token
from revllm.reversible import (
    EulerCell,
    MidpointCell,
    ReversibleStack,
    reversibility_diagnostics,
)

CUDA = torch.cuda.is_available()


def _small_cfg(n_layer: int = 3, **kw) -> GPTConfig:
    base = dict(vocab_size=32, block_size=8, n_head=2, n_embd=8, dropout=0.0, bias=True)
    return GPTConfig(n_layer=n_layer, **(base | kw))


def _to_double(module: nn.Module) -> nn.Module:
    return module.to(torch.float64)


def test_euler_cell_inverse_recovers_inputs():
    torch.manual_seed(0)
    cell = _to_double(EulerCell(_small_cfg()))
    x1 = torch.randn(2, 5, 8, dtype=torch.float64)
    x2 = torch.randn(2, 5, 8, dtype=torch.float64)
    with torch.no_grad():
        y1, y2 = cell.step(x1, x2)
        x1_hat, x2_hat = cell.inverse(y1, y2)
    assert torch.allclose(x1, x1_hat, atol=1e-10, rtol=1e-10)
    assert torch.allclose(x2, x2_hat, atol=1e-10, rtol=1e-10)


def test_midpoint_cell_inverse_recovers_inputs():
    torch.manual_seed(1)
    cell = _to_double(MidpointCell(_small_cfg(), step_size=0.3))
    prev = torch.randn(2, 5, 8, dtype=torch.float64)
    current = torch.randn(2, 5, 8, dtype=torch.float64)
    with torch.no_grad():
        current_out, next_state = cell.step(prev, current)
        prev_hat, current_hat = cell.inverse(current_out, next_state)
    assert torch.allclose(prev, prev_hat, atol=1e-10, rtol=1e-10)
    assert torch.allclose(current, current_hat, atol=1e-10, rtol=1e-10)


def test_midpoint_first_layer_with_half_step_is_a_standard_block():
    """h = 0.5 and p^{-1} = p^0 = x make layer 1 exactly x + Block_residual(x)."""
    torch.manual_seed(5)
    cfg = _small_cfg(n_layer=1, midpoint_h=0.5, reversible_exact=False)
    stack = _to_double(ReversibleStack(cfg, kind="midpoint"))
    cell = stack.cells[0]
    x = torch.randn(2, 5, 8, dtype=torch.float64)
    with torch.no_grad():
        expected = x + cell.attn(cell.ln1(x))
        expected = expected + cell.mlp(cell.ln2(expected))
        assert torch.allclose(stack(x), expected, atol=1e-12)


@pytest.mark.parametrize("kind", ["euler", "midpoint"])
@pytest.mark.parametrize("midpoint_h", [0.5, 1.0 / 3])
def test_reversible_stack_matches_naive_autograd_reference(kind: str, midpoint_h: float):
    torch.manual_seed(2)
    cfg = _small_cfg(n_layer=3, midpoint_h=midpoint_h)
    stack = _to_double(ReversibleStack(cfg, kind=kind))

    x = torch.randn(2, 5, cfg.n_embd, dtype=torch.float64, requires_grad=True)
    out = stack(x)
    out.square().mean().backward()
    grads = [p.grad.clone() for p in stack.parameters()]
    x_grad = x.grad.clone()

    stack.zero_grad()
    x_ref = x.detach().clone().requires_grad_(True)
    out_ref = stack.naive_forward(x_ref)
    out_ref.square().mean().backward()

    assert torch.allclose(out, out_ref, atol=1e-12, rtol=1e-12)
    assert torch.allclose(x_grad, x_ref.grad, atol=1e-10, rtol=1e-10)
    for grad, param in zip(grads, stack.parameters()):
        assert torch.allclose(grad, param.grad, atol=1e-10, rtol=1e-10)


def test_reversible_cells_reject_dropout():
    with pytest.raises(ValueError, match="dropout"):
        ReversibleStack(_small_cfg(dropout=0.1), kind="euler")


@pytest.mark.parametrize("chunk", [3, 7, 1000])
def test_fused_cross_entropy_matches_reference(chunk: int):
    torch.manual_seed(3)
    hidden = torch.randn(4, 5, 6, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(11, 6, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, 11, (4, 5))

    loss, acc = chunked_cross_entropy(hidden, targets, weight, chunk)
    (loss * 2.5).backward()  # non-unit upstream gradient
    gh, gw = hidden.grad.clone(), weight.grad.clone()

    hidden.grad = weight.grad = None
    logits = hidden @ weight.t()
    ref = F.cross_entropy(logits.view(-1, 11), targets.view(-1))
    (ref * 2.5).backward()
    ref_acc = (logits.argmax(-1) == targets).double().mean()

    assert torch.allclose(loss, ref, atol=1e-12)
    assert torch.allclose(acc.double(), ref_acc, atol=1e-12)
    assert torch.allclose(gh, hidden.grad, atol=1e-12)
    assert torch.allclose(gw, weight.grad, atol=1e-12)


def test_fused_cross_entropy_no_grad_path():
    hidden = torch.randn(2, 3, 4)
    weight = torch.randn(9, 4)
    targets = torch.randint(0, 9, (2, 3))
    with torch.no_grad():
        loss, _ = chunked_cross_entropy(hidden, targets, weight, 2)
    ref = F.cross_entropy((hidden @ weight.t()).view(-1, 9), targets.view(-1))
    assert torch.allclose(loss, ref, atol=1e-5)


@pytest.mark.parametrize("variant", ["baseline", "euler", "midpoint"])
def test_full_model_trains_and_counts_match(variant: str):
    torch.manual_seed(4)
    cfg = _small_cfg(variant=variant, loss_chunk_size=7)
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    _, loss, _ = model(idx, idx)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
    baseline = GPT(_small_cfg(variant="baseline"))
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in baseline.parameters()
    )


def test_flops_accounting_reversible_adds_one_backbone_forward():
    base = flops_per_token(GPTConfig(variant="baseline"))
    rev = flops_per_token(GPTConfig(variant="midpoint"))
    assert base["model"] == rev["model"] == base["hardware"]
    extra = rev["hardware"] - base["model"]
    assert 0 < extra < base["forward"]  # backbone only, lm_head not recomputed


@pytest.mark.skipif(not CUDA, reason="autocast gradient check needs CUDA")
@pytest.mark.parametrize("kind", ["euler", "midpoint"])
def test_exact_reversal_under_bf16_autocast_is_bit_exact(kind: str):
    torch.manual_seed(6)
    cfg = GPTConfig(n_layer=6, n_head=4, n_embd=128, block_size=64)
    stack = ReversibleStack(cfg, kind=kind).cuda()
    # Scale weights up so the layers are "sharp" (like trained weights),
    # which is where inexact reversal drifts the most.
    with torch.no_grad():
        for p in stack.parameters():
            if p.dim() == 2:
                p.mul_(8.0)
    x = torch.randn(4, 64, 128, device="cuda")
    diag = reversibility_diagnostics(stack, x, torch.bfloat16)
    assert diag["max_rel_error"] == 0.0
    assert diag["grad_rel_error"] < 1e-6


@pytest.mark.skipif(not CUDA, reason="autocast gradient check needs CUDA")
def test_inexact_fp32_reversal_drifts_under_bf16():
    """Documents why exact mode exists: plain fp32 streams drift under bf16."""
    torch.manual_seed(6)
    cfg = GPTConfig(n_layer=6, n_head=4, n_embd=128, block_size=64, reversible_exact=False)
    stack = ReversibleStack(cfg, kind="midpoint").cuda()
    with torch.no_grad():
        for p in stack.parameters():
            if p.dim() == 2:
                p.mul_(8.0)
    x = torch.randn(4, 64, 128, device="cuda")
    diag = reversibility_diagnostics(stack, x, torch.bfloat16)
    assert diag["max_rel_error"] > 1e-6


@pytest.mark.skipif(not CUDA, reason="memory check needs CUDA")
@pytest.mark.parametrize("kind", ["euler", "midpoint"])
def test_reversible_stack_peak_memory_is_not_proportional_to_depth(kind: str):
    device = "cuda"

    def peak_mem_for_depth(n_layer: int) -> float:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        cfg = _small_cfg(n_layer=n_layer)
        cfg.n_embd = 64
        cfg.n_head = 4
        stack = ReversibleStack(cfg, kind=kind).to(device)
        base = torch.cuda.memory_allocated(device)
        x = torch.randn(4, 32, cfg.n_embd, device=device, requires_grad=True)
        out = stack(x)
        out.square().mean().backward()
        return (torch.cuda.max_memory_allocated(device) - base) / 1024**2

    mem_shallow = peak_mem_for_depth(3)
    mem_deep = peak_mem_for_depth(24)
    # Per-layer caching would grow ~8x here; the reversible stack stays near
    # flat because only one pair of hidden states is ever kept alive.
    assert mem_deep < mem_shallow * 3.0
