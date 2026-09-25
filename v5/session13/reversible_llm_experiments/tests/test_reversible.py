from __future__ import annotations

import pytest
import torch
from torch import nn

from revllm.model import GPTConfig
from revllm.reversible import EulerCell, MidpointCell, ReversibleStack


def _small_cfg(n_layer: int = 3) -> GPTConfig:
    return GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=n_layer,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        bias=True,
    )


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


def _naive_euler_forward(cells: list[EulerCell], x: torch.Tensor, scale: float):
    x1 = x * scale
    x2 = x * scale
    for cell in cells:
        x1, x2 = cell.step(x1, x2)
    return (x1 + x2) * scale


def _naive_midpoint_forward(cells: list[MidpointCell], x: torch.Tensor):
    prev, current = x, x
    for cell in cells:
        prev, current = cell.step(prev, current)
    return current


@pytest.mark.parametrize("kind", ["euler", "midpoint"])
def test_reversible_stack_matches_naive_autograd_reference(kind: str):
    torch.manual_seed(2)
    cfg = _small_cfg(n_layer=3)
    stack = _to_double(ReversibleStack(cfg, kind=kind))
    naive = _to_double(ReversibleStack(cfg, kind=kind))
    naive.load_state_dict(stack.state_dict())

    x = torch.randn(2, 5, cfg.n_embd, dtype=torch.float64, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    out = stack(x)
    out.square().mean().backward()

    if kind == "euler":
        out_ref = _naive_euler_forward(list(naive.cells), x_ref, naive.scale)
    else:
        out_ref = _naive_midpoint_forward(list(naive.cells), x_ref)
    out_ref.square().mean().backward()

    assert torch.allclose(out, out_ref, atol=1e-9, rtol=1e-9)
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-8, rtol=1e-8)
    for param, param_ref in zip(stack.parameters(), naive.parameters()):
        assert param.grad is not None and param_ref.grad is not None
        assert torch.allclose(param.grad, param_ref.grad, atol=1e-8, rtol=1e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="memory check needs CUDA")
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
        x = torch.randn(4, 32, cfg.n_embd, device=device, requires_grad=True)
        out = stack(x)
        out.square().mean().backward()
        return torch.cuda.max_memory_allocated(device) / 1024**2

    mem_shallow = peak_mem_for_depth(3)
    mem_deep = peak_mem_for_depth(24)
    # A per-layer-caching implementation would scale peak memory roughly
    # linearly with depth (8x here); the reversible stack should stay close
    # to flat since only one pair of hidden states is ever kept alive.
    assert mem_deep < mem_shallow * 3.0
