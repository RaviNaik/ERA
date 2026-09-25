"""Memory-efficient reversible transformer stacks (Euler / midpoint variants).

Follows "Reversing Large Language Models for Efficient Training and
Fine-Tuning" (arXiv:2512.02056v2):
  * euler    -> staggered / symplectic-Euler two-stream update (paper eq. 8-9):
               q^l = q^{l-1} + Attn_l(LN1(p^{l-1}))
               p^l = p^{l-1} + MLP_l(LN2(q^l))
  * midpoint -> explicit-midpoint depth recurrence (paper eq. 4-5):
               p^{l+1} = p^{l-1} + 2h * f_l(p^l),
               f_l(p) = Attn_l(LN1(p)) + MLP_l(LN2(p + Attn_l(LN1(p))))

Both variants are wired through a single stack-wide autograd.Function
(`_ReversibleSequenceFunction`) instead of one Function per layer. That is
the detail that actually delivers the promised memory savings: only the
final pair of hidden states for the *whole stack* is kept alive by autograd;
every intermediate per-layer activation is recomputed from the invertible
update and released immediately while walking backward through the stack.
"""

from __future__ import annotations

import torch
from torch import nn


class _ReversibleSequenceFunction(torch.autograd.Function):
    """Runs a chain of invertible cells without caching per-layer activations."""

    @staticmethod
    def forward(ctx, a, b, cells, param_counts, *flat_params):
        with torch.no_grad():
            for cell in cells:
                a, b = cell.step(a, b)
        ctx.cells = cells
        ctx.param_counts = param_counts
        ctx.save_for_backward(a, b)
        return a, b

    @staticmethod
    def backward(ctx, grad_a, grad_b):
        a, b = ctx.saved_tensors
        cells = ctx.cells
        param_counts = ctx.param_counts
        grad_chunks: list[tuple[torch.Tensor, ...]] = []
        for cell, n_params in zip(reversed(cells), reversed(param_counts)):
            with torch.no_grad():
                a_prev, b_prev = cell.inverse(a, b)
            with torch.enable_grad():
                a_req = a_prev.detach().requires_grad_(True)
                b_req = b_prev.detach().requires_grad_(True)
                params = tuple(cell.trainable_params())
                assert len(params) == n_params
                a_out, b_out = cell.step(a_req, b_req)
                grads = torch.autograd.grad(
                    (a_out, b_out),
                    (a_req, b_req) + params,
                    (grad_a, grad_b),
                    allow_unused=True,
                )
            grad_a_prev, grad_b_prev, *cell_param_grads = grads
            cell_param_grads = [
                torch.zeros_like(param) if grad is None else grad
                for grad, param in zip(cell_param_grads, params)
            ]
            grad_chunks.append(tuple(cell_param_grads))
            a, b = a_prev, b_prev
            grad_a, grad_b = grad_a_prev, grad_b_prev

        flat_param_grads: list[torch.Tensor] = []
        for chunk in reversed(grad_chunks):
            flat_param_grads.extend(chunk)
        return (grad_a, grad_b, None, None, *flat_param_grads)


def _flatten_params(
    cells: list[nn.Module],
) -> tuple[list[int], tuple[torch.Tensor, ...]]:
    counts = []
    flat: list[torch.Tensor] = []
    for cell in cells:
        params = list(cell.trainable_params())
        counts.append(len(params))
        flat.extend(params)
    return counts, tuple(flat)


class EulerCell(nn.Module):
    """Staggered/symplectic-Euler two-stream cell: q=x1+Attn(x2), p=x2+MLP(q)."""

    def __init__(self, cfg):
        super().__init__()
        from .model import CausalSelfAttention, MLP

        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_g = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(self.ln_f(x))

    def g(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.ln_g(x))

    def step(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y1 = x1 + self.f(x2)
        y2 = x2 + self.g(y1)
        return y1, y2

    def inverse(
        self, y1: torch.Tensor, y2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x2 = y2 - self.g(y1)
        x1 = y1 - self.f(x2)
        return x1, x2

    def trainable_params(self) -> tuple[torch.Tensor, ...]:
        return tuple(param for param in self.parameters() if param.requires_grad)


class MidpointCell(nn.Module):
    """One layer of the explicit-midpoint depth recurrence (paper eq. 4-5)."""

    def __init__(self, cfg, step_size: float):
        super().__init__()
        from .model import CausalSelfAttention, MLP

        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)
        self.two_h = 2.0 * step_size

    def f(self, p: torch.Tensor) -> torch.Tensor:
        a = self.attn(self.ln1(p))
        m = self.mlp(self.ln2(p + a))
        return a + m

    def step(
        self, prev: torch.Tensor, current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_state = prev + self.two_h * self.f(current)
        return current, next_state

    def inverse(
        self, current: torch.Tensor, next_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prev = next_state - self.two_h * self.f(current)
        return prev, current

    def trainable_params(self) -> tuple[torch.Tensor, ...]:
        return tuple(param for param in self.parameters() if param.requires_grad)


class ReversibleStack(nn.Module):
    """Full-depth reversible backbone; replaces the plain per-block loop.

    A single `_ReversibleSequenceFunction` call spans every layer, so only
    the last layer's pair of hidden states is retained by autograd --
    activation memory stays O(1) in depth instead of O(n_layer), which is
    the entire point of the reversible architectures being compared here.
    """

    def __init__(self, cfg, kind: str):
        super().__init__()
        if kind not in {"euler", "midpoint"}:
            raise ValueError(f"unknown reversible kind: {kind}")
        self.kind = kind
        self.scale = 2.0**-0.5
        if kind == "euler":
            self.cells = nn.ModuleList([EulerCell(cfg) for _ in range(cfg.n_layer)])
        else:
            # 1/n_layer step size keeps the explicit-midpoint recurrence
            # inside its (marginal) stability region as depth grows, per the
            # paper's stability analysis (Sec. 3): |b + lambda*h| <= 2.
            step_size = 1.0 / cfg.n_layer
            self.cells = nn.ModuleList(
                [MidpointCell(cfg, step_size) for _ in range(cfg.n_layer)]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cells = list(self.cells)
        param_counts, flat_params = _flatten_params(cells)
        if self.kind == "euler":
            a0 = x * self.scale
            b0 = x * self.scale
        else:
            a0 = x
            b0 = x
        a, b = _ReversibleSequenceFunction.apply(
            a0, b0, cells, param_counts, *flat_params
        )
        if self.kind == "euler":
            return (a + b) * self.scale
        return b
