"""Memory-efficient reversible transformer stacks (Euler / midpoint variants).

Follows "Reversing Large Language Models for Efficient Training and
Fine-Tuning" (arXiv:2512.02056v2):

  * euler    -> Hamiltonian / symplectic-Euler two-stream update (paper eq. 8-9,
                a_l = b_l = 1):
                    q^l = q^{l-1} + Attn_l(LN1(p^{l-1}))
                    p^l = p^{l-1} + MLP_l(LN2(q^l))
                inverse:
                    p^{l-1} = p^l - MLP_l(LN2(q^l))
                    q^{l-1} = q^l - Attn_l(LN1(p^{l-1}))
  * midpoint -> explicit-midpoint (leapfrog-style) depth recurrence
                (paper eq. 4-5):
                    p^{l+1} = p^{l-1} + 2h * f_l(p^l)
                    f_l(p)  = Attn_l(LN1(p)) + MLP_l(LN2(p + Attn_l(LN1(p))))
                inverse:
                    p^{l-1} = p^{l+1} - 2h * f_l(p^l)

Implementation notes (these are what make the savings real and the gradients
exact):

1. One autograd.Function spans the *whole* stack. Only the final pair of
   hidden states is saved; every intermediate activation is reconstructed
   layer-by-layer during backward and freed right after use. Activation
   memory is therefore O(1) in depth instead of O(n_layer).

2. The backward pass evaluates each sub-layer exactly *once* (RevNet-style,
   Gomez et al. 2017): the same sub-layer evaluation both reconstructs the
   previous state and provides the vector-Jacobian product. A naive
   "inverse(), then re-run step() under autograd" does every sub-layer twice,
   i.e. two extra forward passes instead of one.

3. `custom_fwd` / `custom_bwd` make backward run under the *same* autocast
   state as forward. Without them the autograd engine runs backward outside
   autocast, so the reconstruction recomputes sub-layers in fp32 while the
   forward used bf16. That makes reconstruction inexact and backward slow.
   With matching precision, the sub-layer output recomputed in backward
   matches the forward one, and the only reconstruction error left is fp32
   rounding in the residual add/subtract (see `reversibility_diagnostics`).

4. Dropout is not supported inside reversible cells: the recomputation would
   draw a different dropout mask and break exact inversion.

5. Exact (bit-for-bit) reversal, on by default (`exact=True`). Even with
   matching autocast, fp32 streams are not exactly invertible: x' = (x + u) - u
   differs from x by a rounding error. Worse, that tiny error can flip the
   bf16 rounding of the next sub-layer's input, which turns 1e-8 drift into
   1e-3 drift per layer and compounds over depth (measured: ~10% input drift
   and ~2% gradient error after a few hundred steps). Following fixed-point
   reversible nets (MacKay et al., 2018), the two residual streams are kept in
   float64, and every sub-layer update is rounded onto a fixed 2^-30 grid
   before it is added. Sums of grid values below 2^23 are exact in float64, so
   subtraction recovers the previous state *exactly*, the recomputed
   sub-layers are bit-identical to the forward ones, and the reversible
   gradient equals ordinary autograd's. The rounding is a straight-through
   estimator (gradient = identity). The cost is 8 instead of 4 bytes per
   stream element, i.e. still O(1) in depth.
"""

from __future__ import annotations

import torch
from torch import nn


class _ReversibleSequenceFunction(torch.autograd.Function):
    """Runs a chain of invertible cells without caching per-layer activations."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, a, b, cells, param_counts, *flat_params):
        with torch.no_grad():
            for cell in cells:
                a, b = cell.step(a, b)
        ctx.cells = cells
        ctx.param_counts = param_counts
        ctx.save_for_backward(a, b)
        return a, b

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_a, grad_b):
        a, b = ctx.saved_tensors
        # Walk the stack in reverse: reconstruct each cell's inputs from its
        # outputs and push the gradient through it in the same pass.
        grad_chunks: list[tuple[torch.Tensor | None, ...]] = []
        for cell, n_params in zip(reversed(ctx.cells), reversed(ctx.param_counts)):
            a, b, grad_a, grad_b, param_grads = cell.backward_step(
                a, b, grad_a, grad_b
            )
            assert len(param_grads) == n_params
            grad_chunks.append(param_grads)
            # Release the reconstruction graph for this cell immediately.
            del param_grads

        flat_param_grads: list[torch.Tensor | None] = []
        for chunk in reversed(grad_chunks):
            flat_param_grads.extend(chunk)
        return (grad_a, grad_b, None, None, *flat_param_grads)


_GRID = 2.0**30  # updates are rounded to multiples of 2^-30 in exact mode


def _quantize(update: torch.Tensor, exact: bool) -> torch.Tensor:
    """Rounds a sub-layer update onto the fixed-point grid (straight-through).

    Returns float64 exactly equal to round(update * 2^30) / 2^30. The
    `u + (q - u).detach()` form is exact here: q - u is representable
    (Sterbenz), so u + (q - u) == q bit-for-bit, while the gradient w.r.t.
    `update` is the identity.
    """
    if not exact:
        return update
    u = update.double()
    q = (u * _GRID).round_().div_(_GRID)
    if not (torch.is_grad_enabled() and u.requires_grad):
        return q  # forward pass under no_grad: no straight-through needed
    return u + (q - u).detach()


def _vjp(
    output: torch.Tensor,
    inputs: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor | None, ...]:
    return torch.autograd.grad(output, inputs, grad_output, allow_unused=True)


def _check_no_dropout(cfg) -> None:
    if cfg.dropout > 0.0:
        raise ValueError(
            "reversible cells require dropout == 0.0: backward recomputes every "
            "sub-layer and would draw a different dropout mask"
        )


class EulerCell(nn.Module):
    """Hamiltonian / symplectic-Euler cell: q' = q + Attn(p); p' = p + MLP(q')."""

    def __init__(self, cfg):
        super().__init__()
        from .model import MLP, CausalSelfAttention

        _check_no_dropout(cfg)
        self.exact = getattr(cfg, "reversible_exact", True)
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_g = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        # Streams may be float64 (exact mode); sub-layers run in param dtype.
        return _quantize(self.attn(self.ln_f(x.to(self.ln_f.weight.dtype))), self.exact)

    def g(self, x: torch.Tensor) -> torch.Tensor:
        return _quantize(self.mlp(self.ln_g(x.to(self.ln_g.weight.dtype))), self.exact)

    def f_params(self) -> tuple[torch.Tensor, ...]:
        return (*self.ln_f.parameters(), *self.attn.parameters())

    def g_params(self) -> tuple[torch.Tensor, ...]:
        return (*self.ln_g.parameters(), *self.mlp.parameters())

    def trainable_params(self) -> tuple[torch.Tensor, ...]:
        return self.f_params() + self.g_params()

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

    def backward_step(self, y1, y2, grad_y1, grad_y2):
        """Reconstruct (x1, x2) from (y1, y2) and backprop through the cell.

        y2 = x2 + g(y1):  dL/dy1 += J_g^T dL/dy2 ;  dL/dx2 = dL/dy2 + ...
        y1 = x1 + f(x2):  dL/dx1  = dL/dy1_total ;  dL/dx2 += J_f^T dL/dy1_total
        """
        g_params = self.g_params()
        with torch.enable_grad():
            y1_leaf = y1.detach().requires_grad_(True)
            g_out = self.g(y1_leaf)
            grad_y1_from_g, *grads_g = _vjp(g_out, (y1_leaf, *g_params), grad_y2)
        x2 = y2 - g_out.detach()
        del g_out
        grad_y1_total = grad_y1 + grad_y1_from_g

        f_params = self.f_params()
        with torch.enable_grad():
            x2_leaf = x2.detach().requires_grad_(True)
            f_out = self.f(x2_leaf)
            grad_x2_from_f, *grads_f = _vjp(
                f_out, (x2_leaf, *f_params), grad_y1_total
            )
        x1 = y1 - f_out.detach()
        del f_out

        grad_x1 = grad_y1_total
        grad_x2 = grad_y2 + grad_x2_from_f
        return x1, x2, grad_x1, grad_x2, (*grads_f, *grads_g)


class MidpointCell(nn.Module):
    """One layer of the explicit-midpoint depth recurrence (paper eq. 4-5)."""

    def __init__(self, cfg, step_size: float):
        super().__init__()
        from .model import MLP, CausalSelfAttention

        _check_no_dropout(cfg)
        self.exact = getattr(cfg, "reversible_exact", True)
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)
        self.two_h = 2.0 * step_size

    def f(self, p: torch.Tensor) -> torch.Tensor:
        # Exactly the residual branch of a pre-LN transformer block:
        # Block(p) = p + f(p).
        p = p.to(self.ln1.weight.dtype)
        a = self.attn(self.ln1(p))
        m = self.mlp(self.ln2(p + a))
        return a + m

    def trainable_params(self) -> tuple[torch.Tensor, ...]:
        return tuple(self.parameters())

    def _update(self, current: torch.Tensor) -> torch.Tensor:
        # Written once so forward and inverse subtract bit-identical values.
        return _quantize(self.two_h * self.f(current), self.exact)

    def step(
        self, prev: torch.Tensor, current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_state = prev + self._update(current)
        return current, next_state

    def inverse(
        self, current: torch.Tensor, next_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prev = next_state - self._update(current)
        return prev, current

    def backward_step(self, current, next_state, grad_current_out, grad_next):
        """(prev, current) -> (current, prev + 2h f(current)), reversed."""
        params = self.trainable_params()
        with torch.enable_grad():
            cur_leaf = current.detach().requires_grad_(True)
            update = self._update(cur_leaf)
            grad_cur_from_f, *param_grads = _vjp(
                update, (cur_leaf, *params), grad_next
            )
        prev = next_state - update.detach()
        del update
        grad_prev = grad_next
        grad_current = grad_current_out + grad_cur_from_f
        return prev, current, grad_prev, grad_current, tuple(param_grads)


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


class ReversibleStack(nn.Module):
    """Full-depth reversible backbone that replaces the plain per-block loop.

    One `_ReversibleSequenceFunction` call spans every layer, so autograd keeps
    only the last layer's pair of hidden states. Activation memory is O(1) in
    depth instead of O(n_layer).

    State initialisation / readout (the paper leaves both unspecified):
      * euler:    q^0 = p^0 = x / sqrt(2);  output = (q^L + p^L) / sqrt(2)
      * midpoint: p^{-1} = p^0 = x;         output = p^L
        f is exactly the residual branch of a pre-LN block, so with h = 0.5
        (2h = 1) the first layer is a standard block x + f(x). The default
        h = 0.25 halves every update, which is more stable.
    """

    def __init__(self, cfg, kind: str):
        super().__init__()
        if kind not in {"euler", "midpoint"}:
            raise ValueError(f"unknown reversible kind: {kind}")
        self.kind = kind
        self.exact = getattr(cfg, "reversible_exact", True)
        self.scale = 2.0**-0.5
        if kind == "euler":
            self.cells = nn.ModuleList([EulerCell(cfg) for _ in range(cfg.n_layer)])
        else:
            step_size = getattr(cfg, "midpoint_h", 0.25)
            self.cells = nn.ModuleList(
                [MidpointCell(cfg, step_size) for _ in range(cfg.n_layer)]
            )

    def _init_state(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kind == "euler":
            x = x * self.scale
        x = _quantize(x, self.exact)
        return x, x

    def _readout(self, a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if self.kind == "euler":
            return ((a + b) * self.scale).to(dtype)
        return b.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cells = list(self.cells)
        param_counts, flat_params = _flatten_params(cells)
        a0, b0 = self._init_state(x)
        a, b = _ReversibleSequenceFunction.apply(
            a0, b0, cells, param_counts, *flat_params
        )
        return self._readout(a, b, x.dtype)

    def naive_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Same maths with ordinary autograd (stores every activation).

        Used as the reference for gradient checks and for measuring how much
        activation memory the reversible backward saves.
        """
        a, b = self._init_state(x)
        for cell in self.cells:
            a, b = cell.step(a, b)
        return self._readout(a, b, x.dtype)


@torch.no_grad()
def reconstruction_error(stack: ReversibleStack, x: torch.Tensor) -> dict:
    """Run the stack forward, invert it back, and report per-layer drift.

    Must be called under the same autocast context as training so the numbers
    reflect the precision actually used. In exact mode the error is 0.
    """
    a, b = stack._init_state(x)
    states = [(a.float().clone(), b.float().clone())]
    for cell in stack.cells:
        a, b = cell.step(a, b)
        states.append((a.float().clone(), b.float().clone()))
    per_layer = []
    for layer in range(len(stack.cells) - 1, -1, -1):
        a, b = stack.cells[layer].inverse(a, b)
        ref_a, ref_b = states[layer]
        err = max(
            ((a.float() - ref_a).norm() / ref_a.norm().clamp_min(1e-12)).item(),
            ((b.float() - ref_b).norm() / ref_b.norm().clamp_min(1e-12)).item(),
        )
        per_layer.append(err)
    per_layer.reverse()
    return {
        "max_rel_error": max(per_layer),
        "input_rel_error": per_layer[0],
        "per_layer_rel_error": per_layer,
    }


def gradient_agreement(
    stack: ReversibleStack, x: torch.Tensor, autocast_dtype: torch.dtype | None
) -> dict:
    """Compare reversible-backward gradients with plain-autograd gradients."""
    device_type = x.device.type

    def grads(fn) -> torch.Tensor:
        stack.zero_grad(set_to_none=True)
        xi = x.detach().clone().requires_grad_(True)
        with torch.autocast(
            device_type=device_type,
            dtype=autocast_dtype or torch.bfloat16,
            enabled=autocast_dtype is not None,
        ):
            out = fn(xi)
        out.float().square().mean().backward()
        flat = [p.grad.flatten().float() for p in stack.parameters()]
        flat.append(xi.grad.flatten().float())
        return torch.cat(flat)

    rev = grads(stack)
    ref = grads(stack.naive_forward)
    stack.zero_grad(set_to_none=True)
    cos = torch.nn.functional.cosine_similarity(rev, ref, dim=0).item()
    rel = ((rev - ref).norm() / ref.norm().clamp_min(1e-12)).item()
    return {"grad_cosine": cos, "grad_rel_error": rel}


def reversibility_diagnostics(
    stack: ReversibleStack, x: torch.Tensor, autocast_dtype: torch.dtype | None
) -> dict:
    """Reconstruction drift and gradient agreement for one input batch."""
    was_training = stack.training
    stack.eval()  # identical to train here: no dropout inside reversible cells
    with torch.autocast(
        device_type=x.device.type,
        dtype=autocast_dtype or torch.bfloat16,
        enabled=autocast_dtype is not None,
    ):
        recon = reconstruction_error(stack, x)
    out = recon | gradient_agreement(stack, x, autocast_dtype)
    stack.train(was_training)
    return out
