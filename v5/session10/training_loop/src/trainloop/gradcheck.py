"""Verify one gradient by hand.

Pick one scalar weight. Ask backward() what dL/dw is. Then forget backward
exists: nudge that weight up by h, nudge it down by h, and measure how the loss
actually moved. Central difference:

    dL/dw  ~=  (L(w + h) - L(w - h)) / (2h)

In fp64 with dropout off, the two numbers agree to ~8-10 decimals. If they do
not, something in the step is not what you think it is.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch


@dataclass
class GradCheckResult:
    param_name: str
    index: tuple[int, ...]
    analytic: float   # from backward()
    numeric: float    # from central finite differences
    step_h: float

    @property
    def abs_diff(self) -> float:
        return abs(self.analytic - self.numeric)

    @property
    def rel_diff(self) -> float:
        denom = max(abs(self.analytic), abs(self.numeric), 1e-12)
        return self.abs_diff / denom

    @property
    def matching_decimals(self) -> int:
        d = 0
        while d < 15 and round(self.analytic, d) == round(self.numeric, d):
            d += 1
        return max(d - 1, 0)

    def render(self) -> str:
        return (
            f"parameter    : {self.param_name}{list(self.index)}\n"
            f"analytic  dL/dw (backward)          : {self.analytic:+.12f}\n"
            f"numeric   dL/dw (central diff, h={self.step_h:g}) : {self.numeric:+.12f}\n"
            f"absolute difference                 : {self.abs_diff:.3e}\n"
            f"relative difference                 : {self.rel_diff:.3e}\n"
            f"they agree to {self.matching_decimals} decimal places"
        )


@torch.no_grad()
def _loss_at(model, x, y, param, index, delta) -> float:
    original = param[index].item()
    param[index] = original + delta
    _, loss = model(x, targets=y)
    param[index] = original
    return float(loss.item())


def verify_one_gradient(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    param_name: str = "transformer.h.0.mlp.c_fc.weight",
    index: tuple[int, ...] = (0, 0),
    step_h: float = 1e-4,
) -> GradCheckResult:
    """Run in fp64 with the model in eval() (dropout off) for a clean check.

    Works on a deep copy so the caller's fp32 model is left untouched.
    """
    model = copy.deepcopy(model).double().eval()
    x = x.clone()
    y = y.clone()

    params = dict(model.named_parameters())
    if param_name not in params:
        raise KeyError(f"{param_name} not found. options include:\n  " +
                       "\n  ".join(list(params)[:12]))
    param = params[param_name]

    # analytic gradient
    model.zero_grad(set_to_none=True)
    _, loss = model(x, targets=y)
    loss.backward()
    analytic = float(param.grad[index].item())

    # numeric gradient: central difference
    l_plus = _loss_at(model, x, y, param, index, +step_h)
    l_minus = _loss_at(model, x, y, param, index, -step_h)
    numeric = (l_plus - l_minus) / (2 * step_h)

    return GradCheckResult(param_name, index, analytic, numeric, step_h)
