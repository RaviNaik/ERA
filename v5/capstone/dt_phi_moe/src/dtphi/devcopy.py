"""Move tensors between two GPUs through the CPU instead of a direct GPU-to-GPU copy.

On the lab server a direct cuda:0 -> cuda:1 copy sometimes delivers garbage (position_ids of 4445483695778924510,
random out-of-range indices in the expert loop). Everything that crosses the GPU boundary goes through accelerate's
dispatch hooks, so their send_to_device is wrapped: a CUDA tensor that has to change GPU is copied to the CPU first and
from there to the target. The boundary is crossed once per forward (hidden states), so the cost is small. Autograd
follows the two copies, so the backward pass takes the same route.
"""
from __future__ import annotations

import torch


def _foreign(x, device) -> bool:
    if not (isinstance(x, torch.Tensor) and x.is_cuda):
        return False
    target = torch.device(device) if not isinstance(device, torch.device) else device
    if target.type != "cuda":
        return False
    return x.device != (target if target.index is not None else torch.device("cuda", torch.cuda.current_device()))


def via_cpu(obj, device):
    """Return obj with every CUDA tensor that lives on another GPU than `device` replaced by a copy via the CPU."""
    if _foreign(obj, device):
        return obj.to("cpu").to(device)
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):   # namedtuple
        return type(obj)(*(via_cpu(v, device) for v in obj))
    if isinstance(obj, (list, tuple)):
        return type(obj)(via_cpu(v, device) for v in obj)
    if isinstance(obj, dict):
        return type(obj)((k, via_cpu(v, device)) for k, v in obj.items())
    return obj


def install():
    """Wrap accelerate.hooks.send_to_device once. Returns a function that restores the original."""
    import accelerate.hooks as hooks
    original = hooks.send_to_device
    if getattr(original, "_dtphi_via_cpu", False):
        return lambda: None

    def send_to_device(tensor, device, non_blocking=False, skip_keys=None):
        skip = set(skip_keys) if isinstance(skip_keys, (list, tuple)) else ({skip_keys} if skip_keys else set())
        if isinstance(tensor, dict) and skip:
            tensor = {k: (v if k in skip else via_cpu(v, device)) for k, v in tensor.items()}
        else:
            tensor = via_cpu(tensor, device)
        return original(tensor, device, non_blocking=non_blocking, skip_keys=skip_keys)

    send_to_device._dtphi_via_cpu = True
    hooks.send_to_device = send_to_device

    def restore():
        hooks.send_to_device = original
    return restore
