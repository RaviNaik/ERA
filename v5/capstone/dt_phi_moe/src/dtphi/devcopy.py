"""Move tensors between two GPUs through the CPU instead of a direct GPU-to-GPU copy.

On the lab server a direct cuda:0 -> cuda:1 copy delivers zeros or garbage (position_ids of 4445483695778924510, gates
of zero, random labels; see tools/p2p_check.py). install_global() patches Tensor.to, .cuda and .copy_ so that a copy
between two GPUs goes through the CPU. Containers (accelerate's model outputs) are never rebuilt. Autograd follows the
two copies, so the backward pass takes the same route.
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


def via_cpu(t, device):
    """A tensor on another GPU than `device` comes back as a copy via the CPU; anything else is returned as it is."""
    return t.to("cpu").to(device) if _foreign(t, device) else t


# --- every Python-level GPU-to-GPU copy (ours, the official code's, transformers', accelerate's) -------------------

_ORIG = {}


def _target_device(args, kwargs):
    """The device a Tensor.to(...) call asks for, or None if it only changes dtype."""
    cands = [kwargs.get("device")] + list(args[:1])
    for a in cands:
        if isinstance(a, torch.Tensor):
            return a.device
        if isinstance(a, (torch.device, str)) or (isinstance(a, int) and not isinstance(a, bool)):
            return torch.device(a)
    return None


def _cross_gpu(t, target) -> bool:
    if target is None or not (isinstance(t, torch.Tensor) and t.is_cuda) or target.type != "cuda":
        return False
    idx = target.index if target.index is not None else torch.cuda.current_device()
    return t.device.index != idx


def install_global():
    """Patch Tensor.to, Tensor.cuda and Tensor.copy_ so that a copy between two GPUs goes through the CPU.

    Needed because direct GPU-to-GPU copies return zeros or garbage on the lab server (tools/p2p_check.py): a gate that
    was moved to the other GPU arrived as zeros (every weight deleted), and labels arrived as random numbers.
    Calling it twice is harmless. Returns a function that restores the originals.
    """
    if _ORIG:
        return lambda: None
    to, cuda, copy_ = torch.Tensor.to, torch.Tensor.cuda, torch.Tensor.copy_
    _ORIG.update(to=to, cuda=cuda, copy_=copy_)

    def patched_to(self, *args, **kwargs):
        if _cross_gpu(self, _target_device(args, kwargs)):
            return to(to(self, "cpu"), *args, **kwargs)
        return to(self, *args, **kwargs)

    def patched_cuda(self, device=None, *args, **kwargs):
        target = torch.device("cuda", device) if isinstance(device, int) else (
            torch.device(device) if device is not None else None)
        if _cross_gpu(self, target):
            return cuda(to(self, "cpu"), device, *args, **kwargs)
        return cuda(self, device, *args, **kwargs)

    def patched_copy_(self, src, *args, **kwargs):
        if isinstance(src, torch.Tensor) and src.is_cuda and self.is_cuda and src.device != self.device:
            src = to(src, "cpu")
        return copy_(self, src, *args, **kwargs)

    torch.Tensor.to, torch.Tensor.cuda, torch.Tensor.copy_ = patched_to, patched_cuda, patched_copy_

    def restore():
        torch.Tensor.to, torch.Tensor.cuda, torch.Tensor.copy_ = to, cuda, copy_
        _ORIG.clear()
    return restore
