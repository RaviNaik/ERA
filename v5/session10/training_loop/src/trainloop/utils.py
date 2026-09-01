"""Small shared helpers: seeding, device pick, and the smoke-mode switch."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def is_smoke() -> bool:
    """True when S10_SMOKE=1. Smoke mode shrinks everything so the whole
    notebook runs top-to-bottom on a CPU or a tiny GPU in a couple of minutes."""
    return os.environ.get("S10_SMOKE", "") == "1"


def seed_everything(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "CPU"


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
