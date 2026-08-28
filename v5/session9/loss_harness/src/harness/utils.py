"""Small shared utilities: seeding, device selection, and a shape-printer
that prints a tensor's shape plus a one-line meaning for every dimension
(the very first checkbox of the assignment).
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_shape(name: str, tensor: torch.Tensor, dim_meanings: list[str]) -> None:
    """Print `name: shape` followed by one line per dimension explaining
    what that dimension indexes. `dim_meanings` must have one entry per
    dimension of `tensor`.
    """
    shape = tuple(tensor.shape)
    assert len(dim_meanings) == len(shape), (
        f"{name}: {len(shape)} dims but {len(dim_meanings)} meanings given"
    )
    print(f"{name}: shape = {shape}  dtype = {tensor.dtype}")
    for i, (size, meaning) in enumerate(zip(shape, dim_meanings)):
        print(f"    dim {i} (size {size:>6}): {meaning}")


def collect_params(*modules) -> list:
    """Union of parameters across modules, deduplicated by identity.

    Needed whenever a tied output head is combined with its trunk for an
    optimizer: `head.weight` and `trunk.tok_emb.weight` are the *same*
    nn.Parameter object, so naively concatenating `model.parameters()` and
    `head.parameters()` would list it twice -- and Adam would apply two
    updates to it per step instead of one.
    """
    seen: dict[int, "torch.nn.Parameter"] = {}
    for m in modules:
        for p in m.parameters():
            seen[id(p)] = p
    return list(seen.values())


def human_bytes(n: int) -> str:
    n = float(n)
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TiB"
