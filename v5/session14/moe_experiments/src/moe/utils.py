"""Small shared helpers: seeding, device, plotting style, paths."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ASSETS = os.path.join(REPO_ROOT, "assets")
RESULTS = os.path.join(REPO_ROOT, "results")
CKPT_DIR = os.path.join(REPO_ROOT, "checkpoints")


def is_smoke() -> bool:
    return os.environ.get("MOE_PRESET") == "smoke"


def work_root() -> str:
    """Where Aim repo / results / checkpoints / assets live.

    Normal runs use the project root.  Smoke runs set ``MOE_WORKDIR`` to a
    scratch dir so the real Aim repo and checkpoints are never touched."""
    return os.environ.get("MOE_WORKDIR", REPO_ROOT)


def aim_repo() -> str:
    return work_root()


def results_dir() -> str:
    return os.path.join(work_root(), "results")


def ckpt_dir() -> str:
    return os.path.join(work_root(), "checkpoints")


def assets_dir() -> str:
    return os.path.join(work_root(), "assets")


def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """``MOE_DEVICE`` (e.g. ``cuda:1`` on a shared multi-GPU box) wins; otherwise the default CUDA device."""
    if os.environ.get("MOE_DEVICE"):
        return os.environ["MOE_DEVICE"]
    return "cuda" if torch.cuda.is_available() else "cpu"


def savefig(fig, name: str, dpi: int = 130):
    d = assets_dir()
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  saved {os.path.relpath(path, work_root())}")
    return path


def plot_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "figure.dpi": 110,
        }
    )
