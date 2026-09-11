"""Small shared helpers: seeding, device, plotting style, Aim repo path."""
from __future__ import annotations

import os
import random

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AIM_REPO = REPO_ROOT
ASSETS = os.path.join(REPO_ROOT, "assets")


def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def savefig(fig, name: str, dpi: int = 130):
    os.makedirs(ASSETS, exist_ok=True)
    path = os.path.join(ASSETS, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  saved {os.path.relpath(path, REPO_ROOT)}")
    return path


def plot_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "figure.dpi": 110,
    })
