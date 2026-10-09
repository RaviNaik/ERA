"""Plot helpers reused across notebooks."""
from __future__ import annotations

import numpy as np

COLORS = {"dense": "#1f77b4", "dense_continue": "#1f77b4", "moe_scratch": "#d95f02",
          "moe_upcycled": "#2ca02c"}


def ema(x, alpha: float = 0.1):
    out, m = [], None
    for v in x:
        m = v if m is None else (1 - alpha) * m + alpha * v
        out.append(m)
    return np.array(out)


def color(label: str, default=None):
    for k, c in COLORS.items():
        if label.startswith(k):
            return c
    return default


def heatmap_load(ax, load, title="", vmax=None):
    """load: [L, E] fraction of routed slots; fair share = 1/E."""
    load = np.asarray(load)
    E = load.shape[1]
    im = ax.imshow(load, aspect="auto", cmap="viridis", vmin=0, vmax=vmax or max(2.0 / E, load.max()))
    ax.set_xlabel("expert"); ax.set_ylabel("MoE layer #"); ax.set_title(title)
    ax.set_xticks(range(E)); ax.set_yticks(range(load.shape[0])); ax.grid(False)
    for (i, j), v in np.ndenumerate(load):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="w" if v < 0.6 * im.norm.vmax else "k", fontsize=8)
    return im
