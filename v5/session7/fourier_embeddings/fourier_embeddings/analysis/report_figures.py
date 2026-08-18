"""Generate the chart set for the empirical validation report from a
completed `fe-run-experiment` output: per-arm `metrics.json` (training/val
loss history) plus the standalone `collisions.json` / `order_sensitivity.json`
/ `crosstalk.json` analysis files.

Palette (dataviz skill's validated default categorical order, slots 1-5,
reused unmodified -- fixed per embedding arm across every chart, never
cycled): dense=blue, kronecker=orange, fourier=aqua, fourier_narrow=yellow,
hrr=magenta. D-values in the crosstalk chart use the skill's sequential blue
ramp instead (a magnitude/continuum, not a categorical identity).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from fourier_embeddings.utils.logging_setup import setup_logging

logger = logging.getLogger("fourier_embeddings.analysis.report_figures")

# --- palette (dataviz skill reference instance, light mode) ---
ARM_COLOR = {
    "dense": "#2a78d6",
    "kronecker": "#eb6834",
    "fourier": "#1baf7a",
    "fourier_narrow": "#eda100",
    "hrr": "#e87ba4",
}
ARM_LABEL = {
    "dense": "Dense", "kronecker": "Kronecker", "fourier": "Fourier",
    "fourier_narrow": "Fourier (narrow-band)", "hrr": "HRR",
}
SEQ_BLUE = {256: "#b7d3f6", 512: "#5598e7", 1024: "#256abf", 2048: "#0d366b"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_SECONDARY,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": GRID, "font.size": 11, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
})


def _style_axes(ax):
    ax.grid(True, axis="y", linewidth=0.8, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)


def load_arm_metrics(results_dir: Path, experiment_name: str, arms: list[str]) -> dict:
    out = {}
    for arm in arms:
        p = results_dir / f"{experiment_name}_{arm}" / "metrics.json"
        if p.exists():
            out[arm] = json.loads(p.read_text(encoding="utf-8"))
        else:
            logger.warning(f"no metrics.json for arm '{arm}' at {p}, skipping")
    return out


def fig_loss_curves(metrics: dict, out_path: Path, key: str, ylabel: str, title: str,
                     log_y: bool = False):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for arm, m in metrics.items():
        hist = m["history"]
        if key == "train_loss":
            xs, ys = hist["step"], hist["train_loss"]
        else:
            xs = [v["step"] for v in hist["val_loss"]]
            ys = [v[key] if key in v else v["val_loss"] for v in hist["val_loss"]]
        ax.plot(xs, ys, color=ARM_COLOR[arm], linewidth=2, label=ARM_LABEL[arm])
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=12)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"wrote {out_path}")


def fig_val_loss_zoomed(metrics: dict, out_path: Path, min_step: int = 1000):
    """The full-range val-loss chart flattens all arms into visual
    indistinguishability once loss collapses near zero (see report text for
    why the collapse itself is a methodology flag, not a quality signal) --
    this restricts to the late-training steps where the actual final-ranking
    differences between arms (the real subject of this comparison) are
    visible at all.
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for arm, m in metrics.items():
        hist = m["history"]
        pts = [(v["step"], v["val_loss"]) for v in hist["val_loss"] if v["step"] >= min_step]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=ARM_COLOR[arm], linewidth=2, marker="o", markersize=5,
                 label=ARM_LABEL[arm])
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation loss")
    ax.set_title(f"Validation loss, steps {min_step}+ (zoomed)", color=INK, fontsize=13,
                  loc="left", pad=12)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"wrote {out_path}")


def fig_val_perplexity_curves(metrics: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for arm, m in metrics.items():
        hist = m["history"]
        xs = [v["step"] for v in hist["val_loss"]]
        ys = [math.exp(min(v["val_loss"], 20)) for v in hist["val_loss"]]
        ax.plot(xs, ys, color=ARM_COLOR[arm], linewidth=2, marker="o", markersize=4,
                 label=ARM_LABEL[arm])
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation perplexity")
    ax.set_title("Validation perplexity over training", color=INK, fontsize=13, loc="left", pad=12)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"wrote {out_path}")


def fig_param_breakdown_stacked(metrics: dict, out_path: Path):
    arms = list(metrics.keys())
    cats = ["token_codec", "position_embedding", "transformer_blocks", "ln_f", "lm_head"]
    cat_labels = ["Token codec", "Position table", "Transformer blocks", "Final LayerNorm", "LM head"]
    cat_colors = ["#2a78d6", "#86b6ef", "#c3c2b7", "#e1e0d9", "#52514e"]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    bottoms = [0.0] * len(arms)
    for cat, clabel, color in zip(cats, cat_labels, cat_colors):
        vals = [metrics[a]["param_breakdown"].get(cat, 0) / 1e6 for a in arms]
        ax.bar([ARM_LABEL[a] for a in arms], vals, bottom=bottoms, color=color,
               label=clabel, width=0.6, edgecolor=SURFACE, linewidth=2, zorder=3)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_ylabel("Parameters (millions)")
    ax.set_title("Parameter breakdown by component", color=INK, fontsize=13, loc="left", pad=12)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"wrote {out_path}")


def fig_codec_params_only(metrics: dict, out_path: Path):
    arms = list(metrics.keys())
    vals = [metrics[a]["param_breakdown"]["token_codec"] / 1e6 for a in arms]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    bars = ax.bar([ARM_LABEL[a] for a in arms], vals, color=[ARM_COLOR[a] for a in arms],
                   width=0.55, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}M", ha="center", va="bottom",
                 fontsize=10, color=INK)
    ax.set_ylabel("Token-codec parameters (millions)")
    ax.set_title("Input-path (codec) parameter count, by arm", color=INK, fontsize=13, loc="left", pad=12)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"wrote {out_path}")


def fig_final_metrics_bar(metrics: dict, out_path: Path, key: str, ylabel: str, title: str,
                           transform=lambda x: x):
    arms = list(metrics.keys())
    vals = [transform(metrics[a][key]) for a in arms]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    bars = ax.bar([ARM_LABEL[a] for a in arms], vals, color=[ARM_COLOR[a] for a in arms],
                   width=0.55, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom",
                 fontsize=10, color=INK)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=12)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"wrote {out_path}")


def fig_wall_clock(metrics: dict, out_path: Path):
    arms = list(metrics.keys())
    vals = [metrics[a]["total_time_sec"] / 60 for a in arms]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    bars = ax.bar([ARM_LABEL[a] for a in arms], vals, color=[ARM_COLOR[a] for a in arms],
                   width=0.55, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f} min", ha="center", va="bottom",
                 fontsize=10, color=INK)
    ax.set_ylabel("Wall-clock training time (minutes)")
    ax.set_title("Training wall-clock time, by arm", color=INK, fontsize=13, loc="left", pad=12)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"wrote {out_path}")


def fig_collisions(collisions: dict, out_path: Path):
    exact = collisions["exact_kronecker_collisions"]["per_script_collision_rate"]
    func = collisions["fourier_functional_collisions"]["per_script_collision_rate"]
    scripts = [s for s in exact if s != "other"]
    scripts_sorted = sorted(scripts, key=lambda s: collisions["exact_kronecker_collisions"]["per_script_total"].get(s, 0), reverse=True)

    x = range(len(scripts_sorted))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    b1 = ax.bar([i - width / 2 for i in x], [exact[s] * 100 for s in scripts_sorted],
                 width=width, color=ARM_COLOR["kronecker"], label="Kronecker (exact collision)", zorder=3)
    b2 = ax.bar([i + width / 2 for i in x], [func[s] * 100 for s in scripts_sorted],
                 width=width, color=ARM_COLOR["fourier"], label="Fourier (functional, cos≥threshold)", zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels([s.capitalize() for s in scripts_sorted], rotation=20, ha="right")
    ax.set_ylabel("Collision rate (%)")
    ax.set_title("Token collision rate per script, real V5-scale vocabulary", color=INK,
                  fontsize=13, loc="left", pad=12)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"wrote {out_path}")


def fig_order_sensitivity_box(order_sens: dict, out_path: Path):
    pairs = order_sens["pairs"]
    kron = [p["kronecker_cosine"] for p in pairs]
    four = [p["fourier_cosine"] for p in pairs]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    bp = ax.boxplot([kron, four], tick_labels=["Kronecker", "Fourier"], patch_artist=True,
                      widths=0.5, medianprops={"color": INK, "linewidth": 2},
                      whiskerprops={"color": AXIS}, capprops={"color": AXIS},
                      flierprops={"markeredgecolor": MUTED, "markersize": 3})
    for patch, color in zip(bp["boxes"], [ARM_COLOR["kronecker"], ARM_COLOR["fourier"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor(color)
    ax.set_ylabel("Cosine similarity (rearranged-word pairs)")
    ax.set_title(f"Order-sensitivity: cosine similarity distribution (n={len(pairs)} pairs)",
                  color=INK, fontsize=12, loc="left", pad=12)
    ax.axhline(0, color=AXIS, linewidth=1, linestyle="--", zorder=1)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"wrote {out_path}")


def fig_crosstalk_accuracy(crosstalk: list, out_path: Path):
    d_values = sorted({row["D"] for row in crosstalk})
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for d in d_values:
        rows = sorted([r for r in crosstalk if r["D"] == d], key=lambda r: r["n_bound"])
        xs = [r["n_bound"] for r in rows]
        ys = [r["top1_accuracy"] * 100 for r in rows]
        color = SEQ_BLUE.get(d, "#256abf")
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5, label=f"D={d}")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Bytes bound per token (n)")
    ax.set_ylabel("Unbinding top-1 accuracy (%)")
    ax.set_title("HRR (Design B) crosstalk: retrieval accuracy vs. D and n", color=INK,
                  fontsize=13, loc="left", pad=12)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=10, title="Code width")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--experiment-name", default="full_run")
    ap.add_argument("--arms", nargs="+", default=["dense", "kronecker", "fourier", "fourier_narrow"])
    ap.add_argument("--out-dir", default="report_figures")
    ap.add_argument("--log-file", default="logs/report_figures.log")
    args = ap.parse_args()

    setup_logging(args.log_file, name="fourier_embeddings.analysis.report_figures")

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_arm_metrics(results_dir, args.experiment_name, args.arms)
    if not metrics:
        raise SystemExit(f"no metrics.json found under {results_dir} for experiment "
                          f"'{args.experiment_name}'")

    fig_loss_curves(metrics, out_dir / "train_loss.png", "train_loss", "Training loss",
                     "Training loss over steps (all arms, identical schedule)")
    fig_loss_curves(metrics, out_dir / "val_loss.png", "val_loss", "Validation loss",
                     "Validation loss over steps (all arms, identical schedule)")
    fig_val_loss_zoomed(metrics, out_dir / "val_loss_zoomed.png")
    fig_val_perplexity_curves(metrics, out_dir / "val_perplexity.png")
    fig_param_breakdown_stacked(metrics, out_dir / "param_breakdown.png")
    fig_codec_params_only(metrics, out_dir / "codec_params.png")
    fig_final_metrics_bar(metrics, out_dir / "final_val_loss.png", "final_val_loss",
                           "Final validation loss", "Final validation loss, by arm")
    fig_final_metrics_bar(metrics, out_dir / "final_val_ppl.png", "final_val_perplexity",
                           "Final validation perplexity", "Final validation perplexity, by arm")
    fig_wall_clock(metrics, out_dir / "wall_clock.png")

    exp_dir = results_dir / args.experiment_name
    collisions_path = exp_dir / "collisions.json"
    if collisions_path.exists():
        fig_collisions(json.loads(collisions_path.read_text(encoding="utf-8")),
                        out_dir / "collisions.png")
    else:
        logger.warning(f"no collisions.json at {collisions_path}, skipping that figure")

    order_sens_path = exp_dir / "order_sensitivity.json"
    if order_sens_path.exists():
        fig_order_sensitivity_box(json.loads(order_sens_path.read_text(encoding="utf-8")),
                                   out_dir / "order_sensitivity.png")
    else:
        logger.warning(f"no order_sensitivity.json at {order_sens_path}, skipping that figure")

    crosstalk_path = exp_dir / "crosstalk.json"
    if crosstalk_path.exists():
        fig_crosstalk_accuracy(json.loads(crosstalk_path.read_text(encoding="utf-8")),
                                out_dir / "crosstalk.png")
    else:
        logger.warning(f"no crosstalk.json at {crosstalk_path}, skipping that figure")

    logger.info(f"all figures written to {out_dir}")


if __name__ == "__main__":
    main()
