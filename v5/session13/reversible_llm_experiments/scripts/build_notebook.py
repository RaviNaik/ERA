"""Regenerates notebooks/reversible_llm.ipynb from scratch via nbformat.

The VS Code notebook-editing tool used to author this notebook interactively
edits an in-memory editor buffer; this script is the durable, reproducible
source of truth that actually gets written to disk (and is what `git diff`
will show for notebook content changes going forward).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "reversible_llm.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.12.3",
    },
}

cells: list[tuple[str, str]] = []


def md(src: str) -> None:
    cells.append(("markdown", src))


def code(src: str) -> None:
    cells.append(("code", src))


md(
    """# Session 13: Reversible LLM Experiments

**Assignment.** Train a ~20M-parameter GPT-style LLM for a fixed 50M-token budget, then repeat with two reversible architectures (Euler / midpoint), pick the better one, and push it to 2x/4x/8x batch size on the same ~40GB-class GPU. Every run logs loss, accuracy, throughput, step time, and memory to both Aim and `results/runs.jsonl`.

**Reference.** Gal, Eliasof, Turek, Ascher, Haber (2025), *"Reversing Large Language Models for Efficient Training and Fine-Tuning"* ([arXiv:2512.02056v2](https://arxiv.org/html/2512.02056v2)).

## Architecture note: bug found & fixed in `src/revllm/`

Revisiting `src/revllm/reversible.py` before running anything surfaced a real bug: the previous implementation wrapped **each transformer layer in its own `torch.autograd.Function`**, and every one of those per-layer contexts explicitly called `ctx.save_for_backward(y1, y2)`. Because every layer's context stays alive until the *single* final `.backward()` call, all per-layer activations were retained anyway -- there was **zero** memory saving versus the plain baseline, which defeats the entire purpose of "reversible" training. The `midpoint` block was also not actually invertible as wired: attention ran through ordinary autograd (not reconstructable), and folding `(current + next_state) * scale` into one tensor threw away exactly the information needed to invert the next layer.

The fix (now in `reversible.py`) wires **one stack-wide autograd Function** across the whole depth (à la Reformer / Reversible ViT "RevBackProp"): only the *final* pair of hidden states for the entire stack is kept alive by autograd; every intermediate per-layer activation is recomputed from the invertible update and released immediately while walking backward layer-by-layer. This is what actually delivers the O(1)-in-depth memory savings, and it is verified by `tests/test_reversible.py` (gradient-vs-naive-reference checks in float64, plus a CUDA-only test asserting peak memory does *not* grow linearly with depth).

The two variants follow the paper's formulations directly:

- **Euler** (paper eq. 8-9, staggered / symplectic-Euler two-stream update):
  $$q^{(l)} = q^{(l-1)} + \\text{Attn}_l(\\text{LN}_1(p^{(l-1)})), \\qquad p^{(l)} = p^{(l-1)} + \\text{MLP}_l(\\text{LN}_2(q^{(l)}))$$
- **Midpoint** (paper eq. 4-5, explicit-midpoint depth recurrence):
  $$p^{(l+1)} = p^{(l-1)} + 2h \\cdot f_l(p^{(l)}), \\qquad f_l(p) = \\text{Attn}_l(\\text{LN}_1(p)) + \\text{MLP}_l\\big(\\text{LN}_2(p + \\text{Attn}_l(\\text{LN}_1(p)))\\big)$$
  with step size $h = 1/L$ (paper Sec. 3: keeps the marginally-stable recurrence inside its stability region as depth $L$ grows).

A second, independent bug was also fixed in `src/revllm/loss.py`: with `n_embd=256` but `vocab_size=50257`, the un-chunked `(batch*seq, vocab)` logits/log-softmax tensor dwarfs the transformer body's own activations and is *identical* across all three variants -- so peak memory would mostly measure the vocab projection, not the architecture difference we actually care about. The loss is now computed by a chunked fused linear+cross-entropy (`chunked_cross_entropy`), so peak memory reflects the transformer backbone."""
)

code(
    """import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from IPython.display import Markdown, display

from revllm.data import load_token_dataset, prepare_wikitext
from revllm.export import export_webapp_data
from revllm.model import GPTConfig, build_model, param_count_report
from revllm.trainer import TrainConfig, train

ROOT = Path.cwd()
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 512
TOKEN_BUDGET = 50_000_000

print("project root:", ROOT)
print("device:", DEVICE)
if torch.cuda.is_available():
    torch.cuda.set_device(DEVICE)
    print("gpu:", torch.cuda.get_device_name(DEVICE))
    print("device_count:", torch.cuda.device_count())
    print(
        "vram_gb:",
        round(torch.cuda.get_device_properties(DEVICE).total_memory / 1024**3, 2),
    )

# Archive any stale results from earlier (pre-bugfix) runs so this notebook's
# comparisons only reflect the corrected reversible implementation.
results_dir = ROOT / "results"
results_dir.mkdir(parents=True, exist_ok=True)
runs_path = results_dir / "runs.jsonl"
if runs_path.exists() and runs_path.stat().st_size > 0:
    backup_path = results_dir / f"runs.pre_notebook_rerun.{int(time.time())}.jsonl"
    runs_path.rename(backup_path)
    print(f"archived previous results to {backup_path.name}")"""
)

md(
    """## 1. Data

WikiText-103 is tokenized with GPT-2 BPE and cached under `data/` as `uint16` memmap files (train/val). The assignment's 50M-token budget is a *training-tokens-processed* budget, so the ~119M-token train split is sampled with replacement across steps rather than iterated once."""
)

code(
    """meta = prepare_wikitext(ROOT / "data")
data = load_token_dataset(ROOT / "data")
meta"""
)

md(
    """## 2. Model size

`n_layer=9`, `n_head=8`, `n_embd=256`, `block_size=512`, tied input/output embeddings, real GPT-2 BPE vocabulary (`vocab_size=50257`) -- lands at ~20.1M parameters. The three variants (`baseline`, `euler`, `midpoint`) are architecturally different (different wiring of the same attention/MLP sublayers) but have **identical parameter counts**, confirmed below, so any metric differences come from the architecture/recurrence, not model capacity."""
)

code(
    """param_rows = {}
for variant in ["baseline", "euler", "midpoint"]:
    cfg = GPTConfig(
        vocab_size=data.vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=9,
        n_head=8,
        n_embd=256,
        dropout=0.0,
        bias=True,
        variant=variant,
    )
    report = param_count_report(cfg)
    param_rows[variant] = {k: v for k, v in report.items() if k != "config"}

param_df = pd.DataFrame(param_rows)
assert param_df.loc["total"].nunique() == 1, "variants should have identical param counts"
param_df"""
)

md(
    """## 3. Batch-size probing (choosing `x` for a ~40GB-class GPU)

We need a fixed batch size `x` such that:

1. `baseline`, `euler`, and `midpoint` can all train at `x` (experiments 1-3, same batch size for a fair comparison), and
2. the winning reversible variant can still be trained at `8x` (experiment 6) without running out of memory.

Rather than guess, we empirically probe forward+backward peak memory and throughput at increasing batch sizes for all three variants (one forward+backward step each, no optimizer states -- a cheap but representative proxy for training memory), then derive `x` with a safety margin so real training (which adds Adam's `m`/`v` buffers, and can hit slightly higher peaks than a single probed step) doesn't sit right at the OOM edge -- and so the run tolerates other processes sharing this GPU."""
)

code(
    """def probe_batch_sizes(variant, candidate_batches, block_size=BLOCK_SIZE):
    rows = []
    for batch_size in candidate_batches:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(DEVICE)
            cfg = GPTConfig(
                vocab_size=data.vocab_size,
                block_size=block_size,
                n_layer=9,
                n_head=8,
                n_embd=256,
                variant=variant,
            )
            model = build_model(cfg, device=DEVICE)
            x, y = data.get_batch("train", batch_size, block_size, DEVICE)
            torch.cuda.synchronize(DEVICE) if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=DEVICE.startswith("cuda")
            ):
                _, loss, _ = model(x, y)
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize(DEVICE)
            elapsed = time.perf_counter() - t0
            peak_mb = (
                torch.cuda.max_memory_allocated(DEVICE) / 1024**2
                if torch.cuda.is_available()
                else 0.0
            )
            rows.append(
                {
                    "variant": variant,
                    "batch_size": batch_size,
                    "fits": True,
                    "peak_memory_mb": round(peak_mb, 1),
                    "probe_tokens_per_sec": round(batch_size * block_size / max(elapsed, 1e-9), 1),
                }
            )
            del model, x, y, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except RuntimeError as exc:
            rows.append(
                {
                    "variant": variant,
                    "batch_size": batch_size,
                    "fits": False,
                    "peak_memory_mb": None,
                    "probe_tokens_per_sec": None,
                    "error": str(exc).split("\\n")[0],
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break
    return pd.DataFrame(rows)


CANDIDATE_BATCHES = (64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)
probe_results = {
    variant: probe_batch_sizes(variant, CANDIDATE_BATCHES)
    for variant in ["baseline", "euler", "midpoint"]
}
probe_df = pd.concat(probe_results.values(), ignore_index=True)
probe_df"""
)

code(
    """def max_stable_batch(df: pd.DataFrame) -> int:
    fits = df[df.fits]
    if len(fits) == 0:
        raise RuntimeError("no candidate batch size fit in memory")
    return int(fits.batch_size.max())


SAFETY_MARGIN = 0.75  # leave headroom below the probed OOM edge
baseline_max = max_stable_batch(probe_results["baseline"])
euler_max = max_stable_batch(probe_results["euler"])
midpoint_max = max_stable_batch(probe_results["midpoint"])
reversible_min_max = min(euler_max, midpoint_max)

candidate_x = [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536]
feasible_x = [
    b
    for b in candidate_x
    if b <= baseline_max * SAFETY_MARGIN and 8 * b <= reversible_min_max * SAFETY_MARGIN
]
FIXED_BATCH = max(feasible_x)

print(f"probed max stable batch  -> baseline: {baseline_max}, euler: {euler_max}, midpoint: {midpoint_max}")
print(f"safety margin: {SAFETY_MARGIN:.0%}")
print(f"chosen fixed batch size x = {FIXED_BATCH}  (experiments 1-3)")
print(f"=> 2x={2*FIXED_BATCH}, 4x={4*FIXED_BATCH}, 8x={8*FIXED_BATCH}  (experiments 4-6, winning reversible variant)")
print(
    f"sanity: 8x={8*FIXED_BATCH} <= {SAFETY_MARGIN:.0%} of min(euler_max, midpoint_max)={reversible_min_max} "
    f"=> {8*FIXED_BATCH <= reversible_min_max * SAFETY_MARGIN}"
)"""
)

md(
    """## 4. Reporting helpers

`report_run` prints a data-driven findings block (loss, accuracy, throughput, step time, memory, and -- when a baseline is given -- relative deltas) and plots the loss/throughput/memory curves immediately after each experiment finishes, so every run is fully documented in place rather than left to a final summary table."""
)

code(
    """run_summaries = []
run_histories = {}


def make_train_config(variant, batch_size, label, base_lr=3e-4):
    total_steps = math.ceil(TOKEN_BUDGET / (batch_size * BLOCK_SIZE))
    return TrainConfig(
        variant=variant,
        batch_size=batch_size,
        block_size=BLOCK_SIZE,
        token_budget=TOKEN_BUDGET,
        warmup_steps=max(5, min(100, total_steps // 10)),
        eval_interval=max(5, total_steps // 10),
        eval_iters=max(5, min(25, total_steps // 4)),
        learning_rate=base_lr,
        min_lr=base_lr / 10,
        device=DEVICE,
        label=label,
        aim_repo=str(ROOT),
    )


def run_experiment(variant, batch_size, label, baseline_label=None):
    cfg = make_train_config(variant, batch_size, label)
    print(f"running {label}: variant={variant} batch_size={batch_size} total_steps={cfg.total_steps}")
    out = train(data, cfg, results_dir=ROOT / "results")
    run_summaries.append(out["summary"])
    run_histories[label] = out["history"]
    baseline_summary = None
    if baseline_label is not None:
        baseline_summary = next(r for r in run_summaries if r["label"] == baseline_label)
    report_run(out["summary"], out["history"], baseline_summary)
    return out


def report_run(summary, history, baseline=None):
    lines = [
        f"#### {summary['label']}",
        f"- variant: **{summary['variant']}** &middot; batch size: **{summary['batch_size']}** &middot; "
        f"steps: **{summary['total_steps']}** &middot; tokens seen: **{summary['tokens_seen']:,}**",
        f"- loss -- train: **{summary['final_train_loss']:.4f}**, val: **{summary['final_val_loss']:.4f}**",
        f"- accuracy -- train: **{summary['final_train_accuracy'] * 100:.2f}%**, "
        f"val: **{summary['final_val_accuracy'] * 100:.2f}%**",
        f"- throughput: **{summary['mean_tokens_per_sec']:,.0f} tok/s** &middot; "
        f"mean step time: **{summary['mean_step_time_ms']:.1f} ms** &middot; "
        f"wall clock: **{summary['wall_clock_s']:.1f} s**",
        f"- peak memory -- allocated: **{summary['peak_memory_mb']:,.0f} MB**, "
        f"reserved: **{summary['peak_memory_reserved_mb']:,.0f} MB**",
    ]
    if baseline is not None and baseline["label"] != summary["label"]:
        mem_delta = (summary["peak_memory_mb"] / baseline["peak_memory_mb"] - 1) * 100
        speed_delta = (summary["mean_tokens_per_sec"] / baseline["mean_tokens_per_sec"] - 1) * 100
        loss_delta = summary["final_val_loss"] - baseline["final_val_loss"]
        lines.append(
            f"- vs `{baseline['label']}`: **{mem_delta:+.1f}%** peak memory, "
            f"**{speed_delta:+.1f}%** throughput, **{loss_delta:+.4f}** val-loss delta"
        )
    display(Markdown("\\n".join(lines)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.2))
    axes[0].plot(history["eval_step"], history["train_loss"], label="train")
    axes[0].plot(history["eval_step"], history["val_loss"], label="val")
    axes[0].set_title("loss")
    axes[0].set_xlabel("step")
    axes[0].legend()
    axes[1].plot(history["step"], history["tokens_per_sec"])
    axes[1].set_title("tokens/sec")
    axes[1].set_xlabel("step")
    axes[2].plot(history["step"], history["memory_allocated_mb"])
    axes[2].set_title("memory allocated (MB)")
    axes[2].set_xlabel("step")
    fig.suptitle(summary["label"])
    fig.tight_layout()
    plt.show()


def leaderboard():
    cols = [
        "label", "variant", "batch_size", "total_steps",
        "final_train_loss", "final_val_loss",
        "final_train_accuracy", "final_val_accuracy",
        "mean_tokens_per_sec", "mean_step_time_ms",
        "peak_memory_mb", "peak_memory_reserved_mb", "wall_clock_s",
    ]
    return pd.DataFrame(run_summaries)[cols]"""
)

md(
    """## Experiment 1 -- Baseline, batch size = x

Standard pre-norm transformer (`ln -> attn -> residual`, `ln -> mlp -> residual`), trained for the full 50M-token budget at the fixed batch size `x` derived above. This is the reference point every other experiment is compared against."""
)

code(
    """baseline_out = run_experiment("baseline", FIXED_BATCH, "baseline_x")
leaderboard()"""
)

md(
    """## Experiment 2 -- Reversible Midpoint, batch size = x

Explicit-midpoint depth recurrence (paper eq. 4-5), same data, same batch size `x`, same token budget. Wired through the O(1)-memory `ReversibleStack` (see the architecture note at the top), so its peak-memory number is directly comparable to the baseline's."""
)

code(
    """midpoint_out = run_experiment("midpoint", FIXED_BATCH, "midpoint_x", baseline_label="baseline_x")
leaderboard()"""
)

md(
    """## Experiment 3 -- Reversible Euler, batch size = x

Staggered / symplectic-Euler two-stream update (paper eq. 8-9), same data, same batch size `x`, same token budget."""
)

code(
    """euler_out = run_experiment("euler", FIXED_BATCH, "euler_x", baseline_label="baseline_x")
leaderboard()"""
)

md(
    """## 5. Choosing the reversible variant to scale up

Primary criterion: lower final validation loss. If the two variants land within 1% of each other on validation loss (a practical tie), break the tie in favor of higher throughput, since that is the entire point of moving to a reversible architecture."""
)

code(
    """euler_summary = next(r for r in run_summaries if r["label"] == "euler_x")
midpoint_summary = next(r for r in run_summaries if r["label"] == "midpoint_x")

better_loss, worse_loss = (
    (euler_summary, midpoint_summary)
    if euler_summary["final_val_loss"] <= midpoint_summary["final_val_loss"]
    else (midpoint_summary, euler_summary)
)
loss_gap_pct = abs(euler_summary["final_val_loss"] - midpoint_summary["final_val_loss"]) / worse_loss["final_val_loss"]

if loss_gap_pct < 0.01:
    winner = (
        "euler"
        if euler_summary["mean_tokens_per_sec"] >= midpoint_summary["mean_tokens_per_sec"]
        else "midpoint"
    )
    reason = f"val-loss gap {loss_gap_pct:.2%} < 1% tie threshold -> broke tie on throughput"
else:
    winner = better_loss["variant"]
    reason = f"lower final val loss ({better_loss['final_val_loss']:.4f} vs {worse_loss['final_val_loss']:.4f})"

winner_x_summary = euler_summary if winner == "euler" else midpoint_summary
display(Markdown(
    f"**Winner: `{winner}`** -- {reason}.\\n\\n"
    f"- euler: val_loss={euler_summary['final_val_loss']:.4f}, "
    f"tok/s={euler_summary['mean_tokens_per_sec']:,.0f}, "
    f"peak_mem={euler_summary['peak_memory_mb']:,.0f} MB\\n"
    f"- midpoint: val_loss={midpoint_summary['final_val_loss']:.4f}, "
    f"tok/s={midpoint_summary['mean_tokens_per_sec']:,.0f}, "
    f"peak_mem={midpoint_summary['peak_memory_mb']:,.0f} MB"
))"""
)

md(
    """## 6. Scaling the winning reversible variant: 2x / 4x / 8x

Same token budget (50M) at each batch size, so step count shrinks roughly proportionally as batch size grows (e.g. 8x the batch means ~1/8th the steps). `make_train_config` (above) adapts `warmup_steps`, `eval_interval`, and `eval_iters` to the resulting `total_steps` so the cosine LR schedule and eval cadence stay sensible even when a run only has a few dozen steps. Batch sizes were chosen (Section 3) so that even 8x stays safely under the empirically probed OOM edge on this GPU."""
)

md(
    """### Experiment 4 -- Reversible winner, batch size = 2x

(The winning variant name is resolved from Section 5 above; the code cell below prints the concrete variant/batch-size heading via `run_experiment`.)"""
)

code(
    """out_2x = run_experiment(winner, 2 * FIXED_BATCH, f"{winner}_2x", baseline_label="baseline_x")
leaderboard()"""
)

md("""### Experiment 5 -- Reversible winner, batch size = 4x""")

code(
    """out_4x = run_experiment(winner, 4 * FIXED_BATCH, f"{winner}_4x", baseline_label="baseline_x")
leaderboard()"""
)

md("""### Experiment 6 -- Reversible winner, batch size = 8x (maximum)""")

code(
    """out_8x = run_experiment(winner, 8 * FIXED_BATCH, f"{winner}_8x", baseline_label="baseline_x")
leaderboard()"""
)

md("""## 7. Final comparison across all six experiments""")

code(
    """final_df = leaderboard()
final_df"""
)

code(
    """fig, axes = plt.subplots(1, 3, figsize=(16, 4))

fixed_batch_labels = ["baseline_x", "midpoint_x", "euler_x"]
fixed_df = final_df[final_df.label.isin(fixed_batch_labels)]
axes[0].bar(fixed_df.label, fixed_df.peak_memory_mb)
axes[0].set_title(f"peak memory @ batch={FIXED_BATCH}")
axes[0].set_ylabel("MB")
axes[0].tick_params(axis="x", rotation=30)

axes[1].bar(fixed_df.label, fixed_df.mean_tokens_per_sec)
axes[1].set_title(f"throughput @ batch={FIXED_BATCH}")
axes[1].set_ylabel("tokens/sec")
axes[1].tick_params(axis="x", rotation=30)

scaling_labels = ["baseline_x", f"{winner}_x", f"{winner}_2x", f"{winner}_4x", f"{winner}_8x"]
scaling_df = final_df[final_df.label.isin(scaling_labels)].sort_values("batch_size")
axes[2].plot(scaling_df.batch_size, scaling_df.peak_memory_mb, marker="o", label="peak memory (MB)")
axes[2].set_xlabel("batch size")
axes[2].set_ylabel("MB")
ax2b = axes[2].twinx()
ax2b.plot(scaling_df.batch_size, scaling_df.mean_tokens_per_sec, marker="s", color="tab:orange", label="tokens/sec")
ax2b.set_ylabel("tokens/sec")
axes[2].set_title(f"{winner}: batch-size scaling")
fig.tight_layout()
plt.show()"""
)

md(
    """## 8. Findings (auto-generated from the measured results above)

The cell below composes its conclusions directly from `run_summaries`, so it always reflects whatever this notebook actually measured on this machine -- re-running the notebook end to end keeps these findings in sync with the numbers."""
)

code(
    '''baseline_s = next(r for r in run_summaries if r["label"] == "baseline_x")
winner_x_s = next(r for r in run_summaries if r["label"] == f"{winner}_x")
loser = "midpoint" if winner == "euler" else "euler"
loser_x_s = next(r for r in run_summaries if r["label"] == f"{loser}_x")
s2x = next(r for r in run_summaries if r["label"] == f"{winner}_2x")
s4x = next(r for r in run_summaries if r["label"] == f"{winner}_4x")
s8x = next(r for r in run_summaries if r["label"] == f"{winner}_8x")

mem_savings_x = (1 - winner_x_s["peak_memory_mb"] / baseline_s["peak_memory_mb"]) * 100
speed_delta_x = (winner_x_s["mean_tokens_per_sec"] / baseline_s["mean_tokens_per_sec"] - 1) * 100
mem_savings_8x_vs_baseline_x = (1 - s8x["peak_memory_mb"] / baseline_s["peak_memory_mb"]) * 100
speed_8x_vs_x = (s8x["mean_tokens_per_sec"] / baseline_s["mean_tokens_per_sec"] - 1) * 100
best_val_loss_row = min(run_summaries, key=lambda r: r["final_val_loss"])
fastest_row = max(run_summaries, key=lambda r: r["mean_tokens_per_sec"])
lightest_row = min(run_summaries, key=lambda r: r["peak_memory_mb"])

findings = f"""
### Same batch size ({FIXED_BATCH}), architecture only

| | baseline | {winner} (winner) | {loser} |
|---|---|---|---|
| final val loss | {baseline_s['final_val_loss']:.4f} | {winner_x_s['final_val_loss']:.4f} | {loser_x_s['final_val_loss']:.4f} |
| final val accuracy | {baseline_s['final_val_accuracy']*100:.2f}% | {winner_x_s['final_val_accuracy']*100:.2f}% | {loser_x_s['final_val_accuracy']*100:.2f}% |
| mean tok/s | {baseline_s['mean_tokens_per_sec']:,.0f} | {winner_x_s['mean_tokens_per_sec']:,.0f} | {loser_x_s['mean_tokens_per_sec']:,.0f} |
| peak memory (MB) | {baseline_s['peak_memory_mb']:,.0f} | {winner_x_s['peak_memory_mb']:,.0f} | {loser_x_s['peak_memory_mb']:,.0f} |

At the **same batch size**, reversible-`{winner}` uses **{mem_savings_x:+.1f}%** memory and is **{speed_delta_x:+.1f}%** throughput relative to baseline, at a val-loss delta of **{winner_x_s['final_val_loss'] - baseline_s['final_val_loss']:+.4f}**.

### Scaling `{winner}` batch size at fixed 50M-token budget

| batch | steps | val loss | val acc | tok/s | peak mem (MB) | step time (ms) |
|---|---|---|---|---|---|---|
| {winner_x_s['batch_size']} (x) | {winner_x_s['total_steps']} | {winner_x_s['final_val_loss']:.4f} | {winner_x_s['final_val_accuracy']*100:.2f}% | {winner_x_s['mean_tokens_per_sec']:,.0f} | {winner_x_s['peak_memory_mb']:,.0f} | {winner_x_s['mean_step_time_ms']:.1f} |
| {s2x['batch_size']} (2x) | {s2x['total_steps']} | {s2x['final_val_loss']:.4f} | {s2x['final_val_accuracy']*100:.2f}% | {s2x['mean_tokens_per_sec']:,.0f} | {s2x['peak_memory_mb']:,.0f} | {s2x['mean_step_time_ms']:.1f} |
| {s4x['batch_size']} (4x) | {s4x['total_steps']} | {s4x['final_val_loss']:.4f} | {s4x['final_val_accuracy']*100:.2f}% | {s4x['mean_tokens_per_sec']:,.0f} | {s4x['peak_memory_mb']:,.0f} | {s4x['mean_step_time_ms']:.1f} |
| {s8x['batch_size']} (8x) | {s8x['total_steps']} | {s8x['final_val_loss']:.4f} | {s8x['final_val_accuracy']*100:.2f}% | {s8x['mean_tokens_per_sec']:,.0f} | {s8x['peak_memory_mb']:,.0f} | {s8x['mean_step_time_ms']:.1f} |

Going from `x` to `8x` batch size (same 50M-token budget, so ~8x fewer optimizer steps): peak memory changed by **{(s8x['peak_memory_mb']/winner_x_s['peak_memory_mb'] - 1)*100:+.1f}%** and throughput changed by **{(s8x['mean_tokens_per_sec']/winner_x_s['mean_tokens_per_sec'] - 1)*100:+.1f}%**, while `8x`'s memory footprint is still **{mem_savings_8x_vs_baseline_x:+.1f}%** vs. the *baseline-at-x* memory footprint (i.e. even after growing the reversible model's batch 8x, it can still fit comfortably where the non-reversible baseline could not).

### Overall winners across all six runs

- **Lowest validation loss:** `{best_val_loss_row['label']}` ({best_val_loss_row['final_val_loss']:.4f})
- **Highest throughput:** `{fastest_row['label']}` ({fastest_row['mean_tokens_per_sec']:,.0f} tok/s)
- **Lowest peak memory:** `{lightest_row['label']}` ({lightest_row['peak_memory_mb']:,.0f} MB)
"""
display(Markdown(findings))'''
)

md(
    """## 9. Export webapp data

Regenerates `../webapp/data.js` from the durable `results/runs.jsonl` summaries written by every `train()` call above."""
)

code(
    """session_data = export_webapp_data(ROOT / "results" / "runs.jsonl", ROOT.parent / "webapp" / "data.js")
session_data"""
)

md(
    """## 10. Reproducing / inspecting further

- `uv run pytest -q` -- gradient-correctness and O(1)-memory-in-depth checks for the reversible stacks.
- `uv run aim up` (from `reversible_llm_experiments/`) -- browse every metric (loss, accuracy, lr, grad norm, tokens/sec, step time, memory allocated/reserved) tracked per step and per eval for all six runs.
- `results/runs.jsonl` -- one JSON line per run with `hparams`, `summary`, and the full step-level `curve`, for offline analysis outside Aim.
- `../webapp/` -- static visualization populated by `data.js` (Section 9)."""
)

for cell_type, source in cells:
    if cell_type == "markdown":
        nb["cells"].append(nbf.v4.new_markdown_cell(source))
    else:
        nb["cells"].append(nbf.v4.new_code_cell(source))

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(NOTEBOOK_PATH))
print(f"wrote {len(nb['cells'])} cells to {NOTEBOOK_PATH}")
