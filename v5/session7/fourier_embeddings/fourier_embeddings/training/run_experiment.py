"""Run the controlled multi-arm comparison in one command (research note
Sec. 10, item 1): dense / kronecker / fourier / fourier_narrow (and
optionally hrr), all with identical model size, data, and optimization
hyperparameters -- the only thing that differs between runs is
`--embedding`. Each arm is run as a subprocess (a clean CUDA/Aim context per
run), and this script aggregates their `metrics.json` files into one
`results/<experiment-name>/comparison.json` + a Markdown summary table.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from fourier_embeddings.utils.logging_setup import setup_logging

logger = logging.getLogger("fourier_embeddings.training.run_experiment")

DEFAULT_ARMS = ["dense", "kronecker", "fourier", "fourier_narrow"]


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-name", default="comparison")
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS,
                     choices=["dense", "kronecker", "fourier", "fourier_narrow", "hrr"])
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--aim-repo", default="aim_repo")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--data-dir", default="data_bin")
    ap.add_argument("--tokenizer-path", default="tokenizer_out/tokenizer.json")

    # shared model/training hyperparameters, forwarded verbatim to fe-train
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--n-embd", type=int, default=384)
    ap.add_argument("--pos-dim", type=int, default=32)
    ap.add_argument("--fourier-dim", type=int, default=32)
    ap.add_argument("--hrr-dim", type=int, default=1024)
    ap.add_argument("--codec-mode", default="dynamic", choices=["dynamic", "cached"])
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--max-epochs", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--eval-interval", type=int, default=200)
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--log-interval", type=int, default=20)
    ap.add_argument("--save-interval", type=int, default=500)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-aim", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print commands without running them")
    return ap


def main():
    args = build_argparser().parse_args()

    log_file = str(Path(args.log_dir) / f"experiment_{args.experiment_name}.log")
    logger_local = setup_logging(log_file, name="fourier_embeddings.training.run_experiment")

    exp_dir = Path(args.out_dir) / args.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    forwarded = [
        "--data-dir", args.data_dir, "--tokenizer-path", args.tokenizer_path,
        "--block-size", str(args.block_size), "--n-layer", str(args.n_layer),
        "--n-head", str(args.n_head), "--n-embd", str(args.n_embd),
        "--pos-dim", str(args.pos_dim), "--fourier-dim", str(args.fourier_dim),
        "--hrr-dim", str(args.hrr_dim), "--codec-mode", args.codec_mode,
        "--max-steps", str(args.max_steps),
        "--batch-size", str(args.batch_size), "--grad-accum-steps", str(args.grad_accum_steps),
        "--learning-rate", str(args.learning_rate), "--warmup-steps", str(args.warmup_steps),
        "--eval-interval", str(args.eval_interval), "--eval-iters", str(args.eval_iters),
        "--log-interval", str(args.log_interval), "--save-interval", str(args.save_interval),
        "--dtype", args.dtype, "--seed", str(args.seed),
        "--out-dir", args.out_dir, "--aim-repo", args.aim_repo, "--log-dir", args.log_dir,
    ]
    if args.max_epochs is not None:
        forwarded += ["--max-epochs", str(args.max_epochs)]
    if args.device:
        forwarded += ["--device", args.device]
    if args.no_aim:
        forwarded += ["--no-aim"]

    run_results = {}
    t0 = time.time()
    for arm in args.arms:
        run_name = f"{args.experiment_name}_{arm}"
        cmd = [sys.executable, "-m", "fourier_embeddings.training.train",
               "--run-name", run_name, "--embedding", arm] + forwarded
        logger_local.info(f"=== arm '{arm}' ===\n{' '.join(cmd)}")
        if args.dry_run:
            continue
        t_arm = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t_arm
        if result.returncode != 0:
            logger_local.error(f"arm '{arm}' failed with exit code {result.returncode}")
            run_results[arm] = {"status": "failed", "returncode": result.returncode}
            continue
        metrics_path = Path(args.out_dir) / run_name / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["wall_clock_sec"] = elapsed
            run_results[arm] = metrics
            logger_local.info(f"arm '{arm}' done in {elapsed:.1f}s: "
                               f"final_val_loss={metrics.get('final_val_loss'):.4f} "
                               f"params={metrics.get('total_params'):,}")
        else:
            logger_local.warning(f"arm '{arm}' finished but no metrics.json found at {metrics_path}")
            run_results[arm] = {"status": "missing_metrics"}

    if args.dry_run:
        return

    comparison = {
        "experiment_name": args.experiment_name,
        "arms": args.arms,
        "total_wall_clock_sec": time.time() - t0,
        "results": run_results,
    }
    comparison_path = exp_dir / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    logger_local.info(f"wrote {comparison_path}")

    write_markdown_summary(comparison, exp_dir / "comparison.md")
    logger_local.info(f"wrote {exp_dir / 'comparison.md'}")


def write_markdown_summary(comparison: dict, out_path: Path):
    lines = [f"# Experiment: {comparison['experiment_name']}", "",
             f"Total wall clock: {comparison['total_wall_clock_sec']:.1f}s", "",
             "| Arm | Params (total) | Params (codec) | Final val loss | Best val loss | Final val ppl | Tokens trained |",
             "|---|---|---|---|---|---|---|"]
    for arm, m in comparison["results"].items():
        if "final_val_loss" not in m:
            lines.append(f"| {arm} | - | - | FAILED | - | - | - |")
            continue
        pb = m.get("param_breakdown", {})
        lines.append(
            f"| {arm} | {m.get('total_params', 0):,} | {pb.get('token_codec', 0):,} | "
            f"{m.get('final_val_loss', float('nan')):.4f} | {m.get('best_val_loss', float('nan')):.4f} | "
            f"{m.get('final_val_perplexity', float('nan')):.2f} | {m.get('tokens_trained', 0):,} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
