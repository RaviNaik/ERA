"""Run the controlled multi-arm comparison in one command (research note
Sec. 10, item 1): dense / kronecker / fourier / fourier_narrow (and
optionally hrr), all with identical model size, data, and optimization
hyperparameters -- the only thing that differs between runs is
`--embedding`. Each arm is run as a subprocess (a clean CUDA/Aim context per
run), and this script aggregates their `metrics.json` files into one
`results/<experiment-name>/comparison.json` + a Markdown summary table.

Multi-GPU: pass `--gpu-ids 0 1` (etc.) to run up to `len(gpu-ids)` arms
*concurrently*, one per physical GPU (via `CUDA_VISIBLE_DEVICES`), instead of
one arm at a time on a single card -- e.g. on a 2-GPU box, 4 arms finish in
roughly the time of 2 sequential arms instead of 4. This is arm-level
parallelism (each arm still trains on exactly one GPU), not single-model
multi-GPU (DDP) training -- it fits this project's actual bottleneck (many
independent runs to compare) with far less complexity/risk than distributed
training would add for one model.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    ap.add_argument("--gpu-ids", type=int, nargs="+", default=None,
                     help="physical GPU ids to run arms on concurrently, one arm per id "
                          "(e.g. --gpu-ids 0 1 on a 2-GPU box). Each arm subprocess gets "
                          "CUDA_VISIBLE_DEVICES pinned to exactly one id, so inside that "
                          "process it always sees a single GPU as 'cuda:0' -- no --device "
                          "override needed. Default: sequential, one arm at a time, no "
                          "CUDA_VISIBLE_DEVICES override (unchanged single-GPU behavior).")
    ap.add_argument("--require-cuda", action="store_true",
                     help="forwarded to fe-train: abort an arm immediately if CUDA isn't "
                          "available instead of silently training it on CPU.")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-aim", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print commands without running them")
    ap.add_argument("--force", action="store_true",
                     help="retrain an arm even if results/<experiment-name>_<arm>/metrics.json "
                          "already shows it completed (default: skip completed arms, so "
                          "re-running this command after an interruption only trains what's "
                          "left -- and continues a mid-run arm from its last checkpoint "
                          "instead of restarting it).")
    return ap


def plan_arm(arm: str, args: argparse.Namespace, forwarded: list[str]) -> dict:
    """Decide what to do with one arm: skip (already completed), resume (has a
    checkpoint but didn't finish), or start fresh. Pure filesystem checks, no
    subprocess launched here -- safe to call for every arm up front before
    any scheduling decision (sequential or parallel) is made.
    """
    run_name = f"{args.experiment_name}_{arm}"
    run_dir = Path(args.out_dir) / run_name
    metrics_path = run_dir / "metrics.json"
    checkpoint_path = run_dir / "last.pt"

    if not args.force and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return {"arm": arm, "action": "skip", "metrics": metrics}

    cmd = [sys.executable, "-m", "fourier_embeddings.training.train",
           "--run-name", run_name, "--embedding", arm] + forwarded
    resuming = not args.force and checkpoint_path.exists()
    if resuming:
        cmd = cmd + ["--resume"]
    return {"arm": arm, "action": "resume" if resuming else "fresh", "cmd": cmd,
            "run_name": run_name, "metrics_path": metrics_path,
            "checkpoint_path": checkpoint_path}


def run_arm(spec: dict, gpu_id: int | None, dry_run: bool) -> tuple[str, dict]:
    arm = spec["arm"]
    tag = f"gpu {gpu_id}" if gpu_id is not None else "sequential"
    if spec["action"] == "resume":
        logger.info(f"=== arm '{arm}' [{tag}] (resuming from {spec['checkpoint_path']}) ===\n"
                    f"{' '.join(spec['cmd'])}")
    else:
        logger.info(f"=== arm '{arm}' [{tag}] ===\n{' '.join(spec['cmd'])}")
    if dry_run:
        return arm, {"status": "dry_run"}

    env = None
    if gpu_id is not None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    t_arm = time.time()
    try:
        result = subprocess.run(spec["cmd"], env=env)
    except Exception as e:  # noqa: BLE001 -- one arm's unexpected crash must not kill the others
        logger.error(f"arm '{arm}' [{tag}] raised launching/running the subprocess: {e}")
        return arm, {"status": "exception", "error": str(e)}
    elapsed = time.time() - t_arm

    if result.returncode != 0:
        logger.error(f"arm '{arm}' [{tag}] failed with exit code {result.returncode}")
        return arm, {"status": "failed", "returncode": result.returncode}

    metrics_path = spec["metrics_path"]
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"arm '{arm}' [{tag}] finished but {metrics_path} is unreadable: {e}")
            return arm, {"status": "unreadable_metrics", "error": str(e)}
        metrics["wall_clock_sec"] = elapsed
        logger.info(f"arm '{arm}' [{tag}] done in {elapsed:.1f}s: "
                    f"final_val_loss={metrics.get('final_val_loss'):.4f} "
                    f"params={metrics.get('total_params'):,}")
        return arm, metrics
    logger.warning(f"arm '{arm}' [{tag}] finished but no metrics.json found at {metrics_path}")
    return arm, {"status": "missing_metrics"}


def main():
    args = build_argparser().parse_args()

    global logger
    log_file = str(Path(args.log_dir) / f"experiment_{args.experiment_name}.log")
    logger = setup_logging(log_file, name="fourier_embeddings.training.run_experiment")

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
    if args.require_cuda:
        forwarded += ["--require-cuda"]
    if args.no_aim:
        forwarded += ["--no-aim"]

    specs = [plan_arm(arm, args, forwarded) for arm in args.arms]

    run_results: dict[str, dict] = {}
    for spec in specs:
        if spec["action"] == "skip":
            metrics = spec["metrics"]
            run_results[spec["arm"]] = metrics
            logger.info(f"arm '{spec['arm']}' already completed (found {spec['arm']}'s "
                        f"metrics.json); skipping (--force to retrain): "
                        f"final_val_loss={metrics.get('final_val_loss'):.4f}")

    pending = [s for s in specs if s["action"] != "skip"]

    t0 = time.time()
    if args.gpu_ids:
        # A queue of physical GPU ids acts as the mutual-exclusion mechanism:
        # a worker blocks on gpu_pool.get() until a GPU is actually free (not
        # just "a thread is free"), so two arms can never collide on the same
        # GPU regardless of how unevenly long individual arms take.
        gpu_pool: queue.Queue[int] = queue.Queue()
        for gid in args.gpu_ids:
            gpu_pool.put(gid)

        def _worker(spec):
            # Jitter the launch: several arms starting within the same
            # instant (worst case at t=0, when all GPU slots are free at
            # once) can collide on Aim's shared repo lock during Run
            # creation. fe-train itself retries that with backoff, but
            # spreading launches out here means it usually never has to.
            time.sleep(random.uniform(0, 2.5))
            gid = gpu_pool.get()
            try:
                return run_arm(spec, gpu_id=gid, dry_run=args.dry_run)
            finally:
                gpu_pool.put(gid)

        logger.info(f"running {len(pending)} pending arm(s) across GPUs {args.gpu_ids} "
                    f"(up to {len(args.gpu_ids)} concurrently)")
        with ThreadPoolExecutor(max_workers=len(args.gpu_ids)) as ex:
            futures = [ex.submit(_worker, spec) for spec in pending]
            for fut in as_completed(futures):
                arm, result = fut.result()
                run_results[arm] = result
    else:
        for spec in pending:
            arm, result = run_arm(spec, gpu_id=None, dry_run=args.dry_run)
            run_results[arm] = result

    if args.dry_run:
        return

    comparison = {
        "experiment_name": args.experiment_name,
        "arms": args.arms,
        "gpu_ids": args.gpu_ids,
        "total_wall_clock_sec": time.time() - t0,
        "results": run_results,
    }
    comparison_path = exp_dir / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    logger.info(f"wrote {comparison_path}")

    write_markdown_summary(comparison, exp_dir / "comparison.md")
    logger.info(f"wrote {exp_dir / 'comparison.md'}")


def write_markdown_summary(comparison: dict, out_path: Path):
    lines = [f"# Experiment: {comparison['experiment_name']}", "",
             f"Total wall clock: {comparison['total_wall_clock_sec']:.1f}s", "",
             "| Arm | Params (total) | Params (codec) | Final val loss | Best val loss | Final val ppl | Tokens trained |",
             "|---|---|---|---|---|---|---|"]
    for arm in comparison["arms"]:
        m = comparison["results"].get(arm, {})
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
