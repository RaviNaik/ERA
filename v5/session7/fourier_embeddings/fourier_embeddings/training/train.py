"""Train one arm of the embedding comparison (dense / kronecker / fourier /
fourier_narrow / hrr) and record everything needed to compare arms later:

  - Aim experiment tracking (loss, val loss, lr, tokens/sec, grad norm, ...)
  - a plain-text training log (stdout + file)
  - a `metrics.json` summary (final numbers, param breakdown, config) per run
  - periodic checkpoints

This is deliberately a single, explicit script rather than a generic
trainer framework: every argument you can pass is visible in `build_argparser`
below, matching this project's "code with args to run experiments" brief.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import signal
import time
from pathlib import Path

import torch

from fourier_embeddings.data.dataset import BinDataset, load_meta, dtype_for_vocab
from fourier_embeddings.data.tokenizer import load_tokenizer, build_id_to_bytes
from fourier_embeddings.model.config import ModelConfig, TrainConfig
from fourier_embeddings.model.gpt import GPT
from fourier_embeddings.utils.logging_setup import setup_logging

try:
    from aim import Run as AimRun
except ImportError:  # pragma: no cover
    AimRun = None


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)

    # --- run identity ---
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--aim-repo", default="aim_repo")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--no-aim", action="store_true", help="disable Aim tracking")

    # --- data ---
    ap.add_argument("--data-dir", default="data_bin")
    ap.add_argument("--tokenizer-path", default="tokenizer_out/tokenizer.json")

    # --- embedding codec ---
    ap.add_argument("--embedding", default="dense",
                     choices=["dense", "kronecker", "fourier", "fourier_narrow", "hrr"])
    ap.add_argument("--char-dim", type=int, default=256)
    ap.add_argument("--pos-dim", type=int, default=32)
    ap.add_argument("--fourier-dim", type=int, default=32)
    ap.add_argument("--hrr-dim", type=int, default=1024)
    ap.add_argument("--codec-mode", default="dynamic", choices=["dynamic", "cached"])
    ap.add_argument("--tie-weights", action="store_true")

    # --- model size ---
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--n-embd", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--bias", action="store_true")

    # --- optimization ---
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--max-epochs", type=float, default=None,
                     help="if set, overrides --max-steps: steps = epochs * batches_per_epoch")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    # --- schedule / logging ---
    ap.add_argument("--eval-interval", type=int, default=200)
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--log-interval", type=int, default=20)
    ap.add_argument("--save-interval", type=int, default=500)

    # --- infra ---
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--resume", action="store_true",
                     help="continue from results/<run-name>/last.pt (model, optimizer, step, "
                          "history, best-val-loss, and the same Aim run). Errors out if no "
                          "checkpoint is found -- use a fresh --run-name for a real fresh start.")

    return ap


def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def estimate_loss(model: GPT, dataset: BinDataset, block_size: int, batch_size: int,
                   device: str, eval_iters: int, amp_ctx) -> float:
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = dataset.get_batch(block_size, batch_size, device)
        with amp_ctx:
            _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    args = build_argparser().parse_args()

    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(Path(args.log_dir) / f"{args.run_name}.log")
    logger = setup_logging(log_file, name=f"fourier_embeddings.train.{args.run_name}")
    # FileHandler appends by default (nothing is lost across invocations), so
    # stamp a clear boundary every time a run (re)starts -- otherwise two
    # attempts' output lands in the same file with no visual separator.
    logger.info(f"{'='*20} run start (resume={args.resume}) {'='*20}")
    logger.info(f"args: {vars(args)}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- data ---
    meta = load_meta(args.data_dir)
    vocab_size = meta["vocab_size"]
    dtype_np = dtype_for_vocab(vocab_size)
    train_ds = BinDataset(str(Path(args.data_dir) / "train.bin"), dtype_np)
    val_ds = BinDataset(str(Path(args.data_dir) / "val.bin"), dtype_np)
    logger.info(f"data meta: {meta}")

    tok = load_tokenizer(args.tokenizer_path)
    id_to_bytes = build_id_to_bytes(tok)

    if args.max_epochs is not None:
        batches_per_epoch = train_ds.num_batches_per_epoch(args.block_size, args.batch_size)
        args.max_steps = max(1, int(args.max_epochs * batches_per_epoch))
        logger.info(f"--max-epochs={args.max_epochs} -> batches_per_epoch={batches_per_epoch} "
                    f"-> max_steps={args.max_steps}")

    # --- model ---
    model_cfg = ModelConfig(
        vocab_size=vocab_size, block_size=args.block_size, n_layer=args.n_layer,
        n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout, bias=args.bias,
        tie_weights=args.tie_weights, embedding=args.embedding, char_dim=args.char_dim,
        pos_dim=args.pos_dim, fourier_dim=args.fourier_dim, hrr_dim=args.hrr_dim,
        codec_mode=args.codec_mode,
    )
    train_cfg = TrainConfig(
        data_dir=args.data_dir, tokenizer_path=args.tokenizer_path, max_steps=args.max_steps,
        batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate, min_lr=args.min_lr, warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip, eval_interval=args.eval_interval,
        eval_iters=args.eval_iters, log_interval=args.log_interval, save_interval=args.save_interval,
        device=args.device, dtype=args.dtype, compile=args.compile, seed=args.seed,
        run_name=args.run_name, out_dir=args.out_dir, aim_repo=args.aim_repo, log_dir=args.log_dir,
    )

    model = GPT(model_cfg, id_to_bytes).to(args.device)
    if args.compile:
        model = torch.compile(model)

    param_breakdown = model.param_breakdown() if not args.compile else model._orig_mod.param_breakdown()
    logger.info(f"model param breakdown: {param_breakdown}")

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    ptdtype = dtype_map[args.dtype]
    use_amp = args.device.startswith("cuda") and ptdtype != torch.float32
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype) if use_amp else _nullcontext()
    scaler = torch.amp.GradScaler(enabled=(args.dtype == "float16" and args.device.startswith("cuda")))

    raw_model = model._orig_mod if args.compile else model
    optimizer = raw_model.configure_optimizer(args.weight_decay, args.learning_rate, (0.9, 0.95))

    # --- resume, if requested ---
    step = 0
    history = {"step": [], "train_loss": [], "val_loss": [], "lr": [], "tokens_per_sec": []}
    best_val_loss = float("inf")
    cumulative_time_sec = 0.0
    aim_run_hash = None
    ckpt_path = run_dir / "last.pt"
    if args.resume:
        if not ckpt_path.exists():
            raise SystemExit(f"--resume passed but no checkpoint found at {ckpt_path}. "
                              f"Use a fresh --run-name to start over instead of --resume.")
        logger.info(f"resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=args.device)
        if ckpt.get("model_cfg") != model_cfg.to_dict():
            raise SystemExit(
                "refusing to resume: the checkpoint's model_cfg does not match the model "
                "config built from this invocation's args (embedding/model-size/pos-dim/... "
                "flags must be identical to the original run).\n"
                f"checkpoint: {ckpt.get('model_cfg')}\ncurrent:    {model_cfg.to_dict()}"
            )
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler.is_enabled() and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        step = ckpt["step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        history = ckpt.get("history", history)
        cumulative_time_sec = ckpt.get("cumulative_time_sec", 0.0)
        aim_run_hash = ckpt.get("aim_run_hash")
        if ckpt.get("rng_state") is not None:
            torch.set_rng_state(ckpt["rng_state"].cpu())
        if torch.cuda.is_available() and ckpt.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(ckpt["cuda_rng_state"].cpu())
        logger.info(f"resumed at step {step}/{args.max_steps}, best_val_loss={best_val_loss:.4f}, "
                    f"{cumulative_time_sec:.1f}s already spent on this run")
        if step >= args.max_steps:
            logger.warning(f"checkpoint step {step} already >= --max-steps {args.max_steps}; "
                            f"nothing to do (raise --max-steps to train further)")

    # --- Aim ---
    aim_run = None
    if not args.no_aim and AimRun is not None:
        try:
            aim_run = AimRun(run_hash=aim_run_hash, repo=args.aim_repo, experiment="fourier_embeddings")
            aim_run.name = args.run_name
            aim_run["model_config"] = model_cfg.to_dict()
            aim_run["train_config"] = train_cfg.to_dict()
            aim_run["param_breakdown"] = param_breakdown
        except Exception as e:  # pragma: no cover
            logger.warning(f"could not initialize Aim run: {e}")
            aim_run = None
    elif AimRun is None:
        logger.warning("aim not installed; proceeding without experiment tracking")

    def save_checkpoint(path: Path, step_to_resume_at: int, elapsed_this_segment: float):
        torch.save({
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "step": step_to_resume_at,
            "best_val_loss": best_val_loss,
            "history": history,
            "cumulative_time_sec": cumulative_time_sec + elapsed_this_segment,
            "aim_run_hash": aim_run.hash if aim_run is not None else None,
            "model_cfg": model_cfg.to_dict(),
        }, path)

    # Cloud/cluster preemption typically arrives as SIGTERM; treat it like
    # Ctrl-C so the except-block below gets a chance to save an emergency
    # checkpoint before the process is killed for real.
    def _handle_sigterm(signum, frame):
        raise KeyboardInterrupt("received SIGTERM")
    signal.signal(signal.SIGTERM, _handle_sigterm)

    t_start = time.time()
    tokens_per_step = args.batch_size * args.block_size * args.grad_accum_steps

    logger.info(f"starting training: {args.max_steps} steps, {tokens_per_step} tokens/step, "
                f"embedding={args.embedding}")

    t_last = time.time()
    try:
        while step < args.max_steps:
            lr = get_lr(step, args.warmup_steps, args.max_steps, args.learning_rate, args.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for micro in range(args.grad_accum_steps):
                x, y = train_ds.get_batch(args.block_size, args.batch_size, args.device)
                with amp_ctx:
                    _, loss = model(x, y)
                    loss = loss / args.grad_accum_steps
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                accum_loss += loss.item()

            if args.grad_clip > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if step % args.log_interval == 0:
                t_now = time.time()
                dt = t_now - t_last
                t_last = t_now
                toks_per_sec = (tokens_per_step * args.log_interval) / max(dt, 1e-6) if step > 0 else 0.0
                logger.info(f"step {step:6d}/{args.max_steps} | loss {accum_loss:.4f} | lr {lr:.2e} | "
                            f"tok/s {toks_per_sec:,.0f}")
                history["step"].append(step)
                history["train_loss"].append(accum_loss)
                history["lr"].append(lr)
                history["tokens_per_sec"].append(toks_per_sec)
                if aim_run is not None:
                    aim_run.track(accum_loss, name="loss", step=step, context={"subset": "train"})
                    aim_run.track(lr, name="lr", step=step)
                    aim_run.track(toks_per_sec, name="tokens_per_sec", step=step)

            if step % args.eval_interval == 0 or step == args.max_steps - 1:
                val_loss = estimate_loss(raw_model, val_ds, args.block_size, args.batch_size,
                                          args.device, args.eval_iters, amp_ctx)
                val_ppl = math.exp(min(val_loss, 20))
                logger.info(f"step {step:6d} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")
                history["val_loss"].append({"step": step, "val_loss": val_loss})
                if aim_run is not None:
                    aim_run.track(val_loss, name="loss", step=step, context={"subset": "val"})
                    aim_run.track(val_ppl, name="perplexity", step=step, context={"subset": "val"})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({"model": raw_model.state_dict(), "step": step,
                                "model_cfg": model_cfg.to_dict()}, run_dir / "best.pt")

            if step > 0 and step % args.save_interval == 0:
                save_checkpoint(run_dir / "last.pt", step + 1, time.time() - t_start)
                logger.info(f"checkpoint saved at step {step} -> {run_dir / 'last.pt'}")

            step += 1
    except KeyboardInterrupt:
        # Covers both a real Ctrl-C and SIGTERM (redirected above) -- e.g. a
        # preemptible/spot A6000 instance being reclaimed mid-run. Save an
        # emergency checkpoint at the last *completed* step so `--resume`
        # picks up cleanly, then exit non-zero so a wrapper script (e.g.
        # run_experiment.py) can tell this run did not finish normally.
        save_checkpoint(run_dir / "last.pt", step, time.time() - t_start)
        logger.warning(f"interrupted at step {step}/{args.max_steps}; emergency checkpoint saved "
                        f"to {run_dir / 'last.pt'}. Re-run with --resume to continue.")
        if aim_run is not None:
            aim_run.close()
        raise SystemExit(130)

    total_time = cumulative_time_sec + (time.time() - t_start)
    final_val_loss = estimate_loss(raw_model, val_ds, args.block_size, args.batch_size,
                                    args.device, args.eval_iters, amp_ctx)
    torch.save({"model": raw_model.state_dict(), "step": step, "model_cfg": model_cfg.to_dict()},
               run_dir / "final.pt")

    metrics = {
        "run_name": args.run_name,
        "embedding": args.embedding,
        "model_config": model_cfg.to_dict(),
        "train_config": train_cfg.to_dict(),
        "param_breakdown": param_breakdown,
        "total_params": param_breakdown["total"],
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_loss": final_val_loss,
        "best_val_loss": best_val_loss,
        "final_val_perplexity": math.exp(min(final_val_loss, 20)),
        "total_time_sec": total_time,
        "max_steps": args.max_steps,
        "tokens_trained": args.max_steps * tokens_per_step,
        "history": history,
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info(f"training complete in {total_time:.1f}s. final_val_loss={final_val_loss:.4f} "
                f"best_val_loss={best_val_loss:.4f}. wrote {metrics_path}")

    if aim_run is not None:
        aim_run["final_metrics"] = {k: v for k, v in metrics.items() if k != "history"}
        aim_run.close()


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
