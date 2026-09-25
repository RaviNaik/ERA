"""Instrumented training loop for the baseline and reversible LLM runs.

Every run writes the same metrics to four sinks:

* Aim (`<aim_repo>/.aim`): per-step and per-eval series, hparams, summary,
  plus Aim's own system metrics (GPU util / memory / power).
* `logs/<label>.log`: human-readable text log.
* `logs/<label>.metrics.jsonl`: one JSON object per step / eval, flushed as
  training runs, so a crashed run still leaves its data behind.
* `results/<label>.json`: full record (config, environment, summary, per-step
  history, eval curve, diagnostics). A one-line summary is also appended to
  `results/runs.jsonl`.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import platform
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .data import TokenDataset
from .model import GPTConfig, build_model, flops_per_token
from .reversible import reversibility_diagnostics

MB = 1024**2

# Dense bf16 tensor-core peak (no sparsity), used for MFU. Matched as a
# substring of torch.cuda.get_device_name(). Unknown GPUs report MFU = None
# unless TrainConfig.gpu_peak_tflops is set.
GPU_PEAK_BF16_TFLOPS = {
    "H100 SXM": 989.0,
    "H100 PCIe": 756.0,
    "H100": 756.0,
    "A100": 312.0,
    "L40S": 362.0,
    "A40": 149.7,
    "A10G": 70.0,
    "A10": 125.0,
    "RTX A6000": 154.8,
    "L4": 121.0,
}


@dataclass
class TrainConfig:
    variant: str = "baseline"
    batch_size: int = 128
    block_size: int = 512
    token_budget: int = 50_000_000
    # Optimisation
    learning_rate: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_frac: float = 0.05
    min_warmup_steps: int = 10
    # LR adjustment when batch_size differs from lr_ref_batch:
    # "none" keeps learning_rate; "sqrt" multiplies it by sqrt(batch/ref);
    # "linear" by batch/ref. The result is capped at max_lr.
    lr_scaling: str = "none"
    lr_ref_batch: int = 128
    max_lr: float = 3e-3
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # Model
    n_layer: int = 9
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = True
    midpoint_h: float = 0.25
    reversible_exact: bool = True
    loss_chunk_size: int = 8192
    # Evaluation: fixed windows shared by every run, spaced by tokens so that
    # curves from different batch sizes line up on the token axis.
    eval_interval_tokens: int = 2_500_000
    eval_windows: int = 128  # periodic eval: 128 x 512 = 65k tokens per split
    eval_batch_size: int = 64
    final_eval_train_windows: int = 487  # matches the full val split size
    # Runtime
    seed: int = 1337
    device: str = "cuda"
    compile_model: bool = False
    amp_dtype: str = "bfloat16"  # "bfloat16" | "float16" | "none"
    gpu_peak_tflops: float | None = None
    warmup_skip_frac: float = 0.05  # steps excluded from throughput stats
    log_interval: int = 20
    console_log: bool = False
    run_diagnostics: bool = True
    # Sinks
    aim_repo: str | None = "."
    experiment: str = "session13_reversible_llm"
    label: str = "run"
    tags: list[str] = field(default_factory=list)

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.block_size

    @property
    def total_steps(self) -> int:
        return math.ceil(self.token_budget / self.tokens_per_step)

    @property
    def peak_lr(self) -> float:
        ratio = self.batch_size / self.lr_ref_batch
        scale = {"none": 1.0, "sqrt": math.sqrt(ratio), "linear": ratio}[
            self.lr_scaling
        ]
        return min(self.learning_rate * scale, self.max_lr)

    @property
    def warmup_steps(self) -> int:
        return min(
            self.total_steps,
            max(self.min_warmup_steps, round(self.warmup_frac * self.total_steps)),
        )

    @property
    def eval_every_steps(self) -> int:
        return max(1, round(self.eval_interval_tokens / self.tokens_per_step))

    def model_config(self, vocab_size: int) -> GPTConfig:
        return GPTConfig(
            vocab_size=vocab_size,
            block_size=self.block_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            dropout=self.dropout,
            bias=self.bias,
            variant=self.variant,
            midpoint_h=self.midpoint_h,
            reversible_exact=self.reversible_exact,
            loss_chunk_size=self.loss_chunk_size,
        )

    def to_dict(self) -> dict:
        return asdict(self) | {
            "tokens_per_step": self.tokens_per_step,
            "total_steps": self.total_steps,
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "eval_every_steps": self.eval_every_steps,
        }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def loss_buffer_mb(model_cfg: GPTConfig) -> float:
    """Batch-independent memory the fused loss keeps from forward to backward.

    The fp32 gradient of the tied (vocab x n_embd) embedding / lm_head weight
    is computed during the forward pass (see loss.py) and held until backward.
    It is counted in `activation_saved_mb` but excluded from the per-sequence
    figure, which should reflect only what grows with batch size.
    """
    return model_cfg.vocab_size * model_cfg.n_embd * 4 / MB


def configure_optimizer(
    model: torch.nn.Module, cfg: TrainConfig
) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)
    kwargs = {}
    if torch.device(cfg.device).type == "cuda":
        kwargs["fused"] = True
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.peak_lr,
        betas=(cfg.beta1, cfg.beta2),
        **kwargs,
    )


def learning_rate(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay to min_lr_ratio * peak_lr."""
    peak, floor = cfg.peak_lr, cfg.peak_lr * cfg.min_lr_ratio
    if step < cfg.warmup_steps:
        return peak * (step + 1) / cfg.warmup_steps
    ratio = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    ratio = min(1.0, ratio)
    return floor + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (peak - floor)


def _amp_dtype(cfg: TrainConfig) -> torch.dtype | None:
    if torch.device(cfg.device).type != "cuda" or cfg.amp_dtype == "none":
        return None
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.amp_dtype]


def _autocast(cfg: TrainConfig):
    dtype = _amp_dtype(cfg)
    return torch.autocast(
        device_type=torch.device(cfg.device).type,
        dtype=dtype or torch.bfloat16,
        enabled=dtype is not None,
    )


def gpu_peak_tflops(device: str, override: float | None = None) -> float | None:
    if override is not None:
        return override
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(device)
    for key, value in GPU_PEAK_BF16_TFLOPS.items():
        if key in name:
            return value
    return None


def environment_info(device: str) -> dict:
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": device,
    }
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        info |= {
            "gpu_name": props.name,
            "gpu_total_memory_mb": props.total_memory / MB,
            "gpu_sm_count": props.multi_processor_count,
            "gpu_capability": f"{props.major}.{props.minor}",
            "cudnn": torch.backends.cudnn.version(),
        }
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        info["git_commit"] = None
    return info


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def _plain(value):
    """JSON round-trip: plain Python types only (Aim rejects str subclasses)."""
    return json.loads(json.dumps(_clean(value), default=str))


def _clean(value):
    """Replaces NaN/inf with None so the JSON output is strictly valid."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


class RunLogger:
    """Fans every metric out to Aim, a text log, and a streaming JSONL file."""

    def __init__(self, cfg: TrainConfig, log_dir: Path, hparams: dict):
        self.cfg = cfg
        log_dir.mkdir(parents=True, exist_ok=True)
        self.text_path = log_dir / f"{cfg.label}.log"
        self.jsonl_path = log_dir / f"{cfg.label}.metrics.jsonl"
        self.jsonl = self.jsonl_path.open("w", encoding="utf-8")

        self.logger = logging.getLogger(f"revllm.{cfg.label}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler = logging.FileHandler(self.text_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(fmt)
        self.logger.addHandler(file_handler)
        if cfg.console_log:
            stream = logging.StreamHandler()
            stream.setFormatter(fmt)
            self.logger.addHandler(stream)

        self.run = None
        if cfg.aim_repo is not None:
            try:
                from aim import Run

                self.run = Run(repo=cfg.aim_repo, experiment=cfg.experiment)
                self.run.name = cfg.label
                for tag in [cfg.variant, f"bs{cfg.batch_size}", *cfg.tags]:
                    self.run.add_tag(tag)
                self.run["hparams"] = _plain(hparams)
            except Exception as exc:  # Aim problems must not kill a long run
                self.logger.warning("Aim disabled: %s", exc)
                self.run = None

    @property
    def aim_hash(self) -> str | None:
        return None if self.run is None else self.run.hash

    def info(self, msg: str, *args) -> None:
        self.logger.info(msg, *args)

    def warning(self, msg: str, *args) -> None:
        self.logger.warning(msg, *args)

    def track(self, kind: str, step: int, metrics: dict, subset: str) -> None:
        record = {"kind": kind, "step": step, "subset": subset} | metrics
        self.jsonl.write(json.dumps(_clean(record), default=_json_default) + "\n")
        self.jsonl.flush()
        if self.run is None:
            return
        for name, value in metrics.items():
            if value is None or not isinstance(value, (int, float)):
                continue
            if isinstance(value, float) and not math.isfinite(value):
                continue
            self.run.track(float(value), name=name, step=step, context={"subset": subset})

    def finish(self, summary: dict) -> None:
        if self.run is not None:
            self.run["summary"] = _plain(summary)
            self.run.close()
        self.jsonl.close()
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data: TokenDataset,
    split: str,
    starts: np.ndarray,
    cfg: TrainConfig,
) -> dict[str, float]:
    """Token-weighted loss / accuracy over a fixed set of windows."""
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0.0, 0
    for i in range(0, len(starts), cfg.eval_batch_size):
        x, y = data.windows(
            split, starts[i : i + cfg.eval_batch_size], cfg.block_size, cfg.device
        )
        with _autocast(cfg):
            _, loss, accuracy = model(x, y)
        n = y.numel()
        total_loss += float(loss) * n
        total_correct += float(accuracy) * n
        total_tokens += n
    model.train()
    mean_loss = total_loss / total_tokens
    return {
        "loss": mean_loss,
        "accuracy": total_correct / total_tokens,
        "perplexity": math.exp(min(mean_loss, 50.0)),
        "tokens": total_tokens,
    }


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def train(
    data: TokenDataset,
    cfg: TrainConfig,
    results_dir: str | Path = "results",
    log_dir: str | Path = "logs",
    progress: bool = True,
) -> dict:
    """Trains one model for `cfg.token_budget` tokens and returns its record.

    Never raises on CUDA OOM or divergence: the returned summary has
    status "oom" / "diverged" instead, so a batch-size sweep can find the
    maximum without killing the notebook.
    """
    if torch.device(cfg.device).type == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"
    if cfg.device == "cuda":
        cfg.device = f"cuda:{torch.cuda.current_device()}"
    device = cfg.device
    is_cuda = torch.device(device).type == "cuda"
    if is_cuda:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)  # training batches only
    results_dir, log_dir = Path(results_dir), Path(log_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    env = environment_info(device)
    model_cfg = cfg.model_config(data.vocab_size)
    flops = flops_per_token(model_cfg)
    peak_tflops = gpu_peak_tflops(device, cfg.gpu_peak_tflops)
    gpu_total_mb = env.get("gpu_total_memory_mb")

    if is_cuda:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    mem_before_model = torch.cuda.memory_allocated(device) if is_cuda else 0

    status, error = "ok", None
    model = optimizer = None
    history: dict[str, list] = {
        k: []
        for k in (
            "step", "tokens", "lr", "loss", "accuracy", "grad_norm",
            "step_time_ms", "data_ms", "forward_ms", "backward_ms", "optimizer_ms",
            "tokens_per_sec", "model_tflops", "mfu",
            "memory_allocated_mb", "memory_reserved_mb", "step_peak_memory_mb",
            "activation_saved_mb",
        )
    }
    evals: dict[str, list] = {
        k: []
        for k in (
            "step", "tokens", "train_loss", "val_loss", "train_accuracy",
            "val_accuracy", "val_perplexity", "wall_clock_s",
        )
    }
    memory_info: dict[str, float] = {}
    diagnostics: dict = {}
    final_eval: dict = {}

    n_params = n_params_non_embedding = 0
    param_memory_mb = None
    logger: RunLogger | None = None
    start = time.perf_counter()
    eval_time_s = 0.0
    try:
        model = build_model(model_cfg, device=device, compile_model=cfg.compile_model)
        raw_model = getattr(model, "_orig_mod", model)
        n_params = sum(p.numel() for p in model.parameters())
        n_params_non_embedding = n_params - raw_model.transformer.wte.weight.numel() - raw_model.transformer.wpe.weight.numel()
        param_memory_mb = (
            (torch.cuda.memory_allocated(device) - mem_before_model) / MB if is_cuda else 0.0
        )
        optimizer = configure_optimizer(model, cfg)

        hparams = cfg.to_dict() | {
            "n_params": n_params,
            "n_params_non_embedding": n_params_non_embedding,
            "model_config": asdict(model_cfg),
            "flops_per_token": flops,
            "gpu_peak_tflops": peak_tflops,
            "env": env,
        }
        logger = RunLogger(cfg, log_dir, hparams)
        logger.info(
            "start %s | variant=%s batch=%d steps=%d tokens/step=%d peak_lr=%.2e warmup=%d eval_every=%d",
            cfg.label, cfg.variant, cfg.batch_size, cfg.total_steps,
            cfg.tokens_per_step, cfg.peak_lr, cfg.warmup_steps, cfg.eval_every_steps,
        )
        logger.info("params total=%d non-embedding=%d | env=%s", n_params, n_params_non_embedding, env)

        eval_train_starts = data.eval_starts("train", cfg.block_size, cfg.eval_windows, seed=1)
        eval_val_starts = data.eval_starts("val", cfg.block_size, cfg.eval_windows, seed=2)

        def run_eval(step: int, tokens: int) -> None:
            nonlocal eval_time_s
            t0 = time.perf_counter()
            tr = evaluate(model, data, "train", eval_train_starts, cfg)
            va = evaluate(model, data, "val", eval_val_starts, cfg)
            eval_time_s += time.perf_counter() - t0
            evals["step"].append(step)
            evals["tokens"].append(tokens)
            evals["train_loss"].append(tr["loss"])
            evals["val_loss"].append(va["loss"])
            evals["train_accuracy"].append(tr["accuracy"])
            evals["val_accuracy"].append(va["accuracy"])
            evals["val_perplexity"].append(va["perplexity"])
            evals["wall_clock_s"].append(time.perf_counter() - start)
            for subset, m in (("eval_train", tr), ("eval_val", va)):
                logger.track("eval", step, {
                    "loss": m["loss"], "accuracy": m["accuracy"],
                    "perplexity": m["perplexity"], "tokens_seen": tokens,
                }, subset)
            logger.info(
                "eval step=%d tokens=%d | train_loss=%.4f val_loss=%.4f val_acc=%.4f val_ppl=%.2f",
                step, tokens, tr["loss"], va["loss"], va["accuracy"], va["perplexity"],
            )

        run_eval(0, 0)  # loss at initialisation (~ln(vocab) = 10.8)

        model.train()
        iterator = range(cfg.total_steps)
        if progress:
            iterator = tqdm(iterator, desc=cfg.label, leave=True)
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)] if is_cuda else None

        for step in iterator:
            lr = learning_rate(step, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            if is_cuda:
                torch.cuda.reset_peak_memory_stats(device)
            t_step = time.perf_counter()
            x, y = data.get_batch("train", cfg.batch_size, cfg.block_size, device, gen)
            data_ms = (time.perf_counter() - t_step) * 1000.0

            mem_pre = torch.cuda.memory_allocated(device) if is_cuda else 0
            if is_cuda:
                events[0].record()
            with _autocast(cfg):
                _, loss, accuracy = model(x, y)
            if is_cuda:
                events[1].record()
            mem_post_fwd = torch.cuda.memory_allocated(device) if is_cuda else 0
            loss.backward()
            if is_cuda:
                events[2].record()
            grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if is_cuda:
                events[3].record()
                torch.cuda.synchronize(device)
            step_time_s = max(time.perf_counter() - t_step, 1e-9)

            loss_v, acc_v, grad_norm = float(loss), float(accuracy), float(grad_norm_t)
            tokens = (step + 1) * cfg.tokens_per_step
            tokens_per_sec = cfg.tokens_per_step / step_time_s
            model_tflops = flops["model"] * tokens_per_sec / 1e12
            mfu = model_tflops / peak_tflops if peak_tflops else None
            if is_cuda:
                fwd_ms = events[0].elapsed_time(events[1])
                bwd_ms = events[1].elapsed_time(events[2])
                opt_ms = events[2].elapsed_time(events[3])
                mem_alloc = torch.cuda.memory_allocated(device) / MB
                mem_reserved = torch.cuda.memory_reserved(device) / MB
                step_peak = torch.cuda.max_memory_allocated(device) / MB
            else:
                fwd_ms = bwd_ms = opt_ms = None
                mem_alloc = mem_reserved = step_peak = 0.0
            activation_saved = (mem_post_fwd - mem_pre) / MB
            if step == 0:
                # After the first step: params + AdamW m/v (+ fused-loss buffers are gone).
                memory_info["static_after_first_step_mb"] = mem_alloc
                memory_info["optimizer_state_mb"] = mem_alloc - param_memory_mb - mem_before_model / MB

            row = {
                "step": step, "tokens": tokens, "lr": lr, "loss": loss_v,
                "accuracy": acc_v, "grad_norm": grad_norm,
                "step_time_ms": step_time_s * 1000.0, "data_ms": data_ms,
                "forward_ms": fwd_ms, "backward_ms": bwd_ms, "optimizer_ms": opt_ms,
                "tokens_per_sec": tokens_per_sec, "model_tflops": model_tflops, "mfu": mfu,
                "memory_allocated_mb": mem_alloc, "memory_reserved_mb": mem_reserved,
                "step_peak_memory_mb": step_peak, "activation_saved_mb": activation_saved,
            }
            for key, value in row.items():
                history[key].append(value)
            logger.track("step", step, {k: v for k, v in row.items() if k != "step"}, "train")

            if step % cfg.log_interval == 0 or step == cfg.total_steps - 1:
                logger.info(
                    "step %d/%d | loss %.4f acc %.4f | lr %.2e | gnorm %.3f | %.0f tok/s | "
                    "step %.1fms (fwd %s bwd %s opt %s) | peak %.0fMB act %.0fMB",
                    step, cfg.total_steps, loss_v, acc_v, lr, grad_norm, tokens_per_sec,
                    step_time_s * 1000.0,
                    f"{fwd_ms:.1f}" if fwd_ms is not None else "-",
                    f"{bwd_ms:.1f}" if bwd_ms is not None else "-",
                    f"{opt_ms:.1f}" if opt_ms is not None else "-",
                    step_peak, activation_saved,
                )
                if progress:
                    iterator.set_postfix(loss=f"{loss_v:.3f}", tok_s=f"{tokens_per_sec:,.0f}", mem=f"{step_peak:,.0f}MB")

            if not math.isfinite(loss_v):
                status, error = "diverged", f"non-finite loss at step {step}"
                logger.warning(error)
                break

            if (step + 1) % cfg.eval_every_steps == 0 and step != cfg.total_steps - 1:
                run_eval(step + 1, tokens)

        if status == "ok":
            run_eval(cfg.total_steps, cfg.total_steps * cfg.tokens_per_step)
            # Final, larger evaluation: the *entire* validation split plus an
            # equally sized fixed train subset. These are the headline numbers.
            t0 = time.perf_counter()
            final_eval["val"] = evaluate(model, data, "val", data.eval_starts("val", cfg.block_size, None), cfg)
            final_eval["train"] = evaluate(
                model, data, "train",
                data.eval_starts("train", cfg.block_size, cfg.final_eval_train_windows, seed=3), cfg,
            )
            eval_time_s += time.perf_counter() - t0
            logger.info("final eval | %s", final_eval)

            if cfg.run_diagnostics and cfg.variant in ("euler", "midpoint"):
                raw_model.eval()
                x, _ = data.windows("val", data.eval_starts("val", cfg.block_size, 4, seed=4), cfg.block_size, device)
                with torch.no_grad():
                    pos = torch.arange(x.shape[1], device=x.device)
                    emb = raw_model.transformer.wte(x) + raw_model.transformer.wpe(pos)
                diagnostics = reversibility_diagnostics(raw_model.transformer.h, emb, _amp_dtype(cfg))
                raw_model.train()
                logger.info("reversibility diagnostics | %s", diagnostics)

    except torch.cuda.OutOfMemoryError as exc:
        status, error = "oom", str(exc).splitlines()[0]
        # Drop the step's tensors (graph, batch) before the next sweep point.
        loss = accuracy = x = y = None  # noqa: F841
        if logger is not None:
            logger.warning("CUDA OOM: %s", error)
    finally:
        wall_clock_s = time.perf_counter() - start
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / MB if is_cuda else 0.0

    # ---------------------------------------------------------------- summary
    steps_done = len(history["step"])
    skip = max(3, int(cfg.warmup_skip_frac * steps_done)) if steps_done > 6 else min(1, steps_done)
    warm = slice(skip, None)

    def warm_values(key: str) -> list[float]:
        return [v for v in history[key][warm] if v is not None]

    tps = warm_values("tokens_per_sec")
    mean_tps = _mean(tps)
    peak_step_mem = max(history["step_peak_memory_mb"], default=0.0)
    mean_activation = _mean(warm_values("activation_saved_mb"))
    mean_fwd, mean_bwd = _mean(warm_values("forward_ms")), _mean(warm_values("backward_ms"))
    val_curve = list(zip(evals["tokens"], evals["val_loss"]))
    tail = history["loss"][-max(1, steps_done // 20):]

    def tokens_to(threshold: float) -> int | None:
        return next((t for t, v in val_curve if v <= threshold), None)

    summary = {
        "label": cfg.label,
        "variant": cfg.variant,
        "status": status,
        "error": error,
        "batch_size": cfg.batch_size,
        "block_size": cfg.block_size,
        "tokens_per_step": cfg.tokens_per_step,
        "total_steps": cfg.total_steps,
        "steps_completed": steps_done,
        "tokens_seen": steps_done * cfg.tokens_per_step,
        "n_params": n_params,
        "n_params_non_embedding": n_params_non_embedding,
        "peak_lr": cfg.peak_lr,
        "warmup_steps": cfg.warmup_steps,
        # --- quality
        "final_val_loss": final_eval.get("val", {}).get("loss"),
        "final_val_perplexity": final_eval.get("val", {}).get("perplexity"),
        "final_val_accuracy": final_eval.get("val", {}).get("accuracy"),
        "final_val_tokens": final_eval.get("val", {}).get("tokens"),
        "final_train_loss": final_eval.get("train", {}).get("loss"),
        "final_train_accuracy": final_eval.get("train", {}).get("accuracy"),
        "generalization_gap": (
            final_eval["val"]["loss"] - final_eval["train"]["loss"] if final_eval else None
        ),
        "best_periodic_val_loss": min(evals["val_loss"], default=None),
        "train_loss_last5pct_mean": _mean(tail),
        "tokens_to_val_loss_7": tokens_to(7.0),
        "tokens_to_val_loss_6": tokens_to(6.0),
        "tokens_to_val_loss_5": tokens_to(5.0),
        "tokens_to_val_loss_4.5": tokens_to(4.5),
        # --- speed
        "mean_tokens_per_sec": mean_tps,
        "median_tokens_per_sec": _percentile(tps, 50),
        "p10_tokens_per_sec": _percentile(tps, 10),
        "p90_tokens_per_sec": _percentile(tps, 90),
        "mean_step_time_ms": _mean(warm_values("step_time_ms")),
        "mean_data_ms": _mean(warm_values("data_ms")),
        "mean_forward_ms": mean_fwd,
        "mean_backward_ms": mean_bwd,
        "mean_optimizer_ms": _mean(warm_values("optimizer_ms")),
        "backward_to_forward_ratio": (mean_bwd / mean_fwd) if mean_fwd and mean_bwd else None,
        "wall_clock_s": wall_clock_s,
        "eval_time_s": eval_time_s,
        "train_time_s": wall_clock_s - eval_time_s,
        # --- compute
        "model_flops_per_token": flops["model"],
        "hardware_flops_per_token": flops["hardware"],
        "lm_head_flops_fraction": flops["lm_head_fraction"],
        "total_model_pflops": flops["model"] * steps_done * cfg.tokens_per_step / 1e15,
        "model_tflops_per_sec": (flops["model"] * mean_tps / 1e12) if mean_tps else None,
        "hardware_tflops_per_sec": (flops["hardware"] * mean_tps / 1e12) if mean_tps else None,
        "gpu_peak_tflops": peak_tflops,
        "mfu": (flops["model"] * mean_tps / 1e12 / peak_tflops) if mean_tps and peak_tflops else None,
        "hfu": (flops["hardware"] * mean_tps / 1e12 / peak_tflops) if mean_tps and peak_tflops else None,
        # --- memory
        "peak_memory_mb": peak_memory_mb,
        "peak_step_memory_mb": peak_step_mem,
        "peak_memory_reserved_mb": max(history["memory_reserved_mb"], default=0.0),
        "param_memory_mb": param_memory_mb,
        "optimizer_state_mb": memory_info.get("optimizer_state_mb"),
        "static_memory_mb": memory_info.get("static_after_first_step_mb"),
        "activation_saved_mb": mean_activation,
        "loss_buffer_mb": loss_buffer_mb(model_cfg),
        "activation_saved_mb_per_sample": (
            (mean_activation - loss_buffer_mb(model_cfg)) / cfg.batch_size if mean_activation else None
        ),
        "gpu_total_memory_mb": gpu_total_mb,
        "peak_memory_frac_of_gpu": (peak_step_mem / gpu_total_mb) if gpu_total_mb else None,
        # --- optimisation health
        "mean_grad_norm": _mean(history["grad_norm"]),
        "max_grad_norm": max(history["grad_norm"], default=None),
        "frac_steps_clipped": (
            float(np.mean([g > cfg.grad_clip for g in history["grad_norm"]])) if steps_done else None
        ),
        # --- reversible-only
        "reversibility": diagnostics or None,
        "aim_run_hash": logger.aim_hash if logger else None,
    }
    summary = _clean(summary)
    record = {
        "config": cfg.to_dict(),
        "model_config": asdict(model_cfg),
        "env": env,
        "summary": summary,
        "final_eval": final_eval,
        "evals": evals,
        "history": history,
    }
    record = _clean(record)
    (results_dir / f"{cfg.label}.json").write_text(
        json.dumps(record, indent=1, default=_json_default), encoding="utf-8"
    )
    with (results_dir / "runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"config": record["config"], "summary": summary, "evals": evals}, default=_json_default)
            + "\n"
        )
    if logger is not None:
        for key, value in summary.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value is not None:
                logger.track("summary", 0, {f"summary/{key}": value}, "summary")
        logger.info("summary | %s", json.dumps(summary, default=_json_default))
        logger.finish(summary)

    # Free GPU memory so the next run in the same process starts clean.
    del model, optimizer
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()
    return record


def load_record(label: str, results_dir: str | Path = "results") -> dict | None:
    path = Path(results_dir) / f"{label}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
