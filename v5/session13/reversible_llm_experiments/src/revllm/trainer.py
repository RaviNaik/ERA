"""Training loop with Aim logging for baseline and reversible LLM runs."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .data import TokenDataset
from .model import GPTConfig, build_model


@dataclass
class TrainConfig:
    variant: str = "baseline"
    batch_size: int = 16
    block_size: int = 512
    token_budget: int = 50_000_000
    eval_interval: int = 200
    eval_iters: int = 25
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    n_layer: int = 9
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = True
    seed: int = 1337
    device: str = "cuda"
    compile_model: bool = False
    aim_repo: str | None = "."
    experiment: str = "session13_reversible_llm"
    label: str = "run"

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.block_size

    @property
    def total_steps(self) -> int:
        return math.ceil(self.token_budget / self.tokens_per_step)


def configure_optimizer(
    model: torch.nn.Module, cfg: TrainConfig
) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
    )


def learning_rate(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    if step >= cfg.total_steps:
        return cfg.min_lr
    ratio = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module, data: TokenDataset, cfg: TrainConfig, gen: torch.Generator
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        accuracies = torch.zeros(cfg.eval_iters)
        for idx in range(cfg.eval_iters):
            x, y = data.get_batch(
                split, cfg.batch_size, cfg.block_size, cfg.device, gen
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=torch.device(cfg.device).type == "cuda",
            ):
                _, loss, accuracy = model(x, y)
            losses[idx] = float(loss.item())
            accuracies[idx] = float(accuracy.item())
        out[f"{split}_loss"] = float(losses.mean().item())
        out[f"{split}_accuracy"] = float(accuracies.mean().item())
    model.train()
    return out


def train(
    data: TokenDataset,
    cfg: TrainConfig,
    results_dir: str | Path = "results",
    progress: bool = True,
) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    cfg.device = device
    is_cuda = torch.device(device).type == "cuda"

    model_cfg = GPTConfig(
        vocab_size=data.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        bias=cfg.bias,
        variant=cfg.variant,
    )
    model = build_model(model_cfg, device=device, compile_model=cfg.compile_model)
    optimizer = configure_optimizer(model, cfg)

    run = None
    hparams = asdict(cfg) | {
        "n_params": sum(param.numel() for param in model.parameters())
    }
    if cfg.aim_repo is not None:
        from aim import Run

        run = Run(repo=cfg.aim_repo, experiment=cfg.experiment)
        run.name = cfg.label
        run["hparams"] = hparams

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    history = {
        "step": [],
        "train_loss_step": [],
        "train_accuracy_step": [],
        "lr": [],
        "grad_norm": [],
        "tokens_per_sec": [],
        "step_time_ms": [],
        "memory_allocated_mb": [],
        "memory_reserved_mb": [],
        "eval_step": [],
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }
    start = time.perf_counter()
    last_time = start
    model.train()
    iterator = range(cfg.total_steps)
    if progress:
        iterator = tqdm(iterator, desc=cfg.label)
    for step in iterator:
        lr = learning_rate(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = data.get_batch("train", cfg.batch_size, cfg.block_size, device, gen)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=is_cuda,
        ):
            _, loss, accuracy = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        )
        optimizer.step()

        if is_cuda:
            torch.cuda.synchronize(device)
        now = time.perf_counter()
        step_time_s = max(now - last_time, 1e-9)
        tokens_per_sec = cfg.tokens_per_step / step_time_s
        last_time = now
        mem_allocated_mb = (
            torch.cuda.memory_allocated(device) / 1024**2 if is_cuda else 0.0
        )
        mem_reserved_mb = (
            torch.cuda.memory_reserved(device) / 1024**2 if is_cuda else 0.0
        )
        history["step"].append(step)
        history["train_loss_step"].append(float(loss.item()))
        history["train_accuracy_step"].append(float(accuracy.item()))
        history["lr"].append(lr)
        history["grad_norm"].append(grad_norm)
        history["tokens_per_sec"].append(tokens_per_sec)
        history["step_time_ms"].append(step_time_s * 1000.0)
        history["memory_allocated_mb"].append(mem_allocated_mb)
        history["memory_reserved_mb"].append(mem_reserved_mb)
        if run is not None:
            run.track(
                float(loss.item()),
                name="loss",
                step=step,
                context={"subset": "train_step"},
            )
            run.track(
                float(accuracy.item()),
                name="accuracy",
                step=step,
                context={"subset": "train_step"},
            )
            run.track(lr, name="lr", step=step)
            run.track(grad_norm, name="grad_norm", step=step)
            run.track(tokens_per_sec, name="tokens_per_sec", step=step)
            run.track(step_time_s * 1000.0, name="step_time_ms", step=step)
            run.track(mem_allocated_mb, name="memory_allocated_mb", step=step)
            run.track(mem_reserved_mb, name="memory_reserved_mb", step=step)

        if step % cfg.eval_interval == 0 or step == cfg.total_steps - 1:
            metrics = estimate_loss(model, data, cfg, gen)
            history["eval_step"].append(step)
            history["train_loss"].append(metrics["train_loss"])
            history["val_loss"].append(metrics["val_loss"])
            history["train_accuracy"].append(metrics["train_accuracy"])
            history["val_accuracy"].append(metrics["val_accuracy"])
            if run is not None:
                run.track(
                    metrics["train_loss"],
                    name="loss",
                    step=step,
                    context={"subset": "train"},
                )
                run.track(
                    metrics["val_loss"],
                    name="loss",
                    step=step,
                    context={"subset": "val"},
                )
                run.track(
                    metrics["train_accuracy"],
                    name="accuracy",
                    step=step,
                    context={"subset": "train"},
                )
                run.track(
                    metrics["val_accuracy"],
                    name="accuracy",
                    step=step,
                    context={"subset": "val"},
                )
            last_time = time.perf_counter()

    elapsed = time.perf_counter() - start
    peak_memory_mb = 0.0
    peak_memory_reserved_mb = 0.0
    if is_cuda:
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_memory_reserved_mb = torch.cuda.max_memory_reserved(device) / 1024**2
    warm_tokens_per_sec = history["tokens_per_sec"][
        max(1, len(history["tokens_per_sec"]) // 10) :
    ]
    warm_step_time_ms = history["step_time_ms"][
        max(1, len(history["step_time_ms"]) // 10) :
    ]
    summary = {
        "label": cfg.label,
        "variant": cfg.variant,
        "batch_size": cfg.batch_size,
        "tokens_seen": cfg.total_steps * cfg.tokens_per_step,
        "total_steps": cfg.total_steps,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_train_accuracy": history["train_accuracy"][-1],
        "final_val_accuracy": history["val_accuracy"][-1],
        "mean_tokens_per_sec": float(np.mean(warm_tokens_per_sec)),
        "mean_step_time_ms": float(np.mean(warm_step_time_ms)),
        "peak_memory_mb": peak_memory_mb,
        "peak_memory_reserved_mb": peak_memory_reserved_mb,
        "wall_clock_s": elapsed,
        "n_params": hparams["n_params"],
    }
    if run is not None:
        run["summary"] = summary
        for key, value in summary.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                run.track(float(value), name=f"summary/{key}", step=0)
        run.close()

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"hparams": hparams, "summary": summary, "curve": history})
            + "\n"
        )
    return {"summary": summary, "history": history, "config": hparams}
