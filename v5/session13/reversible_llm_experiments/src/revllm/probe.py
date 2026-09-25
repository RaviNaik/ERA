"""Memory / speed probes: how large a batch fits, and what one step costs.

A probe runs a few *real* training steps (forward, backward, fused AdamW),
so optimizer state and gradient buffers are included. It uses random token
ids: memory and speed do not depend on the token values.
"""

from __future__ import annotations

import gc
import time

import torch

from .model import GPTConfig, build_model
from .trainer import MB, TrainConfig, configure_optimizer, loss_buffer_mb

GIB = 1024**3


def _free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def probe_step(
    model_cfg: GPTConfig,
    batch_size: int,
    device: str,
    steps: int = 3,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Peak memory and median step time for `steps` training steps."""
    row = {"variant": model_cfg.variant, "batch_size": batch_size}
    model = optimizer = x = loss = None
    try:
        _free()
        torch.cuda.reset_peak_memory_stats(device)
        base = torch.cuda.memory_allocated(device)
        model = build_model(model_cfg, device=device)
        optimizer = configure_optimizer(model, TrainConfig(device=device))
        x = torch.randint(0, model_cfg.vocab_size, (batch_size, model_cfg.block_size), device=device)
        times, saved = [], []
        for _ in range(steps):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            pre = torch.cuda.memory_allocated(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                _, loss, _ = model(x, x)
            saved.append((torch.cuda.memory_allocated(device) - pre) / MB)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)
        step_s = sorted(times)[len(times) // 2]
        row |= {
            "fits": True,
            "peak_memory_mb": (torch.cuda.max_memory_allocated(device) - base) / MB,
            "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / MB,
            "activation_saved_mb": saved[-1],
            # Excludes the batch-independent fused-loss weight-grad buffer.
            "activation_saved_mb_per_sample": (saved[-1] - loss_buffer_mb(model_cfg)) / batch_size,
            "step_time_ms": step_s * 1000,
            "tokens_per_sec": batch_size * model_cfg.block_size / step_s,
        }
    except torch.cuda.OutOfMemoryError:
        row |= {"fits": False}
    finally:
        del model, optimizer, x, loss
        _free()
    return row


def find_max_batch(
    model_cfg: GPTConfig,
    device: str,
    start: int = 64,
    limit: int = 16384,
    granularity: int = 32,
    budget_frac: float = 1.0,
) -> tuple[int, list[dict]]:
    """Largest batch (to `granularity`) whose probed step fits.

    Doubles from `start` until OOM (or until peak memory exceeds `budget_frac`
    of total VRAM), then bisects. Returns (max_batch, probe rows).
    """
    total_mb = torch.cuda.get_device_properties(device).total_memory / MB
    rows: list[dict] = []

    def fits(batch: int) -> bool:
        row = probe_step(model_cfg, batch, device, steps=2)
        row["within_budget"] = row["fits"] and row["peak_memory_mb"] <= budget_frac * total_mb
        rows.append(row)
        return row["within_budget"]

    lo, hi = 0, None
    batch = start
    while batch <= limit:
        if fits(batch):
            lo, batch = batch, batch * 2
        else:
            hi = batch
            break
    if hi is None:
        return lo, rows
    while hi - lo > granularity:
        mid = (lo + hi) // 2 // granularity * granularity
        if mid <= lo:
            break
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return lo, rows
