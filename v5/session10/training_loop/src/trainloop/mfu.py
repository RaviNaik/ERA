"""Compute your own MFU, honestly.

MFU (Model FLOPs Utilisation) = the FLOP/s the model actually needed, divided by
the FLOP/s the accelerator can theoretically deliver.

    achieved FLOP/s = flops_per_token * tokens_per_second
    MFU             = achieved FLOP/s / peak FLOP/s

We time a real training loop (forward + backward + optimizer step), count the
tokens that went through it, and divide. `PEAK_TFLOPS` holds vendor dense
peak numbers -- override with S10_PEAK_TFLOPS if you know your card's better.
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass

import torch

from .data import CharDataset, get_batch

# Vendor dense peak throughput, TFLOP/s. bf16/fp16 tensor-core, no sparsity.
PEAK_TFLOPS = {
    "H100": 989.0,      # SXM, bf16
    "H200": 989.0,
    "A100": 312.0,
    "A6000": 155.0,     # RTX A6000, bf16
    "4090": 165.0,      # RTX 4090, bf16 (non-sparse)
    "L4": 121.0,
    "L40": 181.0,
    "V100": 125.0,      # fp16
    "T4": 65.0,
    "RTX 500 Ada": 43.0,  # ~43 TFLOP/s bf16 tensor (laptop)
    "CPU": 0.5,         # order-of-magnitude placeholder
}


def guess_peak_tflops(device_name: str) -> tuple[float, str]:
    override = os.environ.get("S10_PEAK_TFLOPS")
    if override:
        return float(override), f"S10_PEAK_TFLOPS={override}"
    for key, val in PEAK_TFLOPS.items():
        if key.lower() in device_name.lower():
            return val, f"matched '{key}' in device name"
    return PEAK_TFLOPS["CPU"], "no match; using CPU placeholder (set S10_PEAK_TFLOPS)"


@dataclass
class MFUConfig:
    batch_size: int = 16
    seq_len: int = 256
    warmup: int = 5
    measure: int = 20
    lr: float = 3e-3
    dtype: str = "bf16"  # "bf16" | "fp16" | "fp32"


@dataclass
class MFUResult:
    device: str
    peak_tflops: float
    peak_source: str
    flops_per_token: float
    tokens_per_sec: float
    achieved_tflops: float
    mfu: float
    step_ms: float
    dtype: str

    def render(self) -> str:
        return (
            f"device                : {self.device}\n"
            f"dtype                  : {self.dtype}\n"
            f"flops/token (6N + attn): {self.flops_per_token:,.0f}\n"
            f"mean step time         : {self.step_ms:.2f} ms\n"
            f"tokens/sec             : {self.tokens_per_sec:,.0f}\n"
            f"achieved               : {self.achieved_tflops:,.2f} TFLOP/s\n"
            f"peak ({self.peak_source}) : {self.peak_tflops:,.1f} TFLOP/s\n"
            f"MFU                    : {self.mfu * 100:.2f} %\n"
            f"distance to 40%        : {max(0.0, 0.40 - self.mfu) * 100:.2f} points"
        )


def measure_mfu(model, ds: CharDataset, cfg: MFUConfig, device) -> MFUResult:
    from .utils import device_name

    model = copy.deepcopy(model).to(device).train()  # don't mutate the caller's model
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(11)

    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.dtype]
    use_autocast = device.type == "cuda" and cfg.dtype != "fp32"
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.dtype == "fp16" and device.type == "cuda"))

    def one_step():
        x, y = get_batch(ds, "train", cfg.batch_size, cfg.seq_len, device, generator=gen)
        opt.zero_grad(set_to_none=True)
        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                _, loss = model(x, targets=y)
        else:
            _, loss = model(x, targets=y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(cfg.warmup):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(cfg.measure):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tokens = cfg.measure * cfg.batch_size * cfg.seq_len
    tok_per_sec = tokens / elapsed
    fpt = model.flops_per_token()
    achieved = fpt * tok_per_sec
    dev = device_name(device)
    peak, source = guess_peak_tflops(dev if device.type == "cuda" else "CPU")

    return MFUResult(
        device=dev,
        peak_tflops=peak,
        peak_source=source,
        flops_per_token=fpt,
        tokens_per_sec=tok_per_sec,
        achieved_tflops=achieved / 1e12,
        mfu=(achieved / 1e12) / peak,
        step_ms=1000 * elapsed / cfg.measure,
        dtype=cfg.dtype,
    )
