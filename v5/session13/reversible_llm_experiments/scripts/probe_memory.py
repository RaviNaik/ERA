"""One-off script: probe batch-size scaling for baseline/euler/midpoint variants."""

from __future__ import annotations

import time

import torch

from revllm.model import GPTConfig, build_model

DEVICE = "cuda:1"
torch.cuda.set_device(DEVICE)


def probe(variant: str, batch_sizes: list[int], block_size: int = 512) -> list[tuple]:
    rows = []
    for bs in batch_sizes:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(DEVICE)
            cfg = GPTConfig(
                vocab_size=50257,
                block_size=block_size,
                n_layer=9,
                n_head=8,
                n_embd=256,
                variant=variant,
            )
            model = build_model(cfg, device=DEVICE)
            x = torch.randint(0, cfg.vocab_size, (bs, block_size), device=DEVICE)
            y = torch.randint(0, cfg.vocab_size, (bs, block_size), device=DEVICE)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss, _ = model(x, y)
            loss.backward()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
            rows.append(
                (variant, bs, True, round(peak, 1), round(bs * block_size / dt, 1))
            )
            del model, x, y, loss
        except RuntimeError as exc:
            rows.append((variant, bs, False, str(exc).splitlines()[0], None))
            torch.cuda.empty_cache()
            break
    return rows


if __name__ == "__main__":
    candidates = [64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
    for variant in ["baseline", "euler", "midpoint"]:
        for row in probe(variant, candidates):
            print(row)
