"""Validation protocol item 4 (research note Sec. 10.4 / Sec. 8.3), Design B
(HRR binding) only: vary `D` and the number of bytes bound per token, and
directly measure unbinding accuracy, to check the interference-scaling
prediction (`O(sqrt(n/D))` relative noise) against a real implementation.

For each (D, n) setting: build `n` random byte values bound at `n` distinct
positions and summed into one code (`HRRBindingEmbedding.bind_sum`), then for
each position try to recover the bound byte value by unbinding and taking
the nearest neighbour (by cosine similarity) among all 256 `c_wave` vectors.
Top-1 retrieval accuracy across many trials is the empirical crosstalk
measure; we also report the residual's relative magnitude for direct
comparison to Plate's O(1/sqrt(D)) per-pair prediction.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from fourier_embeddings.model.codecs import HRRBindingEmbedding
from fourier_embeddings.utils.logging_setup import setup_logging

logger = logging.getLogger("fourier_embeddings.analysis.crosstalk")


@torch.no_grad()
def measure_crosstalk(D: int, n_bound: int, pos_dim: int, n_trials: int,
                       seed: int) -> dict:
    codec = HRRBindingEmbedding(vocab_size=1, d_model=1, id_to_bytes=[b""], D=D, pos_dim=pos_dim)
    gen = torch.Generator().manual_seed(seed)

    correct = 0
    total = 0
    residual_norms = []
    signal_norms = []

    for _ in range(n_trials):
        positions = torch.randperm(pos_dim, generator=gen)[:n_bound]
        byte_values = torch.randint(0, 256, (n_bound,), generator=gen)

        byte_ids = byte_values.unsqueeze(0)  # [1, n_bound] -- "positions" dim reused as bound-item dim
        valid_mask = torch.ones_like(byte_ids, dtype=torch.bool)
        length = torch.tensor([n_bound])

        # bind_sum expects byte_ids indexed along a "pos_dim"-sized axis using
        # torch.arange(P); to bind at the *sampled* positions (not 0..n_bound-1)
        # we build the summed code manually here using the same FFT machinery.
        c_spec = torch.fft.rfft(codec.c_wave, dim=-1)  # [256, D//2+1]
        p_spec = torch.fft.rfft(codec.pos_wave, dim=-1)  # [pos_dim, D//2+1]
        weight = 1.0 / (n_bound ** 0.5)
        bound_spec = c_spec[byte_values] * p_spec[positions] * weight
        summed_spec = bound_spec.sum(dim=0)
        code = torch.fft.irfft(summed_spec, n=D)

        for k in range(n_bound):
            recovered = codec.unbind(code, positions[k].item())
            sims = torch.nn.functional.cosine_similarity(
                recovered.unsqueeze(0), codec.c_wave, dim=-1
            )
            pred = sims.argmax().item()
            correct += int(pred == byte_values[k].item())
            total += 1

            true_vec = codec.c_wave[byte_values[k]] * weight
            residual = recovered - true_vec
            residual_norms.append(residual.norm().item())
            signal_norms.append(true_vec.norm().item())

    residual_norms = torch.tensor(residual_norms)
    signal_norms = torch.tensor(signal_norms)
    predicted_relative_noise = (max(n_bound - 1, 0) / D) ** 0.5

    return {
        "D": D,
        "n_bound": n_bound,
        "n_trials": n_trials,
        "top1_accuracy": correct / max(total, 1),
        "mean_residual_norm": residual_norms.mean().item(),
        "mean_signal_norm": signal_norms.mean().item(),
        "measured_relative_noise": (residual_norms.mean() / signal_norms.mean()).item(),
        "predicted_relative_noise_O(sqrt((n-1)/D))": predicted_relative_noise,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--D-values", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    ap.add_argument("--n-bound-values", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--pos-dim", type=int, default=32)
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/crosstalk.json")
    ap.add_argument("--log-file", default="logs/crosstalk.log")
    args = ap.parse_args()

    setup_logging(args.log_file, name="fourier_embeddings.analysis.crosstalk")

    results = []
    for D in args.D_values:
        for n_bound in args.n_bound_values:
            if n_bound > args.pos_dim:
                continue
            r = measure_crosstalk(D, n_bound, args.pos_dim, args.n_trials, args.seed)
            logger.info(f"D={D:5d} n_bound={n_bound:3d} -> top1_acc={r['top1_accuracy']:.3f} "
                        f"measured_noise={r['measured_relative_noise']:.3f} "
                        f"predicted_noise={r['predicted_relative_noise_O(sqrt((n-1)/D))']:.3f}")
            results.append(r)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
