# Deletion Tuning of Phi-mini-MoE-instruct on GSM8K (SFT): dt_phi_mini_sft_gsm8k_seed0_v4

## Training (official Deletion Tuning code, commit 94f0340)

- 400 steps, batch 4 x accum 1, lr 0.0005, sharpness 100.0 -> 1000.0, weight_reg 1e-05, init_sparsity 0.000488281, seed 0
- 2048 GSM8K training samples (pool 2048), 512 expert gate projections, 0 without a Wanda score
- deleted at step 0: 0.1837% of gate weights; at the end: 0.6108% (12,296,351 of 2,013,265,920)
- 7035 s training (12.8 s per step); peak GPU memory GiB {'cuda:0': 35.2, 'cuda:1': 31.0}
- saved mask: 23.2 MB (official packbits + zstd)

## Training: LoRA

- 400 steps, batch 4 x accum 1, lr 0.0002, seed 0, r 16, alpha 32.0, targets w1
- 41,418,752 trainable parameters (0.5416% of 7,647,632,704); 4466 s (11.16 s per step); peak GPU memory GiB {'cuda:0': 16.5}

## Training: Full FT

- 400 steps, batch 4 x accum 1, lr 1e-05, seed 0, scope all, bf16 weights, stochastic rounding
- 7,647,632,704 trainable parameters (100.0000% of 7,647,632,704); 7045 s (17.61 s per step); peak GPU memory GiB {'cuda:0': 32.7, 'cuda:1': 32.8}

## Results

|  | base | Deletion Tuning | LoRA | Full FT | change (Deletion Tuning) | change (LoRA) | change (Full FT) |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K accuracy, chat template (%) | 55.95 | 67.10 | 74.98 | 54.36 | +11.14 | +19.03 | -1.59 |
| hellaswag (%) | 63.00 | 62.50 | 62.00 | 63.00 | -0.50 | -1.00 | +0.00 |
| mmlu (%) | 70.48 | 70.21 | 69.99 | 68.74 | -0.27 | -0.49 | -1.74 |
| truthfulqa_mc2 (%) | 52.21 | 52.63 | 52.26 | 50.46 | +0.41 | +0.04 | -1.75 |
| winogrande (%) | 71.50 | 72.50 | 74.00 | 72.50 | +1.00 | +2.50 | +1.00 |
| prior-task mean (%) | 64.30 | 64.46 | 64.56 | 63.68 | +0.16 | +0.26 | -0.62 |
| forward KL per token (base -> tuned, GSM8K) | 0 | n/a | n/a | n/a |  |  |  |

Forward KL is measured on the base model's own greedy replies to the evaluated GSM8K test questions.
Prior-task numbers: lm-evaluation-harness, zero-shot, chat template, bf16; limit 200 (0 = full benchmarks).

## Paired comparison with the base model (same questions)

| variant, format | n | base % | variant % | change (points) | 95% interval | base-only / variant-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Deletion Tuning, chat template | 1319 | 56.0 | 67.1 | +11.1 | [+8.8, +13.5] | 62 / 209 | 8.5e-20 |
| LoRA, chat template | 1319 | 56.0 | 75.0 | +19.0 | [+16.5, +21.6] | 49 / 300 | 3.9e-45 |
| Full FT, chat template | 1319 | 56.0 | 54.4 | -1.6 | [-4.6, +1.5] | 226 / 205 | 0.34 |

Interval: bootstrap over questions, 10000 resamples, seed 0. 'base-only / variant-only' are the questions only the base model / only the variant answered correctly; McNemar p is the exact two-sided test on those two counts.
