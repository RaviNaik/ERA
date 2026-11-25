# Deletion Tuning of Phi-mini-MoE-instruct on GSM8K (SFT): dt_phi_smoke_baselines_smoke

## Training: LoRA

- 5 steps, batch 4 x accum 1, lr 0.0002, seed 0, r 16, alpha 32.0, targets w1
- 41,418,752 trainable parameters (0.5416% of 7,647,632,704); 89 s (17.73 s per step); peak GPU memory GiB {'cuda:0': 7.9, 'cuda:1': 8.2}

## Training: Full FT

- 5 steps, batch 4 x accum 1, lr 1e-05, seed 0, scope gate, bf16 weights, stochastic rounding
- 2,013,265,920 trainable parameters (26.3253% of 7,647,632,704); 84 s (16.78 s per step); peak GPU memory GiB {'cuda:0': 13.2, 'cuda:1': 13.2}

## Results

|  | base | LoRA | Full FT | change (LoRA) | change (Full FT) |
|---|---:|---:|---:|---:|---:|
| GSM8K accuracy, training format (%) | 18.75 | 18.75 | 25.00 | +0.00 | +6.25 |
| GSM8K accuracy, chat template (%) | 12.50 | 12.50 | 12.50 | +0.00 | +0.00 |
| replies with '####', training format (%) | 12.50 | 18.75 | 37.50 | +6.25 | +25.00 |
| hellaswag (%) | 75.00 | 75.00 | 50.00 | +0.00 | -25.00 |
| humaneval_instruct (%) | 25.00 | 25.00 | 25.00 | +0.00 | +0.00 |
| winogrande (%) | 50.00 | 50.00 | 75.00 | +0.00 | +25.00 |
| prior-task mean (%) | 50.00 | 50.00 | 50.00 | +0.00 | +0.00 |
| team10 indic_v1 exact (%) | 34.09 | 31.82 | 31.82 | -2.27 | -2.27 |
| team10 safety_v1 exact (%) | 36.59 | 39.02 | 39.02 | +2.44 | +2.44 |
| forward KL per token (base -> tuned, GSM8K) | 0 | 0.0002 | 0.0193 |  |  |

Forward KL is measured on the base model's own greedy replies to the evaluated GSM8K test questions.
Prior-task numbers: lm-evaluation-harness, zero-shot, chat template, bf16; limit 4 (0 = full benchmarks).

## Paired comparison with the base model (same questions)

| variant, format | n | base % | variant % | change (points) | 95% interval | base-only / variant-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoRA, training format | 16 | 18.8 | 18.8 | +0.0 | [-18.8, +18.8] | 1 / 1 | 1 |
| LoRA, chat template | 16 | 12.5 | 12.5 | +0.0 | [+0.0, +0.0] | 0 / 0 | 1 |
| Full FT, training format | 16 | 18.8 | 25.0 | +6.2 | [-12.5, +25.0] | 1 / 2 | 1 |
| Full FT, chat template | 16 | 12.5 | 12.5 | +0.0 | [+0.0, +0.0] | 0 / 0 | 1 |

Interval: bootstrap over questions, 10000 resamples, seed 0. 'base-only / variant-only' are the questions only the base model / only the variant answered correctly; McNemar p is the exact two-sided test on those two counts.
