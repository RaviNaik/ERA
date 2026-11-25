# Deletion Tuning of Phi-mini-MoE-instruct on GSM8K (SFT): dt_phi_mini_sft_gsm8k_seed0_v4

## Training (official Deletion Tuning code, commit 94f0340)

- 400 steps, batch 4 x accum 1, lr 0.0005, sharpness 100.0 -> 1000.0, weight_reg 1e-05, init_sparsity 0.000488281, seed 0
- 2048 GSM8K training samples (pool 2048), 512 expert gate projections, 0 without a Wanda score
- deleted at step 0: 0.1837% of gate weights; at the end: 0.6108% (12,296,351 of 2,013,265,920)
- 7035 s training (12.8 s per step); peak GPU memory GiB {'cuda:0': 35.2, 'cuda:1': 31.0}
- saved mask: 23.2 MB (official packbits + zstd)

## Results

| | base | Deletion Tuning | change |
|---|---:|---:|---:|
| GSM8K accuracy, training format (%) | 33.50 | 78.50 | +45.00 |
| GSM8K accuracy, chat template (%) | 58.50 | 71.50 | +13.00 |
| replies with '####', training format (%) | 23.50 | 99.50 | +76.00 |
| hellaswag (%) | 70.00 | 70.00 | +0.00 |
| humaneval_instruct (%) | 14.00 | 12.00 | -2.00 |
| ifeval (%) | 67.11 | 63.16 | -3.95 |
| mmlu (%) | 69.54 | 69.75 | +0.21 |
| truthfulqa_mc2 (%) | 56.96 | 56.26 | -0.70 |
| winogrande (%) | 78.00 | 80.00 | +2.00 |
| prior-task mean (%) | 59.27 | 58.53 | -0.74 |
| team10 indic_v1 exact (%) | 52.27 | 52.27 | +0.00 |
| team10 safety_v1 exact (%) | 56.10 | 56.10 | +0.00 |
| forward KL per token (base -> masked, GSM8K) | 0 | 0.1412 | |

Forward KL is measured on the base model's own greedy replies to the evaluated GSM8K test questions.
Prior-task numbers: lm-evaluation-harness, zero-shot, chat template, bf16; limit 50 (0 = full benchmarks).
