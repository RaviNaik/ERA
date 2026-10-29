# microsoft/rStar-Coder

Source: https://huggingface.co/datasets/microsoft/rStar-Coder

## Summary

| Field | Value |
|---|---|
| Category | coding |
| Training stage | SFT + RL |
| Licence tag (Hub, read 29 Oct 2026) | cc-by-4.0 |
| Our decision | **use** |
| Gated | False |
| Last modified | 2025-07-20 |
| Downloads (30 d) | 7902 |
| Languages | not stated |
| Task categories | not stated |
| Size category | not stated |
| Total rows (all configs) | 1392267 |

## Caveats and open checks

- Has separate SFT and RL configs; the sample comes from the first config. The card names no generator in its licence section (GPT-4o and Claude appear only in benchmark tables); the paper may describe GPT-4o-based problem synthesis: verify in arXiv 2505.21297 before relying on it.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Benchmark overlap

- Risk: **Medium** (evidence: not stated; the paper may say more (recollection (unverified)))
- Seeds come from competitive-programming sites. The card quotes LiveCodeBench, HumanEval and MBPP results but has no decontamination note. Check against LiveCodeBench and CodeElo.
- Card-level check only, no text-overlap test run. See `../BENCHMARK_OVERLAP.md`.

## Configs and splits

- `seed_sft` / `train`
- `seed_testcase` / `train`
- `synthetic_rl` / `train`
- `synthetic_rl_testcase` / `train`
- `synthetic_sft` / `train`

## Fields (from the sampled config)

- `question_id`: {"dtype": "string", "_type": "Value"}
- `question`: {"dtype": "string", "_type": "Value"}
- `starter_code`: {"dtype": "string", "_type": "Value"}
- `response`: {"dtype": "string", "_type": "Value"}
- `code`: {"dtype": "string", "_type": "Value"}
- `verified`: {"dtype": "bool", "_type": "Value"}
- `is_passed`: {"dtype": "bool", "_type": "Value"}

## Sample

`sample_20.jsonl` holds the first 20 rows from `seed_sft/train`. Each line is one row; `_config` and `_split` say where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> # rStar-Coder Dataset
> 
> 
> 
> [Project GitHub](https://github.com/microsoft/rStar) | [Paper](https://arxiv.org/abs/2505.21297)
> 
> 
> 
> ## Dataset Description
> rStar-Coder is a large-scale competitive code problem dataset containing 418K programming problems, 580K long-reasoning solutions, and rich test cases of varying difficulty levels. This dataset aims to enhance code reasoning capabilities in large language models, particularly in handling competitive code problems.
> 
> Experiments on Qwen models (1.5B-14B) across various  code reasoning benchmarks demonstrate the superiority of rStar-Coder dataset, achieving leading performance comparable to frontier reasoning LLMs with much smaller model sizes. On LiveCodeBench, rStar-Coder improves Qwen2.5-7B from 17.4% to an impressive 57.3%,  and Qwen2.5-14B from 23.3% to 62.5%, surpassing o3-mini (low) by 3.1%. On the more challenging USA Computing Olympiad, our 7B model achieves an average pass@1 accuracy of 16.15%, outperforming the frontier-level QWQ-32B.  
> 
> 
> | Model                                     |  LiveCodeBench   |    HumanEval    |    HumanEval+    |      MBPP       |      MBPP+      |
> | ----------------------------------------- | :--------------: | :-------------: | :--------------: | :-------------: | :-------------: |
> | GPT-4o                                    |       30.0       |      92.7       |       87.2       |      87.6       |      72.2       |
> | Claude3.5-Sonnet                          |       32.0       |      92.1       |        86        |       91        |      74.6       |
> | OpenAI o3-mini-2025-01-31 (low)           |       59.4       |        -        |        -         |        -        |        -        |
> | QWQ-32B                                   |       63.4       |      95.6       |       89.8       |
> ...[card continues at the source link]
