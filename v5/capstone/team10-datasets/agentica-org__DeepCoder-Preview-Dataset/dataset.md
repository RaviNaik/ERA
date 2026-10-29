# agentica-org/DeepCoder-Preview-Dataset

Source: https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset

## Summary

| Field | Value |
|---|---|
| Category | coding |
| Training stage | RL |
| Licence tag (Hub, read 29 Oct 2026) | mit |
| Our decision | **use** |
| Gated | False |
| Last modified | 2025-04-09 |
| Downloads (30 d) | 3793 |
| Languages | en |
| Task categories | not stated |
| Size category | 10K<n<100K |
| Total rows (all configs) | 24974 |

## Caveats and open checks

- About 24k problems with tests; configs are codeforces, lcbv5, primeintellect, taco.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Benchmark overlap

- Risk: **High** (evidence: card)
- The training set includes 600 LiveCodeBench v5 problems submitted 1 May 2023 to 31 Jul 2024. The test data is LiveCodeBench v5 (1 Aug 2024 to 1 Feb 2025) and Codeforces problems from Qwen/CodeElo. Use `train` only; treat the `test` configs as eval; do not report LiveCodeBench results before 1 Aug 2024.
- Card-level check only, no text-overlap test run. See `../BENCHMARK_OVERLAP.md`.

## Configs and splits

- `codeforces` / `test`
- `lcbv5` / `train`
- `lcbv5` / `test`
- `primeintellect` / `train`
- `taco` / `train`

## Fields (from the sampled config)

- `problem`: {"dtype": "string", "_type": "Value"}
- `solutions`: {"feature": {"dtype": "string", "_type": "Value"}, "_type": "List"}
- `tests`: {"dtype": "string", "_type": "Value"}

## Sample

`sample_20.jsonl` holds the first 20 rows from `primeintellect/train`. Each line is one row; `_config` and `_split` say where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> ## Data
> 
> Our training dataset consists of 24K problems paired with their test cases:
> 
> - 7.5K **TACO Verified** problems.
> - 16K verified coding problems from **PrimeIntellect’s SYNTHETIC-1.**
> - 600 **LiveCodeBench (v5)** problems submitted between **May 1, 2023** and **July 31, 2024.**
> 
> Our test dataset consists of:
> - **LiveCodeBench (v5)** problems between **August 1, 2024** and **February 1, 2025**.
> - **Codeforces** problems from `Qwen/CodeElo`.
> 
> ## Format
> 
> Each row in the dataset contains:
> 
> - **problem**: The coding problem, usually extracted from competitive coding websites.
> - **tests**: The test cases corresponding to the problem. We've ensured that all problems are fully verifiable and have >= 5 test cases.
> 
> We note different datasets have different keys beyond problems and tests.
> 
> ## Citation
> 
> ```bibtex
> @misc{deepcoder2025,
>   title={DeepCoder: A Fully Open-Source 14B Coder at O3-mini Level},
>   author={Michael Luo, Sijun Tan, Roy Huang, Ameen Patel, Alpay Ariyak, Qingyang Wu, Xiaoxiang Shi, Rachel Xin, Colin Cai, Maurice Weber, Ce Zhang, Li Erran Li, Raluca Ada Popa, Ion Stoica},
>   howpublished={\url{https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51}},
>   note={Notion Blog},
>   year={2025}
> }
> ```
