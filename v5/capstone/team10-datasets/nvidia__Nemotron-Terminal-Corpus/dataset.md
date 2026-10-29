# nvidia/Nemotron-Terminal-Corpus

Source: https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus

## Summary

| Field | Value |
|---|---|
| Category | coding |
| Training stage | SFT (terminal) |
| Licence tag (Hub, read 29 Oct 2026) | cc-by-4.0 |
| Our decision | **use, check** |
| Gated | False |
| Last modified | 2026-02-27 |
| Downloads (30 d) | 22485 |
| Languages | en |
| Task categories | question-answering |
| Size category | 100K<n<1M |
| Total rows (all configs) | 366154 |

## Caveats and open checks

- Teacher model not stated on the card (Qwen3-Coder and Gemini appear only in comparison text).
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Benchmark overlap

- Risk: **Medium** (evidence: not stated)
- The card reports gains on Terminal-Bench 2.0 but does not state task overlap. Check if we report Terminal-Bench.
- Card-level check only, no text-overlap test run. See `../BENCHMARK_OVERLAP.md`.

## Configs and splits

- `dataset_adapters` / `train`
- `skill_based_easy` / `train`
- `skill_based_medium` / `train`
- `skill_based_mixed` / `train`

## Fields (from the sampled config)

- `conversations`: {"feature": {"content": {"dtype": "string", "_type": "Value"}, "role": {"dtype": "string", "_type": "Value"}}, "_type": 
- `agent`: {"dtype": "string", "_type": "Value"}
- `model`: {"dtype": "string", "_type": "Value"}
- `model_provider`: {"dtype": "string", "_type": "Value"}
- `date`: {"dtype": "string", "_type": "Value"}
- `task`: {"dtype": "string", "_type": "Value"}
- `episode`: {"dtype": "string", "_type": "Value"}
- `run_id`: {"dtype": "string", "_type": "Value"}
- `trial_name`: {"dtype": "string", "_type": "Value"}
- `enable_thinking`: {"dtype": "bool", "_type": "Value"}

## Sample

`sample_20.jsonl` holds the first 20 rows from `skill_based_easy/train`. Each line is one row; `_config` and `_split` say where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> # Terminal-Corpus: Large-Scale SFT Dataset for Terminal Agents
> 
> Terminal-Corpus is a large-scale Supervised Fine-Tuning (SFT) dataset designed to scale the terminal interaction capabilities of Large Language Models (LLMs). Developed by NVIDIA, this dataset was built using the **Terminal-Task-Gen** pipeline, which combines dataset adaptation with synthetic task generation across diverse domains.
> 
> ## 🚀 Key Results & Performance
> 
> The high-quality trajectories in Terminal-Corpus enable models of various sizes to achieve performance that rivals or exceeds much larger frontier models on the **Terminal-Bench 2.0** benchmark.
> 
> ### 1. Overall Performance Comparison
> Training on Terminal-Corpus yields substantial gains across the Qwen3 model family:
> 
> | Model Size | Base Model (Qwen3) Accuracy | Nemotron-Terminal Accuracy | Improvement |
> | :--- | :---: | :---: | :---: |
> | **8B** | 2.5% ± 0.5 | **13.0% ± 2.2** | ~5.2x |
> | **14B** | 4.0% ± 1.3 | **20.2% ± 2.7** | ~5.0x |
> | **32B** | 3.4% ± 1.6 | **27.4% ± 2.4** | ~8.0x |
> 
> The **Nemotron-Terminal-32B** (27.4%) outperforms the 480B-parameter **Qwen3-Coder** (23.9%) and **Gemini 2.5 Flash** (16.9%). **Nemotron-Terminal-14B** (20.2%) achieves higher accuracy than the 120B **GPT-OSS (high)** (18.7%).
> 
> ### 2. Domain-Specific Breakthroughs
> The dataset unlocks functional utility in complex domains where base models previously showed near-zero capability:
> 
> | Category | Qwen3-32B (Base) | Nemotron-Terminal-32B |
> | :--- | :---: | :---: |
> | **Data Querying** | 0.0% | **60.0%** |
> | **Model Training** | 0.0% | **50.0%** |
> | **Data Processing** | 5.0% | **50.0%** |
> | **Debugging** | 0.0% | **33.3%** |
> | **Software Engineering** | 5.0% | **31.7%** |
> 
> ## 📂 Dataset Composition
> 
> The released dataset contains approximately 366k high-quality execution tr
> ...[card continues at the source link]
