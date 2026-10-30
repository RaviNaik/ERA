# nvidia/Nemotron-Math-v2

Source: https://huggingface.co/datasets/nvidia/Nemotron-Math-v2

## Summary

| Field | Value |
|---|---|
| Category | math |
| Training stage | SFT |
| Licence tag (Hub, read 29 Oct 2026) | cc-by-4.0, cc-by-sa-4.0 |
| Our decision | **allowed (CC-BY-SA, attribution needed)** (TSAI update, 30 Oct). Generator: gpt-oss-120b (open weights, Apache-2.0), confirmed allowed by TSAI on 30 Oct |
| Gated | False |
| Last modified | 2026-02-11 |
| Downloads (30 d) | 8773 |
| Languages | en |
| Task categories | text-generation |
| Size category | not stated |
| Total rows (all configs) | 7085839 |

## Caveats and open checks

- Mixed cc-by-4.0 and cc-by-sa-4.0: the share-alike part needs a team decision.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Benchmark overlap

- Risk: **Low** (evidence: card)
- The card says it is decontaminated to avoid overlap with public benchmarks.
- Card-level check only, no text-overlap test run. See `../BENCHMARK_OVERLAP.md`.

## Configs and splits

- `default` / `high_part00`
- `default` / `high_part01`
- `default` / `high_part02`
- `default` / `medium`
- `default` / `low`

## Fields (from the sampled config)

- `uuid`: {"dtype": "string", "_type": "Value"}
- `expected_answer`: {"dtype": "string", "_type": "Value"}
- `problem`: {"dtype": "string", "_type": "Value"}
- `original_expected_answer`: {"dtype": "string", "_type": "Value"}
- `changed_answer_to_majority`: {"dtype": "bool", "_type": "Value"}
- `data_source`: {"dtype": "string", "_type": "Value"}
- `messages`: {"feature": {"role": {"dtype": "string", "_type": "Value"}, "content": {"dtype": "string", "_type": "Value"}, "reasoning
- `used_in`: {"feature": {"dtype": "string", "_type": "Value"}, "_type": "List"}
- `metadata`: {"reason_low_with_tool": {"count": {"dtype": "int64", "_type": "Value"}, "pass": {"dtype": "int64", "_type": "Value"}, "
- `license`: {"dtype": "string", "_type": "Value"}
- `tools`: {"feature": {"type": {"dtype": "string", "_type": "Value"}, "function": {"name": {"dtype": "string", "_type": "Value"}, 
- `url`: {"dtype": "string", "_type": "Value"}
- `user_name`: {"dtype": "string", "_type": "Value"}
- `user_url`: {"dtype": "string", "_type": "Value"}

## Sample

`sample_20.jsonl` holds the first 20 rows from `default/high_part00`. Each line is one row; `_config` and `_split` say where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> # Nemotron-Math-v2
> 
> This repository contains the dataset accompanying the paper [Nemotron-Math: Efficient Long-Context Distillation of Mathematical Reasoning from Multi-Mode Supervision](https://arxiv.org/abs/2512.15489).
> 
> Code: [NeMo-Skills](https://github.com/NVIDIA-NeMo/Skills)
> 
> Documentation: [NeMo-Skills Nemotron-Math-v2 Documentation](https://nvidia-nemo.github.io/Skills/releases/nemotron-math-v2/)
> 
> ## Dataset Description
> 
> Nemotron-Math-v2 is a large-scale mathematical reasoning dataset containing approximately 347K high-quality mathematical problems and 7M model-generated reasoning trajectories. The dataset integrates human-authored problem sets with systematically generated solution traces produced under multiple reasoning modes and tool-use configurations.
> 
> Each problem is solved multiple times by the [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) model under six settings (high/medium/low reasoning × with/without Python TIR). Answers are verified using an LLM-as-a-judge pipeline, and trivial or unreliable problems are removed through pass-rate filtering. Only solutions whose final answers match the verified reference are included, resulting in a challenging, clean, and high-quality dataset suitable for training and evaluating mathematical reasoning systems.
> 
> All components of the pipeline, including problem extraction and data generation, are implemented using [NeMo-Skills](https://github.com/NVIDIA-NeMo/Skills). For detailed information, please refer to the **[official documentation](https://nvidia-nemo.github.io/Skills/releases/nemotron-math-v2/)**.
> 
> This dataset is ready for commercial use.  
> 
> ## Dataset Owner(s):
> NVIDIA Corporation
> 
> ## Dataset Creation Date:
> Created on: Dec 3, 2025  
> Last Modified on: Dec 18, 2025
> 
> ## License/Terms of Use: 
> The
> ...[card continues at the source link]
