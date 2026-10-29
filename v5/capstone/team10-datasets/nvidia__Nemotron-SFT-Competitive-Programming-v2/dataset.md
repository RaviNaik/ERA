# nvidia/Nemotron-SFT-Competitive-Programming-v2

Source: https://huggingface.co/datasets/nvidia/Nemotron-SFT-Competitive-Programming-v2

## Summary

| Field | Value |
|---|---|
| Category | coding |
| Training stage | SFT |
| Licence tag (Hub, read 29 Oct 2026) | cc-by-4.0, odc-by, mit |
| Our decision | **use, check** |
| Gated | False |
| Last modified | 2026-03-11 |
| Downloads (30 d) | 4127 |
| Languages | en |
| Task categories | text-generation |
| Size category | not stated |
| Total rows (all configs) | unknown |

## Caveats and open checks

- Mixed per-subset licences (cc-by-4.0, odc-by, mit): read the card per subset.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Configs and splits

- `default` / `exercism`
- `default` / `text_to_sql`
- `default` / `competitive_coding_cpp`
- `default` / `competitive_coding_python`

## Fields (from the sampled config)

Not available.

## Sample

`sample_20.jsonl` holds the first 20 rows read directly from the raw file(s) `data/competitive_programming_python_00.jsonl` (the datasets API had no preview for this dataset). Each line is one row; `_file` says where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> ## Dataset Description:
> Nemotron-Competitive-Programming-v2 is a large-scale synthetic coding and reasoning dataset designed to push LLM performance on challenging programming and systems tasks. It combines Python and C++ samples across unique competitive programming questions.  
> In addition, the dataset includes samples targetting a subset of programming language exercises from [Exercism](https://exercism.org/).  
> 
> Beyond problem solving, the dataset includes Text-to-SQL, which aims at training LLMs to reason and perform tasks involving SQL.
> 
> 
> This dataset is ready for commercial use.
> 
> ### Competitive Coding
> The competitive coding subset of Nemotron-Competitive-Programming-v2 is one of the largest reasoning-based synthetic dataset to date for coding, comprising 336k+ samples in Python and 332k++ samples in C++ across 34,799 unique competitive programming questions. 
> 
> This subset is designed for supervised fine-tuning (SFT) tasks of code completion and code critique.
> 
> - [Github Repo](https://github.com/NVIDIA/NeMo-Skills) - Access the complete pipeline used to generate the data and perform SFT.
> 
> 
> 
> ### Python Exercises
> 
> The Exercism subset is a synthetic dataset derived from 115 Python Exercism problems (with problems from the Aider benchmark decontaminated) totalling 40K generated coding problems and their solutions.
> 
> ### Text-To-SQL
> 
> The Text-to-SQL Dataset contains 96.5k high-fidelity synthetic records designed to train LLMs to reason and perform better on SQL tasks. Unlike standard benchmarks, every sample includes a reasoning trace, teaching models to plan and validate logic before writing code. Generated via NVIDIA Data Designer, it simulates messy enterprise environments with noisy data, distractor tables, and dialect-specific constraints (MySQL, PostgreSQL, SQLit
> ...[card continues at the source link]
