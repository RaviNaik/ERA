# zwhe99/DeepMath-103K

Source: https://huggingface.co/datasets/zwhe99/DeepMath-103K

## Summary

| Field | Value |
|---|---|
| Category | math |
| Training stage | RL (and SFT) |
| Licence tag (Hub, read 29 Oct 2026) | mit |
| Our decision | **use** |
| Gated | False |
| Last modified | 2025-05-29 |
| Downloads (30 d) | 16760 |
| Languages | en |
| Task categories | text-generation, text2text-generation |
| Size category | 100K<n<1M |
| Total rows (all configs) | 103022 |

## Caveats and open checks

- 103k problems with verifiable answers.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Benchmark overlap

- Risk: **Low** (evidence: card)
- The card says it was decontaminated against common benchmarks using semantic matching.
- Card-level check only, no text-overlap test run. See `../BENCHMARK_OVERLAP.md`.

## Configs and splits

- `default` / `train`

## Fields (from the sampled config)

- `question`: {"dtype": "string", "_type": "Value"}
- `final_answer`: {"dtype": "string", "_type": "Value"}
- `difficulty`: {"dtype": "float64", "_type": "Value"}
- `topic`: {"dtype": "string", "_type": "Value"}
- `r1_solution_1`: {"dtype": "string", "_type": "Value"}
- `r1_solution_2`: {"dtype": "string", "_type": "Value"}
- `r1_solution_3`: {"dtype": "string", "_type": "Value"}

## Sample

`sample_20.jsonl` holds the first 20 rows from `default/train`. Each line is one row; `_config` and `_split` say where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> # DeepMath-103K
> 
>   
>     
>       
>         
>       
>     
>     
>       
>         
>       
>     
>     
>       
>         
>       
>     
>     
>       
>         
>       
>     
>   
> 
> 
> ## 🔥 News
> - **May 8, 2025**: We found that 48 samples contained hints that revealed the answers. The relevant questions have now been revised to remove the leaked answers.
> - **April 14, 2025**: We release **`DeepMath-103K`**, a large-scale dataset featuring challenging, verifiable, and decontaminated math problems tailored for RL and SFT. We open source:
> 
> ## 📦 Resource
> 
> - 🤗 Training data: [`DeepMath-103K`](https://huggingface.co/datasets/zwhe99/DeepMath-103K)
> - 🤗 Model weights: [`DeepMath-Zero-7B`](https://huggingface.co/zwhe99/DeepMath-Zero-7B), [`DeepMath-Zero-Math-7B`](https://huggingface.co/zwhe99/DeepMath-Zero-Math-7B), [`DeepMath-1.5B`](https://huggingface.co/zwhe99/DeepMath-1.5B),  [`DeepMath-Omn-1.5B`](https://huggingface.co/zwhe99/DeepMath-Omn-1.5B)
> - 💻 Code: [`DeepMath`](https://github.com/zwhe99/DeepMath)
> - 📝 Paper: [`arXiv:2504.11456`](https://arxiv.org/abs/2504.11456)
> 
> 
> ## 📖 Overview
> 
> **`DeepMath-103K`** is meticulously curated to push the boundaries of mathematical reasoning in language models. Key features include:
> 
> **1. Challenging Problems**: DeepMath-103K has a strong focus on difficult mathematical problems (primarily Levels 5-9), significantly raising the complexity bar compared to many existing open datasets.
> 
>  
> 
> Difficulty distribution comparison. 
> 
> **2. Data Diversity and Novelty**: DeepMath-103K spans a wide spectrum of mathematical subjects, including Algebra, Calculus, Number Theory, Geometry, Probability, and Discrete Mathematics.
> 
>  
> 
> Hierarchical breakdown of mathematical topics covered in DeepMath-103K.
> 
> The problems in DeepMath-103K are novel and unique, whereas many existing datasets a
> ...[card continues at the source link]
