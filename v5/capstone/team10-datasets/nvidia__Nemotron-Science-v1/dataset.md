# nvidia/Nemotron-Science-v1

Source: https://huggingface.co/datasets/nvidia/Nemotron-Science-v1

## Summary

| Field | Value |
|---|---|
| Category | science |
| Training stage | SFT |
| Licence tag (Hub, read 29 Oct 2026) | cc-by-4.0 |
| Our decision | **use, check** |
| Gated | False |
| Last modified | 2025-12-15 |
| Downloads (30 d) | 5185 |
| Languages | en |
| Task categories | not stated |
| Size category | not stated |
| Total rows (all configs) | 226334 |

## Caveats and open checks

- Generator models not read; check them.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Configs and splits

- `default` / `MCQ`
- `default` / `RQA`

## Fields (from the sampled config)

- `uuid`: {"dtype": "string", "_type": "Value"}
- `messages`: {"feature": {"role": {"dtype": "string", "_type": "Value"}, "content": {"dtype": "string", "_type": "Value"}, "reasoning
- `license`: {"dtype": "string", "_type": "Value"}
- `used_in`: {"feature": {"dtype": "string", "_type": "Value"}, "_type": "List"}
- `tools`: {"feature": {"dtype": "null", "_type": "Value"}, "_type": "List"}

## Sample

`sample_20.jsonl` holds the first 20 rows from `default/MCQ`. Each line is one row; `_config` and `_split` say where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> ## Dataset Description:
> Nemotron-Science-v1 is a synthetic science reasoning dataset with two subsets: an MCQA set that improves on the STEM portion of Nemotron-Post-Training-v1 using GPT-OSS-120B to generate GPQA-style questions and reasoning traces, and an RQA set of synthetic chemistry questions. 
> 
> This dataset is ready for commercial use.
> 
> The Nemotron-Science-v1  dataset contains the following subsets:
> 
> ### MCQA
> This subset is an improvement of the STEM subset in [Nemotron-Post-Training-Dataset-v1](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1) with reasoning traces generated with GPT-OSS-120B. The dataset consists of synthetic science questions designed to mimic GPQA-style topics and subtopics, generated to enhance large language model reasoning capabilities in scientific domains. 
> 
> ### RQA
> This dataset consists of synthetic chemistry questions, generated to enhance large language model reasoning capabilities in scientific domains.
> 
> 
> ## Dataset Owner(s):
> NVIDIA Corporation
> 
> ## Dataset Creation Date:
> Created on: Dec 3, 2025  
> Last Modified on: Dec 3, 2025
> 
> ## License/Terms of Use: 
> This dataset is governed by the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
> 
> ## Intended Usage:
> This dataset is intended for LLM engineers and research teams developing and training large language models with a focus on improving scientific reasoning and problem-solving capabilities. It is suitable for supervised training and data augmentation in science-based model development pipelines.
> 
> ## Dataset Characterization
> **Data Collection Method**  
> Synthetic - LLM-generated scientific question and solution pairs 
> 
> 
> **Labeling Method**  
> Synthetic - Model-genera
> ...[card continues at the source link]
