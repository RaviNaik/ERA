# nvidia/OpenMathReasoning

Source: https://huggingface.co/datasets/nvidia/OpenMathReasoning

## Summary

| Field | Value |
|---|---|
| Category | math |
| Training stage | SFT + RL prompts |
| Licence tag (Hub, read 29 Oct 2026) | cc-by-4.0 |
| Our decision | **use, check** |
| Gated | False |
| Last modified | 2025-05-27 |
| Downloads (30 d) | 68644 |
| Languages | en |
| Task categories | question-answering, text-generation |
| Size category | 1M<n<10M |
| Total rows (all configs) | 5678317 |

## Caveats and open checks

- Long CoT and tool-integrated solutions; generator terms (DeepSeek-R1, QwQ). Splits are cot, tir, genselect and additional_problems.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Configs and splits

- `default` / `cot`
- `default` / `tir`
- `default` / `genselect`
- `default` / `additional_problems`

## Fields (from the sampled config)

- `expected_answer`: {"dtype": "string", "_type": "Value"}
- `problem_type`: {"dtype": "string", "_type": "Value"}
- `problem_source`: {"dtype": "string", "_type": "Value"}
- `generation_model`: {"dtype": "string", "_type": "Value"}
- `pass_rate_72b_tir`: {"dtype": "string", "_type": "Value"}
- `problem`: {"dtype": "string", "_type": "Value"}
- `generated_solution`: {"dtype": "string", "_type": "Value"}
- `inference_mode`: {"dtype": "string", "_type": "Value"}
- `used_in_kaggle`: {"dtype": "bool", "_type": "Value"}

## Sample

`sample_20.jsonl` holds the first 20 rows from `default/cot`, `default/tir`. Each line is one row; `_config` and `_split` say where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> # OpenMathReasoning
> 
>  
> OpenMathReasoning is a large-scale math reasoning dataset for training large language models (LLMs). 
> This dataset contains 
> 
> * 306K unique mathematical problems sourced from [AoPS forums](https://artofproblemsolving.com/community) with: 
>     * 3.2M long chain-of-thought (CoT) solutions
>     * 1.7M long tool-integrated reasoning (TIR) solutions
>     * 566K samples that select the most promising solution out of many candidates (GenSelect)
> * Additional 193K problems sourced from AoPS forums (problems only, no solutions)
> 
> We used [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) to preprocess problems, and 
> [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) and [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) to generate solutions. 
> 
> This dataset was a foundation of our winning submission to the 
> [AIMO-2 Kaggle competition](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/leaderboard).
> 
> See our [paper](https://arxiv.org/abs/2504.16891) to learn more details!
> 
> **_NOTE:_** We initially reported 540K unique problems in our dataset, but this figure represented the question count at the pipeline's beginning. But the actual CoT and TIR solutions in the released dataset correspond to 306K problems. Since our OpenMath-Nemotron models were trained on this reduced subset, all published results remain reproducible with the current release—only our initial problem count was overstated.
> 
> Two factors explain this discrepancy in question numbers. 
> - Our filtering process removed many questions due to format restrictions (yes/no questions, multiple choice questions, etc.), benchmark decontamination, and the inability of existing LLMs to generate valid solutions for certain problems. 
> - A pipeline bug caused
> ...[card continues at the source link]
