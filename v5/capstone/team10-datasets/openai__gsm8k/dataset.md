# openai/gsm8k

Source: https://huggingface.co/datasets/openai/gsm8k

## Summary

| Field | Value |
|---|---|
| Category | math |
| Training stage | eval only (benchmark) |
| Licence tag (Hub, read 29 Oct 2026) | mit |
| Our decision | **eval only: do not train on it** |
| Gated | False |
| Last modified | 2026-03-23 |
| Downloads (30 d) | 980999 |
| Languages | en |
| Task categories | text-generation |
| Size category | 1K<n<10K |
| Total rows (all configs) | 17584 |

## Caveats and open checks

- Small, human-written grade-school problems. Standard benchmark: used for evaluation, not for training. Other math sets may contain its problems, so decontaminate them against its test problems.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Configs and splits

- `main` / `train`
- `main` / `test`
- `socratic` / `train`
- `socratic` / `test`

## Fields (from the sampled config)

- `question`: {"dtype": "string", "_type": "Value"}
- `answer`: {"dtype": "string", "_type": "Value"}

## Sample

`sample_20.jsonl` holds the first 20 rows from `main/train`. Each line is one row; `_config` and `_split` say where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> # Dataset Card for GSM8K
> 
> ## Table of Contents
> - [Dataset Description](#dataset-description)
>   - [Dataset Summary](#dataset-summary)
>   - [Supported Tasks](#supported-tasks-and-leaderboards)
>   - [Languages](#languages)
> - [Dataset Structure](#dataset-structure)
>   - [Data Instances](#data-instances)
>   - [Data Fields](#data-instances)
>   - [Data Splits](#data-instances)
> - [Dataset Creation](#dataset-creation)
>   - [Curation Rationale](#curation-rationale)
>   - [Source Data](#source-data)
>   - [Annotations](#annotations)
>   - [Personal and Sensitive Information](#personal-and-sensitive-information)
> - [Considerations for Using the Data](#considerations-for-using-the-data)
>   - [Social Impact of Dataset](#social-impact-of-dataset)
>   - [Discussion of Biases](#discussion-of-biases)
>   - [Other Known Limitations](#other-known-limitations)
> - [Additional Information](#additional-information)
>   - [Dataset Curators](#dataset-curators)
>   - [Licensing Information](#licensing-information)
>   - [Citation Information](#citation-information)
> 
> ## Dataset Description
> 
> - **Homepage:** https://openai.com/blog/grade-school-math/
> - **Repository:** https://github.com/openai/grade-school-math
> - **Paper:** https://arxiv.org/abs/2110.14168
> - **Leaderboard:** [Needs More Information]
> - **Point of Contact:** [Needs More Information]
> 
> ### Dataset Summary
> 
> GSM8K (Grade School Math 8K) is a dataset of 8.5K high quality linguistically diverse grade school math word problems. The dataset was created to support the task of question answering on basic mathematical problems that require multi-step reasoning.
> - These problems take between 2 and 8 steps to solve.
> - Solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations (+ − ×÷) to reach the final answer.
> - A brigh
> ...[card continues at the source link]
