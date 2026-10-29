# TIGER-Lab/AceCode-87K

Source: https://huggingface.co/datasets/TIGER-Lab/AceCode-87K

## Summary

| Field | Value |
|---|---|
| Category | coding |
| Training stage | RL |
| Licence tag (Hub, read 29 Oct 2026) | mit |
| Our decision | **use** |
| Gated | False |
| Last modified | 2025-02-08 |
| Downloads (30 d) | 16276 |
| Languages | en |
| Task categories | not stated |
| Size category | 10K<n<100K |
| Total rows (all configs) | 87149 |

## Caveats and open checks

- Problems with test cases. The card says each question was rewritten by GPT-4o-mini, so OpenAI's terms apply to the question text (verified on the card, 29 Oct).
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Benchmark overlap

- Risk: **Medium** (evidence: not stated)
- Questions come from several instruction sets (see the `source` field), rewritten by an LLM, with HumanEval/MBPP-style tests. Check against HumanEval and MBPP.
- Card-level check only, no text-overlap test run. See `../BENCHMARK_OVERLAP.md`.

## Configs and splits

- `default` / `train`

## Fields (from the sampled config)

- `id`: {"dtype": "string", "_type": "Value"}
- `source`: {"dtype": "string", "_type": "Value"}
- `question`: {"dtype": "string", "_type": "Value"}
- `test_cases`: {"feature": {"dtype": "string", "_type": "Value"}, "_type": "List"}
- `inferences`: {"feature": {"completion": {"dtype": "string", "_type": "Value"}, "completion_id": {"dtype": "int64", "_type": "Value"},
- `context_messages`: {"feature": {"content": {"dtype": "string", "_type": "Value"}, "role": {"dtype": "string", "_type": "Value"}}, "_type": 

## Sample

`sample_20.jsonl` holds the first 20 rows from `default/train`. Each line is one row; `_config` and `_split` say where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> # 🂡 AceCode-87K
> 
> [Paper](https://arxiv.org/abs/2502.01718) | 
> [Github](https://github.com/TIGER-AI-Lab/AceCoder) |
> [AceCode-87K](https://huggingface.co/datasets/TIGER-Lab/AceCode-87K) |
> [AceCodePair-300K](https://huggingface.co/datasets/TIGER-Lab/AceCodePair-300K) |
> [RM/RL Models](https://huggingface.co/collections/TIGER-Lab/acecoder-67a16011a6c7d65cad529eba)
> 
> We introduce AceCoder, the first work to propose a fully automated pipeline for synthesizing large-scale reliable tests used for the reward model training and reinforcement learning in the coding scenario. To do this, we curated the dataset AceCode-87K, where we start from a seed code dataset and prompt powerful LLMs to "imagine" proper test cases for the coding question and filter the noisy ones. We sample inferences from existing coder models and compute their pass rate as the reliable and verifiable rewards for both training the reward model and conducting the reinforcement learning for coder LLM.
> 
> - **This dataset is the official AceCodeRM-87K after the test case filtering**. 
> - Each question in the dataset is rewritten by GPT-4o-mini along with an average of **16** cleaned test cases.
> 
> ![https://tiger-ai-lab.github.io/AceCoder/static/images/ac_overview.png](https://tiger-ai-lab.github.io/AceCoder/static/images/ac_overview.png)
> 
> ## Data Formats
> 
> - `Id` (str): Unique identifier for each question
> - `source` (str): which dataset
> - `question` (str): the question
> - `test_cases` (List[str]): test cases for the question
> - `inferences` (List[dict]):
>     - `model_name` (str): the model name
>     - `completion_id` (int): the completion id
>     - `completion` (str): the completion
>     - `pass_rate` (float): the pass rate
>     - `test_results` (List[int]): the test results
> - `context_messages` (List[List[dict]]): context mes
> ...[card continues at the source link]
