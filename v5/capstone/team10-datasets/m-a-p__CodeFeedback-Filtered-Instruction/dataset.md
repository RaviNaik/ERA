# m-a-p/CodeFeedback-Filtered-Instruction

Source: https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction

## Summary

| Field | Value |
|---|---|
| Category | coding |
| Training stage | SFT |
| Licence tag (Hub, read 29 Oct 2026) | apache-2.0 |
| Our decision | **use, check** |
| Gated | False |
| Last modified | 2024-02-26 |
| Downloads (30 d) | 22241 |
| Languages | en |
| Task categories | question-answering |
| Size category | 10K<n<100K |
| Total rows (all configs) | 156526 |

## Caveats and open checks

- Built from several sources: check each.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Benchmark overlap

- Risk: **Medium** (evidence: not stated)
- Filtered from Magicoder-OSS-Instruct, a ShareGPT Python subset, Magicoder-Evol-Instruct and Evol-Instruct-Code. No decontamination statement. Evol-Instruct-style code sets have had HumanEval-contamination concerns (recollection (unverified)). Check against HumanEval and MBPP.
- Card-level check only, no text-overlap test run. See `../BENCHMARK_OVERLAP.md`.

## Configs and splits

- `default` / `train`

## Fields (from the sampled config)

- `query`: {"dtype": "string", "_type": "Value"}
- `answer`: {"dtype": "string", "_type": "Value"}
- `resource`: {"dtype": "string", "_type": "Value"}
- `lang`: {"dtype": "string", "_type": "Value"}

## Sample

`sample_20.jsonl` holds the first 20 rows from `default/train`. Each line is one row; `_config` and `_split` say where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement
> 
> 
> 
> 
> 
>   [🏠Homepage] 
>   |
>   [🛠️Code] 
> 
> 
> 
> ## OpenCodeInterpreter
> OpenCodeInterpreter is a family of open-source code generation systems designed to bridge the gap between large language models and advanced proprietary systems like the GPT-4 Code Interpreter. It significantly advances code generation capabilities by integrating execution and iterative refinement functionalities.
> 
> For further information and related work, refer to our paper: ["OpenCodeInterpreter: A System for Enhanced Code Generation and Execution"](https://arxiv.org/abs/2402.14658) available on arXiv.
> 
> ## Dataset Description
> 
> CodeFeedback-Filtered-Instruction is a curated collection of code instruction queries extracted from four prominent open-source code instruction tuning datasets: [Magicoder-OSS-Instruct](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K), [Python code subset of ShareGPT](https://huggingface.co/datasets/ajibawa-2023/Python-Code-23k-ShareGPT), [Magicoder-Evol-Instruct](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K), and [Evol-Instruct-Code](https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1). 
> Initially, 287k queries were aggregated from these datasets. To isolate the most intricate and informative instructions, a rigorous filtering process was employed. 
> This involved utilizing the Qwen-72B-Chat, an open-source chat model, for selective filtering. 
> The code queries are evaluated along with their corresponding responses within the compiled datasets by the LLM, assigning a complexity score ranging from 1 to 5, and only those rated 4 or 5 were retained for the seed set. 
> This meticulous filtering process resulted in a final collection of 156k high-qualit
> ...[card continues at the source link]
