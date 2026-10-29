# nvidia/Nemotron-SFT-SWE-v3

Source: https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v3

## Summary

| Field | Value |
|---|---|
| Category | coding |
| Training stage | SFT (agent trajectories) |
| Licence tag (Hub, read 29 Oct 2026) | cc-by-4.0, apache-2.0, mit, bsd-3-clause, bsd-2-clause |
| Our decision | **use** |
| Gated | False |
| Last modified | 2026-06-06 |
| Downloads (30 d) | 3631 |
| Languages | en |
| Task categories | text-generation |
| Size category | 100K<n<1M |
| Total rows (all configs) | 237970 |

## Caveats and open checks

- OpenHands / SWE-agent / mini-SWE-agent trajectories; per-source licences are on the card.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Configs and splits

- `default` / `train`

## Fields (from the sampled config)

- `messages`: {"feature": {"role": {"dtype": "string", "_type": "Value"}, "content": {"dtype": "string", "_type": "Value"}, "reasoning
- `uuid`: {"dtype": "string", "_type": "Value"}
- `license`: {"dtype": "string", "_type": "Value"}

## Sample

`sample_20.jsonl` holds the first 20 rows from `default/train`. Each line is one row; `_config` and `_split` say where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> ## Dataset Description:
> 
> Nemotron-SFT-SWE-v3 is a software engineering instruction tuning dataset designed to advance the capabilities of LLMs on SWE-Bench style tasks. 
> It includes agentic trajectories collected using a variety of agent harnesses, including the OpenHands, SWE-agent, and mini-SWE-agent frameworks.
> 
> This dataset is ready for commercial use.
> 
> ## Dataset Owner(s):
> NVIDIA Corporation
> 
> ## Dataset Creation Date:
> Created on: 2026-06-04 
> Last Modified on: 2026-06-04
> 
> ## License/Terms of Use: 
> This dataset is licensed under Creative Commons Attribution 4.0 International (CC-BY 4.0).  
> Additional Information: Apache 2.0 License; MIT License; BSD-3 License; BSD-2 License.
> 
> ## Intended Usage:
> 
> This dataset is intended for LLM engineers and research teams building autonomous software engineering agents and code-focused assistants. It is suitable for 
> supervised fine-tuning and distillation of models that must interpret real-world issue statements, plan multi-step tool use, navigate codebases, and implement fixes in 
> a SWE-Bench–style setting. The trajectories can also be used to benchmark and debug agent policies, improve repository-aware reasoning, and study robust, regression-free 
> code editing behaviors in both academic and production environments.
> 
> ## Dataset Characterization
> **Data Collection Method** 
> Hybrid: Automated, Synthetic  
> 
> **Labeling Method** 
> Hybrid: Automated, Synthetic 
> 
> ## Dataset Format  
> Modality: Text  
> Format: JSONL  
> Structure: Text + Metadata
> 
> ## Dataset Quantification
> 
> | Subset | Samples |
> |--------|---------|
> | Total            | 237,970 |
> 
> Total Data Storage: ~11.7GB
> 
> ## Reference(s):
> 
> * [Training Software Engineering Agents and Verifiers with SWE-Gym](https://arxiv.org/abs/2412.21139)
> * [R2E-Gym: Procedural Environments and Hybrid Verif
> ...[card continues at the source link]
