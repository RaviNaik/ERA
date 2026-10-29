# nebius/SWE-rebench-openhands-trajectories

Source: https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories

## Summary

| Field | Value |
|---|---|
| Category | coding |
| Training stage | SFT (agent trajectories) |
| Licence tag (Hub, read 29 Oct 2026) | cc-by-4.0 |
| Our decision | **use** |
| Gated | False |
| Last modified | 2025-12-27 |
| Downloads (30 d) | 10753 |
| Languages | not stated |
| Task categories | not stated |
| Size category | 10K<n<100K |
| Total rows (all configs) | 67074 |

## Caveats and open checks

- OpenHands trajectories from Qwen3-Coder-480B on SWE-rebench tasks.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Benchmark overlap

- Risk: **Medium** (evidence: not stated)
- Trajectories on SWE-rebench tasks; no overlap statement. Apply the same task-ID check as SWE-rebench-V2.
- Card-level check only, no text-overlap test run. See `../BENCHMARK_OVERLAP.md`.

## Configs and splits

- `default` / `train`

## Fields (from the sampled config)

- `trajectory_id`: {"dtype": "string", "_type": "Value"}
- `instance_id`: {"dtype": "string", "_type": "Value"}
- `repo`: {"dtype": "string", "_type": "Value"}
- `trajectory`: {"feature": {"content": {"dtype": "string", "_type": "Value"}, "name": {"dtype": "string", "_type": "Value"}, "role": {"
- `tools`: {"feature": {"function": {"description": {"dtype": "string", "_type": "Value"}, "name": {"dtype": "string", "_type": "Va
- `model_patch`: {"dtype": "string", "_type": "Value"}
- `exit_status`: {"dtype": "string", "_type": "Value"}
- `resolved`: {"dtype": "int64", "_type": "Value"}
- `gen_tests_correct`: {"dtype": "float64", "_type": "Value"}
- `pred_passes_gen_tests`: {"dtype": "float64", "_type": "Value"}

## Sample

`sample_20.jsonl` holds the first 20 rows from `default/train`. Each line is one row; `_config` and `_split` say where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> # Dataset Summary
> 
> **SWE-rebench-OpenHands-Trajectories** is a dataset of multi-turn agent trajectories for software engineering tasks, collected
> using [Qwen/Qwen3-Coder-480B-A35B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct) with OpenHands (v0.54.0) agent scaffolding.
> This dataset captures complete agent execution traces as they attempt to resolve real GitHub issues from
> [nebius/SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench).
> Each trajectory contains the agent's step-by-step reasoning, actions, and environmental observations.
> 
> | Metric | | [SWE-bench/SWE-smith-trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) | [Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k](https://huggingface.co/datasets/Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k) | [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) | [SWE-Gym/OpenHands-Sampled-Trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-Sampled-Trajectories) | [R2E-Gym/R2EGym-SFT-Trajectories](https://huggingface.co/datasets/R2E-Gym/R2EGym-SFT-Trajectories) | Ours |
> |--------|------|------|------|------|------|------|------|
> | **Scaffolding** | | SWE-agent | mini-swe-agent-plus | Closed-source | OpenHands | OpenHands | OpenHands (v0.54.0) |
> | **Bootstrapping Model** | Name | claude-3-7-sonnet-20250219claude-3-5-sonnet-20241022gpt-4o-2024-08-06 | *unknown\** | Qwen2.5-72B-InstructLlama3-70B-Instruct | gpt-4o-2024-08-06claude-3-5-sonnet-20241022 | Claude-Sonnet-3.5-v2 | Qwen3-Coder-480B-A35B-Instruct |
> | | Uses function calling | ✅  | ❌ | ❌ | ✅  | ✅  | ✅ |
> | **Repositories** | | 129 | 123 | 1,202 | 11 | *unknown\** | **1,823** |
> | **Issues** | Resolved Count | 7,270 | **10,894** | 838 | 294
> ...[card continues at the source link]
