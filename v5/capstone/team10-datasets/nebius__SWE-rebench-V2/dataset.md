# nebius/SWE-rebench-V2

Source: https://huggingface.co/datasets/nebius/SWE-rebench-V2

## Summary

| Field | Value |
|---|---|
| Category | coding |
| Training stage | RL (SWE tasks) |
| Licence tag (Hub, read 29 Oct 2026) | cc-by-4.0 |
| Our decision | **use** |
| Gated | False |
| Last modified | 2026-10-01 |
| Downloads (30 d) | 104537 |
| Languages | en |
| Task categories | text-generation |
| Size category | not stated |
| Total rows (all configs) | 32079 |

## Caveats and open checks

- 20 languages including Go; check overlap with any SWE-bench eval sets.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Configs and splits

- `default` / `train`

## Fields (from the sampled config)

- `base_commit`: {"dtype": "string", "_type": "Value"}
- `created_at`: {"dtype": "string", "_type": "Value"}
- `image_name`: {"dtype": "string", "_type": "Value"}
- `instance_id`: {"dtype": "string", "_type": "Value"}
- `interface`: {"dtype": "string", "_type": "Value"}
- `language`: {"dtype": "string", "_type": "Value"}
- `license`: {"dtype": "string", "_type": "Value"}
- `patch`: {"dtype": "string", "_type": "Value"}
- `pr_description`: {"dtype": "string", "_type": "Value"}
- `problem_statement`: {"dtype": "string", "_type": "Value"}
- `repo`: {"dtype": "string", "_type": "Value"}
- `test_patch`: {"dtype": "string", "_type": "Value"}
- `FAIL_TO_PASS`: {"feature": {"dtype": "string", "_type": "Value"}, "_type": "List"}
- `PASS_TO_PASS`: {"feature": {"dtype": "string", "_type": "Value"}, "_type": "List"}
- `install_config`: {"base_image_name": {"dtype": "string", "_type": "Value"}, "docker_specs": {"_variant": {"dtype": "string", "_type": "Va
- `meta`: {"llm_metadata": {"code": {"dtype": "string", "_type": "Value"}, "confidence": {"dtype": "float64", "_type": "Value"}, "

## Sample

`sample_20.jsonl` holds the first 20 rows from `default/train`. Each line is one row; `_config` and `_split` say where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> # SWE-rebench-V2
> 
> ## Dataset Summary
> 
> SWE-rebench-V2 is a curated dataset of software-engineering tasks derived from real GitHub issues and pull requests. The dataset contains 32,079 samples covering Python, Go, TypeScript, JavaScript, Rust, Java, PHP, Kotlin, Julia, Elixir, Scala, Swift, Dart, C, C++, C#, R, Clojure, OCaml, and Lua.
> For log parser functions, base Dockerfiles, and the prompts used, please see https://github.com/SWE-rebench/SWE-rebench-V2  
> The detailed technical report is available at [“SWE-rebench V2: Language-Agnostic SWE Task Collection at Scale”](https://arxiv.org/abs/2602.23866).
> 
> ❗ **Benchmark overlap:** We excluded repositories from the original SWE-bench, but did not explicitly filter other benchmarks. Before training, check the [overlap analysis and task IDs](https://github.com/raimondasl/agentleak#case-3-swe-rebench-v2-vs-multilingual-swe-benchmarks) and exclude tasks that overlap with your evaluation benchmarks.
> 
> ## Quick Start
> 
> ```python
> from datasets import load_dataset
> 
> ds = load_dataset("nebius/SWE-rebench-V2", split="train")
> print(len(ds))  # 32079
> ```
> 
> ## Dataset Structure
> 
> | Field | Type | Description |
> |---|---|---|
> | `instance_id` | `string` | Unique identifier for the instance |
> | `repo` | `string` | GitHub repository in `owner/repo` format |
> | `base_commit` | `string` | Git commit SHA of the base before the fix |
> | `patch` | `string` | The gold patch that resolves the issue |
> | `test_patch` | `string` | Diff adding or modifying tests that verify the fix |
> | `problem_statement` | `string` | Issue description the patch addresses |
> | `pr_description` | `string` | Full pull request description |
> | `created_at` | `int64` | Unix timestamp (milliseconds) of the issue/PR creation |
> | `image_name` | `string` | Docker image name used for the
> ...[card continues at the source link]
