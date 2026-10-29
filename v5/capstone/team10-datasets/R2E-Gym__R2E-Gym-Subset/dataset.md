# R2E-Gym/R2E-Gym-Subset

Source: https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset

## Summary

| Field | Value |
|---|---|
| Category | coding |
| Training stage | RL (SWE tasks) |
| Licence tag (Hub, read 29 Oct 2026) | apache-2.0 |
| Our decision | **use** |
| Gated | False |
| Last modified | 2026-07-23 |
| Downloads (30 d) | 17717 |
| Languages | not stated |
| Task categories | not stated |
| Size category | not stated |
| Total rows (all configs) | 4578 |

## Caveats and open checks

- Card is sparse; licence from the tag and the project repo.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Benchmark overlap

- Risk: **Medium** (evidence: not stated)
- The card does not state overlap with SWE-bench. Compare the `repo_name` field with the SWE-bench repos.
- Card-level check only, no text-overlap test run. See `../BENCHMARK_OVERLAP.md`.

## Configs and splits

- `default` / `train`

## Fields (from the sampled config)

- `repo_name`: {"dtype": "string", "_type": "Value"}
- `docker_image`: {"dtype": "string", "_type": "Value"}
- `commit_hash`: {"dtype": "string", "_type": "Value"}
- `parsed_commit_content`: {"dtype": "string", "_type": "Value"}
- `execution_result_content`: {"dtype": "string", "_type": "Value"}
- `modified_files`: {"feature": {"dtype": "string", "_type": "Value"}, "_type": "List"}
- `modified_entity_summaries`: {"feature": {"ast_type_str": {"dtype": "string", "_type": "Value"}, "end_lineno": {"dtype": "int64", "_type": "Value"}, 
- `relevant_files`: {"feature": {"dtype": "string", "_type": "Value"}, "_type": "List"}
- `num_non_test_files`: {"dtype": "int64", "_type": "Value"}
- `num_non_test_func_methods`: {"dtype": "int64", "_type": "Value"}
- `num_non_test_lines`: {"dtype": "int64", "_type": "Value"}
- `prompt`: {"dtype": "string", "_type": "Value"}
- `problem_statement`: {"dtype": "string", "_type": "Value"}
- `expected_output_json`: {"dtype": "string", "_type": "Value"}

## Sample

`sample_20.jsonl` holds the first 20 rows from `default/train`. Each line is one row; `_config` and `_split` say where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

Card text not available (gated or empty).
