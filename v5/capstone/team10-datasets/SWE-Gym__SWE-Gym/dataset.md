# SWE-Gym/SWE-Gym

Source: https://huggingface.co/datasets/SWE-Gym/SWE-Gym

## Summary

| Field | Value |
|---|---|
| Category | coding |
| Training stage | RL (SWE tasks) |
| Licence tag (Hub, read 29 Oct 2026) | mit |
| Our decision | **use** |
| Gated | False |
| Last modified | 2025-05-10 |
| Downloads (30 d) | 24240 |
| Languages | not stated |
| Task categories | not stated |
| Size category | not stated |
| Total rows (all configs) | 2438 |

## Caveats and open checks

- Python repositories keep their own licences. Also in Amit's merged licence table.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Configs and splits

- `default` / `train`

## Fields (from the sampled config)

- `instance_id`: {"dtype": "string", "_type": "Value"}
- `hints_text`: {"dtype": "string", "_type": "Value"}
- `patch`: {"dtype": "string", "_type": "Value"}
- `test_patch`: {"dtype": "string", "_type": "Value"}
- `created_at`: {"dtype": "string", "_type": "Value"}
- `problem_statement`: {"dtype": "string", "_type": "Value"}
- `repo`: {"dtype": "string", "_type": "Value"}
- `base_commit`: {"dtype": "string", "_type": "Value"}
- `version`: {"dtype": "string", "_type": "Value"}
- `PASS_TO_PASS`: {"feature": {"dtype": "string", "_type": "Value"}, "_type": "List"}
- `FAIL_TO_PASS`: {"feature": {"dtype": "string", "_type": "Value"}, "_type": "List"}

## Sample

`sample_20.jsonl` holds the first 20 rows from `default/train`. Each line is one row; `_config` and `_split` say where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> SWE-Gym contains 2438 instances sourced from 11 Python repos, following SWE-Bench data collection procedure.
> 
> **Get started at project page [github.com/SWE-Gym/SWE-Gym](https://github.com/SWE-Gym/SWE-Gym)**
> 
> 
> ![image/png](https://cdn-uploads.huggingface.co/production/uploads/61568f37272f2d87a99ba884/GHyyhROIieakMg6nuO0yH.png)
> 
> ![image/png](https://cdn-uploads.huggingface.co/production/uploads/61568f37272f2d87a99ba884/gBKSiohnGKnp3toKw6H3Q.png)
