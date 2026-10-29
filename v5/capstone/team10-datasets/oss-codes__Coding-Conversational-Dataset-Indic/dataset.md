# oss-codes/Coding-Conversational-Dataset-Indic

Source: https://huggingface.co/datasets/oss-codes/Coding-Conversational-Dataset-Indic

## Summary

| Field | Value |
|---|---|
| Category | indic, coding |
| Training stage | SFT |
| Licence tag (Hub, read 29 Oct 2026) | mit |
| Our decision | **check** |
| Gated | False |
| Last modified | 2025-04-07 |
| Downloads (30 d) | 280 |
| Languages | not stated |
| Task categories | not stated |
| Size category | not stated |
| Total rows (all configs) | 0 |

## Caveats and open checks

- Small; origin and generator not verified.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Configs and splits

- `default` / `validation`
- `default` / `test`

## Fields (from the sampled config)

Not available.

## Sample

`sample_20.jsonl` holds the first 20 rows read directly from the raw file(s) `hi/python_sharegpt_conversations.jsonl`, `ta/python_sharegpt_conversations.jsonl` (the datasets API had no preview for this dataset). Each line is one row; `_file` says where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

Card text not available (gated or empty).
