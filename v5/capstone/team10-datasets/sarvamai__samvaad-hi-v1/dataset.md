# sarvamai/samvaad-hi-v1

Source: https://huggingface.co/datasets/sarvamai/samvaad-hi-v1

## Summary

| Field | Value |
|---|---|
| Category | indic |
| Training stage | SFT |
| Licence tag (Hub, read 29 Oct 2026) | apache-2.0 |
| Our decision | **use, check** |
| Gated | False |
| Last modified | 2024-07-16 |
| Downloads (30 d) | 738 |
| Languages | en, hi |
| Task categories | text-generation |
| Size category | not stated |
| Total rows (all configs) | 101476 |

## Caveats and open checks

- Hindi, English and Hinglish chats; the card does not say which model wrote the replies.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Configs and splits

- `default` / `train`

## Fields (from the sampled config)

- `messages`: {"feature": {"content": {"dtype": "string", "_type": "Value"}, "role": {"dtype": "string", "_type": "Value"}}, "_type": 

## Sample

`sample_20.jsonl` holds the first 20 rows from `default/train`. Each line is one row; `_config` and `_split` say where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> 100k high-quality conversations in English, Hindi, and Hinglish curated exclusively with an Indic context.
