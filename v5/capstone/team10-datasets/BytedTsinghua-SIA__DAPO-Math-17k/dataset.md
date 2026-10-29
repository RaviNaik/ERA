# BytedTsinghua-SIA/DAPO-Math-17k

Source: https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k

## Summary

| Field | Value |
|---|---|
| Category | math |
| Training stage | RL |
| Licence tag (Hub, read 29 Oct 2026) | apache-2.0 |
| Our decision | **use** |
| Gated | False |
| Last modified | 2025-04-18 |
| Downloads (30 d) | 17316 |
| Languages | en |
| Task categories | text-generation |
| Size category | 1M<n<10M |
| Total rows (all configs) | 1791700 |

## Caveats and open checks

- Verifiable integer-answer math prompts.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Benchmark overlap

- Risk: **Medium** (evidence: not stated)
- Competition-style math problems; the card has no data description or decontamination note. Check against AIME 2024/25 and MATH-500.
- Card-level check only, no text-overlap test run. See `../BENCHMARK_OVERLAP.md`.

## Configs and splits

- `default` / `train`

## Fields (from the sampled config)

- `data_source`: {"dtype": "string", "_type": "Value"}
- `prompt`: {"feature": {"content": {"dtype": "string", "_type": "Value"}, "role": {"dtype": "string", "_type": "Value"}}, "_type": 
- `ability`: {"dtype": "string", "_type": "Value"}
- `reward_model`: {"ground_truth": {"dtype": "string", "_type": "Value"}, "style": {"dtype": "string", "_type": "Value"}}
- `extra_info`: {"index": {"dtype": "string", "_type": "Value"}}

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
