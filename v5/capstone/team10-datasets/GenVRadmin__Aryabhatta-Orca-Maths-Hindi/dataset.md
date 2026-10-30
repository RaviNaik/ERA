# GenVRadmin/Aryabhatta-Orca-Maths-Hindi

Source: https://huggingface.co/datasets/GenVRadmin/Aryabhatta-Orca-Maths-Hindi

## Summary

| Field | Value |
|---|---|
| Category | indic, math |
| Training stage | SFT |
| Licence tag (Hub, read 29 Oct 2026) | mit |
| Our decision | **excluded (TSAI ruling, 30 Oct): translation of GPT-4-generated Orca-Math (OpenAI terms)** |
| Gated | False |
| Last modified | 2024-03-18 |
| Downloads (30 d) | 27 |
| Languages | not stated |
| Task categories | not stated |
| Size category | not stated |
| Total rows (all configs) | 200000 |

## Caveats and open checks

- Hindi translation of Orca-Math, which is GPT-4 generated (from my knowledge of Orca-Math, not stated on this card): OpenAI terms may apply.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Benchmark overlap

- Risk: **Medium** (evidence: not stated)
- Hindi translation of Orca-Math; the seed problems are not stated on this card. Check against GSM8K/MATH test via the English source.
- Card-level check only, no text-overlap test run. See `../BENCHMARK_OVERLAP.md`.

## Configs and splits

- `default` / `train`

## Fields (from the sampled config)

- `Question`: {"dtype": "string", "_type": "Value"}
- `Answer`: {"dtype": "string", "_type": "Value"}

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
