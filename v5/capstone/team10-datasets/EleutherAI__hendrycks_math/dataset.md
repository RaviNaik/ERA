# EleutherAI/hendrycks_math

Source: https://huggingface.co/datasets/EleutherAI/hendrycks_math

## Summary

| Field | Value |
|---|---|
| Category | math |
| Training stage | SFT / RL |
| Licence tag (Hub, read 29 Oct 2026) | mit |
| Our decision | **use** |
| Gated | False |
| Last modified | 2025-01-12 |
| Downloads (30 d) | 152356 |
| Languages | not stated |
| Task categories | not stated |
| Size category | not stated |
| Total rows (all configs) | 12500 |

## Caveats and open checks

- Competition math. Also a standard benchmark: keep the test split out of training.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Configs and splits

- `algebra` / `train`
- `algebra` / `test`
- `counting_and_probability` / `train`
- `counting_and_probability` / `test`
- `geometry` / `train`
- `geometry` / `test`
- `intermediate_algebra` / `train`
- `intermediate_algebra` / `test`
- `number_theory` / `train`
- `number_theory` / `test`
- `prealgebra` / `train`
- `prealgebra` / `test`
- `precalculus` / `train`
- `precalculus` / `test`

## Fields (from the sampled config)

- `problem`: {"dtype": "string", "_type": "Value"}
- `level`: {"dtype": "string", "_type": "Value"}
- `type`: {"dtype": "string", "_type": "Value"}
- `solution`: {"dtype": "string", "_type": "Value"}

## Sample

`sample_20.jsonl` holds the first 20 rows from `algebra/train`. Each line is one row; `_config` and `_split` say where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> ## Dataset Summary
> 
> MATH dataset from https://github.com/hendrycks/math
> 
> 
> ### Citation Information
> ```
> @article{hendrycksmath2021,
>   title={Measuring Mathematical Problem Solving With the MATH Dataset},
>   author={Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora and Steven Basart and Eric Tang and Dawn Song and Jacob Steinhardt},
>   journal={NeurIPS},
>   year={2021}
> }
> ```
