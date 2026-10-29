# open-thoughts/OpenThoughts3-1.2M

Source: https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M

## Summary

| Field | Value |
|---|---|
| Category | math, science, coding |
| Training stage | SFT |
| Licence tag (Hub, read 29 Oct 2026) | apache-2.0 |
| Our decision | **use, check** |
| Gated | False |
| Last modified | 2025-06-09 |
| Downloads (30 d) | 24946 |
| Languages | not stated |
| Task categories | text-generation |
| Size category | not stated |
| Total rows (all configs) | 1200000 |

## Caveats and open checks

- Reasoning traces on math, code and science; teacher model terms.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Configs and splits

- `default` / `train`

## Fields (from the sampled config)

- `difficulty`: {"dtype": "int64", "_type": "Value"}
- `source`: {"dtype": "string", "_type": "Value"}
- `domain`: {"dtype": "string", "_type": "Value"}
- `conversations`: {"feature": {"from": {"dtype": "string", "_type": "Value"}, "value": {"dtype": "string", "_type": "Value"}}, "_type": "L

## Sample

`sample_20.jsonl` holds the first 20 rows from `default/train`. Each line is one row; `_config` and `_split` say where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> paper |
> dataset |
> model
> 
> 
> > [!NOTE]
> > We have released a paper for OpenThoughts! See our paper [here](https://arxiv.org/abs/2506.04178).
> 
> 
>  
> 
> 
> 
> # OpenThoughts3-1.2M
> 
> ## Dataset Description
> 
> - **Homepage:** https://www.open-thoughts.ai/
> - **Repository:** https://github.com/open-thoughts/open-thoughts
> 
> Open-source state-of-the-art reasoning dataset with 1.2M rows. 🚀
> 
> [OpenThoughts3-1.2M](https://arxiv.org/abs/2506.04178) is the third iteration in our line of [OpenThoughts](https://openthoughts.ai) datasets, building on our previous [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) and [OpenThoughts2-1M](https://huggingface.co/datasets/open-thoughts/OpenThoughts2-1M).
> This time around, we scale even further and generate our dataset in a much more systematic way -- [OpenThoughts3-1.2M](https://arxiv.org/abs/2506.04178) is the result of a rigorous experimental pipeline, that ablates over design choices surrounding question sourcing and selection, as well as answer generation. 
> The final dataset consists of 850,000 math questions, 250,000 code questions, and 100,000 science questions. Annotations are done with QwQ-32B.
> 
> This dataset was used to train [OpenThinker3-7B](https://huggingface.co/open-thoughts/OpenThinker3-7B), which beats **all** similarly sized open-data reasoning models.
> 
> See our [paper](https://arxiv.org/abs/2506.04178) and [blog post](https://www.open-thoughts.ai/blog/ot3) for more details. OpenThinker3-32B to follow! 👀
> 
> 
> # OpenThinker3-7B Model Performance
> 
> Our [OpenThinker3-7B](https://huggingface.co/open-thoughts/OpenThinker3-7B) model trained on this dataset is the state-of-the-art open-data reasoning dataset at its scale. 
> 
> We conduct all our evaluations using [Evalchemy](https://github.com/mlfoundations/evalchemy). In t
> ...[card continues at the source link]
