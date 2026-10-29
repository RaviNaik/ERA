# ai4bharat/indic-align

Source: https://huggingface.co/datasets/ai4bharat/indic-align

## Summary

| Field | Value |
|---|---|
| Category | indic |
| Training stage | SFT |
| Licence tag (Hub, read 29 Oct 2026) | cc-by-4.0 |
| Our decision | **use, check** |
| Gated | False |
| Last modified | 2024-07-25 |
| Downloads (30 d) | 2052 |
| Languages | as, bn, gu, en, hi, kn, ml, mr, ne, or, pa, sa, ta, te, ur |
| Task categories | text-generation |
| Size category | 100M<n<1B |
| Total rows (all configs) | 97419884 |

## Caveats and open checks

- 14 Indic languages. Many parts have Llama-2-70B-Chat responses (Llama 2 licence) and translated Dolly, OpenAssistant, WikiHow and ShareGPT prompts.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Configs and splits

- `Anudesh` / `train`
- `Dolly_T` / `train`
- `HHRLHF_T` / `train`
- `Indic_ShareLlama` / `train`
- `IndoWordNet` / `train`
- `OpenAssistant_T` / `train`
- `Toxic_Matrix` / `train`
- `WikiHow` / `train`
- `Wiki_Chat` / `train`
- `Wiki_Conv` / `train`

## Fields (from the sampled config)

- `id`: {"dtype": "string", "_type": "Value"}
- `interactions`: {"feature": {"feature": {"dtype": "string", "_type": "Value"}, "_type": "List"}, "_type": "List"}
- `num_turns`: {"dtype": "int64", "_type": "Value"}

## Sample

`sample_20.jsonl` holds the first 20 rows from `Anudesh/train`. Each line is one row; `_config` and `_split` say where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> # IndicAlign
> 
> A diverse collection of Instruction and Toxic alignment datasets for 14 Indic Languages. The collection comprises of:
> - **IndicAlign - Instruct**
>   - Indic-ShareLlama
>   - Dolly-T
>   - OpenAssistant-T
>   - WikiHow
>   - IndoWordNet
>   - Anudesh
>   - Wiki-Conv
>   - Wiki-Chat
> - **IndicAlign - Toxic**
>   - HHRLHF-T
>   - Toxic-Matrix
>  
> We use IndicTrans2 ([Gala et al., 2023](https://openreview.net/forum?id=vfT4YuzAYA)) for the translation of the datasets. 
> We recommend the readers to check out our paper [on Arxiv](https://arxiv.org/abs/2403.06350) for detailed information on the curation process of these collections.
> 
> ## Dataset Summaries
> 
> **IndicShareLlama**- Collection of first user prompts from [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) along with responses from [Llama2-70B-Chat](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) model.
> 
> **Dolly-T**- Translated and Romanised version of [Dolly-15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
> 
> **OpenAssistant-T**- Translated and Romanised version of [OpenAssistant v1](https://huggingface.co/datasets/OpenAssistant/oasst1)
> 
> **WikiHow** - Translated and Romanised version of [WikiHow](https://huggingface.co/datasets/ai4bharat/indic-instruct-data-v0.1)
> 
> **IndoWordNet**- Novel dataset created by converting the entried of [IndoWordNet](https://pypi.org/project/pyiwn/) to Instruction-Response pairs in 18 Indic languages.
> 
> **Anudesh**- A crowd-sourced collection of prompts accompanied by responses generated from the Llama2-70B-Chat model.
> 
> **Wiki-Conv**- Collection of short, to-the-point conversations on Wikipedia passages and Wiki-Infoboxes created using Llama2-70B-Chat model.
> 
> **Wiki-Chat**- Collection of long, open conversations on Wikipedia passages, create
> ...[card continues at the source link]
