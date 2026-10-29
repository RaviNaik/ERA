# CohereLabs/aya_dataset

Source: https://huggingface.co/datasets/CohereLabs/aya_dataset

## Summary

| Field | Value |
|---|---|
| Category | indic |
| Training stage | SFT |
| Licence tag (Hub, read 29 Oct 2026) | apache-2.0 |
| Our decision | **use** |
| Gated | False |
| Last modified | 2025-04-15 |
| Downloads (30 d) | 21476 |
| Languages | amh, arb, ary, ars, acq, arz, apc, ben, ceb, dan, deu, ell, eng, eus, fil, fin, fra, gle, guj, hat, hau, hin, hun, ibo, ind, ita, jav, jpn, kan, kir, kor, kur, lit, mal, mar, mlg, msa, mya, nep, nld, nso, nya, pan, pes, pol, por, pus, rus, sin, sna, snd, som, spa, sqi, srp, sun, swa, swe, tam, tel, tha, tur, ukr, urd, vie, wol, xho, yor, zho, zul |
| Task categories | other |
| Size category | 100K<n<1M |
| Total rows (all configs) | 205568 |

## Caveats and open checks

- Human-written prompts and completions, Apache-2.0. The sample is filtered to Tamil and Marathi rows (Hindi rows could not be fetched); use the `language` column to select Hindi, Telugu and others from the full set.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Configs and splits

- `default` / `test`
- `default` / `train`
- `demographics` / `train`

## Fields (from the sampled config)

- `inputs`: {"dtype": "string", "_type": "Value"}
- `targets`: {"dtype": "string", "_type": "Value"}
- `language`: {"dtype": "string", "_type": "Value"}
- `language_code`: {"dtype": "string", "_type": "Value"}
- `annotation_type`: {"dtype": "string", "_type": "Value"}
- `user_id`: {"dtype": "string", "_type": "Value"}

## Sample

`sample_20.jsonl` holds 13 rows filtered on the `language` column (Hindi, Tamil, Marathi where available). Each line is one row.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> ![Aya Header](https://huggingface.co/datasets/CohereLabs/aya_dataset/resolve/main/aya_header.png)
> # Dataset Summary
> The `Aya Dataset` is a multilingual instruction fine-tuning dataset curated by an open-science community via [Aya Annotation Platform](https://aya.for.ai/) from Cohere Labs. The dataset contains a total of 204k human-annotated prompt-completion pairs along with the demographics data of the annotators.
> This dataset can be used to train, finetune, and evaluate multilingual LLMs.
> 
> - **Curated by:** Contributors of [Aya Open Science Intiative](https://aya.for.ai/).
> - **Language(s):** 65 languages (71 including dialects & scripts).
> - **License:** [Apache 2.0](https://opensource.org/license/apache-2-0)
> - **Aya Datasets Family:**
> 
>   | Name | Explanation |
>   |------|--------------|
>   | [aya_dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset) | Human-annotated multilingual instruction finetuning dataset, comprising over 204K instances across 65 languages. |
>   | [aya_collection](https://huggingface.co/datasets/CohereLabs/aya_collection) | Created by applying instruction-style templates from fluent speakers to 44 datasets, including translations of 19 instruction-style datasets into 101 languages, providing 513M instances for various tasks.|
>   | [aya_collection_language_split](https://huggingface.co/datasets/CohereLabs/aya_collection_language_split) | Aya Collection structured based on language level subsets. |
>   | [aya_evaluation_suite](https://huggingface.co/datasets/CohereLabs/aya_evaluation_suite) | A diverse evaluation set for multilingual open-ended generation, featuring 250 culturally grounded prompts in 7 languages, 200 translated prompts in 24 languages, and human-edited versions selected for cross-cultural relevance from English Dolly in 6 lang
> ...[card continues at the source link]
