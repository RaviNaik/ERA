# CohereLabs/aya_collection

Source: https://huggingface.co/datasets/CohereLabs/aya_collection

## Summary

| Field | Value |
|---|---|
| Category | indic |
| Training stage | SFT |
| Licence tag (Hub, read 29 Oct 2026) | apache-2.0 |
| Our decision | **use, check** |
| Gated | False |
| Last modified | 2025-04-15 |
| Downloads (30 d) | 12363 |
| Languages | ace, afr, amh, ara, aze, ban, bbc, bel, bem, ben, bjn, bul, cat, ceb, ces, cym, dan, deu, ell, eng, epo, est, eus, fil, fin, fon, fra, gla, gle, glg, guj, hat, hau, heb, hin, hrv, hun, hye, ibo, ind, isl, ita, jav, jpn, kan, kas, kat, kau, kaz, khm, kin, kir, kor, kur, lao, lav, lij, lit, ltz, mad, mal, man, mar, min, mkd, mlg, mlt, mon, mri, msa, mya, nep, nij, nld, nor, nso, nya, pan, pes, pol, por, pus, ron, rus, sin, slk, slv, smo, sna, snd, som, sot, spa, sqi, srp, sun, swa, swe, tam, taq, tel, tgk, tha, tur, twi, ukr, urd, uzb, vie, wol, xho, yid, yor, zho, zul |
| Task categories | text-classification, summarization, translation |
| Size category | 100M<n<1B |
| Total rows (all configs) | 513755590 |

## Caveats and open checks

- Templated tasks in 114 languages; the card does not state source-dataset or model terms.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Configs and splits

- `aya_dataset` / `train`
- `templated_afriqa` / `test`
- `templated_afriqa` / `train`
- `templated_afriqa` / `validation`
- `templated_afrisenti` / `test`
- `templated_afrisenti` / `train`
- `templated_afrisenti` / `validation`
- `templated_amharic_qa` / `test`
- `templated_amharic_qa` / `train`
- `templated_amharic_qa` / `validation`
- `templated_armenian_instruct` / `test`
- `templated_armenian_instruct` / `train`
- `templated_bengali_news` / `train`
- `templated_dutch_imdb` / `test`
- `templated_dutch_imdb` / `train`
- `templated_hindi_headline` / `test`
- `templated_hindi_headline` / `train`
- `templated_hindi_news` / `test`
- `templated_hindi_news` / `train`
- `templated_indic_paraphrase` / `train`
- `templated_indic_sentiment` / `train`
- `templated_indo_stories` / `train`
- `templated_japanese_instruct` / `train`
- `templated_joke_explaination` / `train`
- `templated_ligurian_news` / `validation`
- `templated_ligurian_news` / `test`
- `templated_ligurian_news` / `train`
- `templated_masakhanews` / `test`
- `templated_masakhanews` / `train`
- `templated_masakhanews` / `validation`
- `templated_mintaka` / `test`
- `templated_mintaka` / `train`
- `templated_mintaka` / `validation`
- `templated_ntx_llm` / `train`
- `templated_nusax_senti` / `test`
- `templated_nusax_senti` / `train`
- `templated_nusax_senti` / `validation`
- `templated_persian_farstail` / `test`
- `templated_persian_farstail` / `train`
- `templated_persian_farstail` / `validation`
- ... and 86 more

## Fields (from the sampled config)

- `id`: {"dtype": "int64", "_type": "Value"}
- `inputs`: {"dtype": "string", "_type": "Value"}
- `targets`: {"dtype": "string", "_type": "Value"}
- `dataset_name`: {"dtype": "string", "_type": "Value"}
- `sub_dataset_name`: {"dtype": "string", "_type": "Value"}
- `task_type`: {"dtype": "string", "_type": "Value"}
- `template_id`: {"dtype": "int64", "_type": "Value"}
- `language`: {"dtype": "string", "_type": "Value"}
- `split`: {"dtype": "string", "_type": "Value"}
- `script`: {"dtype": "string", "_type": "Value"}

## Sample

`sample_20.jsonl` holds the first 20 rows from `templated_hindi_headline/train`, `templated_tamil_stories/train`. Each line is one row; `_config` and `_split` say where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> ![Aya Header](https://huggingface.co/datasets/CohereLabs/aya_collection/resolve/main/aya_header.png)
> 
> ****This dataset is uploaded in two places: here and additionally [here](https://huggingface.co/datasets/CohereLabs/aya_collection_language_split) as 'Aya Collection Language Split.' These datasets are identical in content but differ in structure of upload. This dataset is structured by folders split according to dataset name. The version [here](https://huggingface.co/datasets/CohereLabs/aya_collection_language_split) instead divides the Aya collection into folders split by language. We recommend you use the language split version if you are only interested in downloading data for a single or smaller set of languages, and this version if you want to download dataset according to data source or the entire collection.**** 
> 
> # Dataset Summary
> The Aya Collection is a massive multilingual collection consisting of 513 million instances of prompts and completions covering a wide range of tasks.
> This collection incorporates instruction-style templates from fluent speakers and applies them to a curated list of datasets, as well as translations of instruction-style datasets into 101 languages. Aya Dataset, a human-curated multilingual instruction and response dataset, is also part of this collection. See our paper for more details regarding the collection.   
> 
> - **Curated by:** Contributors of [Aya Open Science Intiative](https://cohere.com/research/aya)
> - **Language(s):** 115 languages
> - **License:** [Apache 2.0](https://opensource.org/license/apache-2-0)
>   
> - **Aya Datasets Family:**
> | Name | Explanation |
> |------|--------------|
> | [aya_dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset) | Human-annotated multilingual instruction finetuning dataset, comprising over
> ...[card continues at the source link]
