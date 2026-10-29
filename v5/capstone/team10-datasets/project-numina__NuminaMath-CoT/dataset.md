# project-numina/NuminaMath-CoT

Source: https://huggingface.co/datasets/project-numina/NuminaMath-CoT

## Summary

| Field | Value |
|---|---|
| Category | math |
| Training stage | SFT |
| Licence tag (Hub, read 29 Oct 2026) | apache-2.0 |
| Our decision | **use, check** |
| Gated | False |
| Last modified | 2024-11-25 |
| Downloads (30 d) | 172546 |
| Languages | en |
| Task categories | text-generation |
| Size category | not stated |
| Total rows (all configs) | 859594 |

## Caveats and open checks

- Renamed from AI-MO/NuminaMath-CoT. The card says solutions were realigned into chain-of-thought format by an LLM but does not name it (my earlier note that it was GPT-4o is not on the card). Problems come from exam PDFs and forums (AoPS, Chinese K-12, olympiads, GSM8K, MATH): the Apache tag covers the dataset, not necessarily the underlying problem text.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Configs and splits

- `default` / `train`
- `default` / `test`

## Fields (from the sampled config)

- `source`: {"dtype": "string", "_type": "Value"}
- `problem`: {"dtype": "string", "_type": "Value"}
- `solution`: {"dtype": "string", "_type": "Value"}
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

> # Dataset Card for NuminaMath CoT
> 
> ## Dataset Description
> 
> - **Homepage:** https://projectnumina.ai
> - **Repository:** https://github.com/project-numina/aimo-progress-prize
> - **Paper:** https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf
> - **Leaderboard:** 
> - **Point of Contact:** [Jia Li](jia@projectnumina.ai)
> 
> 
> ### Dataset Summary
> 
> Approximately 860k math problems, where each solution is formatted in a Chain of Thought (CoT) manner. The sources of the dataset range from Chinese high school math exercises to US and international mathematics olympiad competition problems. The data were primarily collected from online exam paper PDFs and mathematics discussion forums. The processing steps include (a) OCR from the original PDFs, (b) segmentation into problem-solution pairs, (c) Translation into English, (d) realignment to produce a CoT reasoning format, and (e) final answer formatting.
> 
> 
> ### Source breakdown
> 
> | Source | Number of Samples |
> | --- | --- |
> | aops_forum | 30201 |
> | amc_aime | 4072 |
> | cn_k12 | 276591 |
> | gsm8k | 7345 |
> | math | 7478 |
> | olympiads | 150581 |
> | orca_math | 153334 |
> | synthetic_amc | 62111 |
> | synthetic_math | 167895 |
> | **Total** | **859608** |
> 
> ### Licensing Information
> 
> The dataset is available under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
> 
> ### Citation Information
> 
> ```
> @misc{numina_math_datasets,
>   author = {Jia LI and Edward Beeching and Lewis Tunstall and Ben Lipkin and Roman Soletskyi and Shengyi Costa Huang and Kashif Rasul and Longhui Yu and Albert Jiang and Ziju Shen and Zihan Qin and Bin Dong and Li Zhou and Yann Fleureau and Guillaume Lample and Stanislas Polu},
>   title = {NuminaMath},
>   year = {2024},
>   publisher = {Numina},
>   journal = {Hugging Face reposi
> ...[card continues at the source link]
