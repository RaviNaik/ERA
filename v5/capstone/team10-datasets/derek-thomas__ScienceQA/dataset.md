# derek-thomas/ScienceQA

Source: https://huggingface.co/datasets/derek-thomas/ScienceQA

## Summary

| Field | Value |
|---|---|
| Category | science |
| Training stage | SFT (multimodal MCQ) |
| Licence tag (Hub, read 29 Oct 2026) | cc-by-sa-4.0 |
| Our decision | **check** |
| Gated | False |
| Last modified | 2023-02-25 |
| Downloads (30 d) | 35003 |
| Languages | en |
| Task categories | multiple-choice, question-answering, other, visual-question-answering, text-classification |
| Size category | 10K<n<100K |
| Total rows (all configs) | 21208 |

## Caveats and open checks

- cc-by-sa-4.0 (share-alike); multiple-choice with image context.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Configs and splits

- `default` / `train`
- `default` / `validation`
- `default` / `test`

## Fields (from the sampled config)

- `image`: {"_type": "Image"}
- `question`: {"dtype": "string", "_type": "Value"}
- `choices`: {"feature": {"dtype": "string", "_type": "Value"}, "_type": "List"}
- `answer`: {"dtype": "int8", "_type": "Value"}
- `hint`: {"dtype": "string", "_type": "Value"}
- `task`: {"dtype": "string", "_type": "Value"}
- `grade`: {"dtype": "string", "_type": "Value"}
- `subject`: {"dtype": "string", "_type": "Value"}
- `topic`: {"dtype": "string", "_type": "Value"}
- `category`: {"dtype": "string", "_type": "Value"}
- `skill`: {"dtype": "string", "_type": "Value"}
- `lecture`: {"dtype": "string", "_type": "Value"}
- `solution`: {"dtype": "string", "_type": "Value"}

## Sample

`sample_20.jsonl` holds the first 20 rows from `default/train`. Each line is one row; `_config` and `_split` say where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> # Dataset Card Creation Guide
> 
> ## Table of Contents
> - [Dataset Card Creation Guide](#dataset-card-creation-guide)
>   - [Table of Contents](#table-of-contents)
>   - [Dataset Description](#dataset-description)
>     - [Dataset Summary](#dataset-summary)
>     - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
>     - [Languages](#languages)
>   - [Dataset Structure](#dataset-structure)
>     - [Data Instances](#data-instances)
>     - [Data Fields](#data-fields)
>     - [Data Splits](#data-splits)
>   - [Dataset Creation](#dataset-creation)
>     - [Curation Rationale](#curation-rationale)
>     - [Source Data](#source-data)
>       - [Initial Data Collection and Normalization](#initial-data-collection-and-normalization)
>       - [Who are the source language producers?](#who-are-the-source-language-producers)
>     - [Annotations](#annotations)
>       - [Annotation process](#annotation-process)
>       - [Who are the annotators?](#who-are-the-annotators)
>     - [Personal and Sensitive Information](#personal-and-sensitive-information)
>   - [Considerations for Using the Data](#considerations-for-using-the-data)
>     - [Social Impact of Dataset](#social-impact-of-dataset)
>     - [Discussion of Biases](#discussion-of-biases)
>     - [Other Known Limitations](#other-known-limitations)
>   - [Additional Information](#additional-information)
>     - [Dataset Curators](#dataset-curators)
>     - [Licensing Information](#licensing-information)
>     - [Citation Information](#citation-information)
>     - [Contributions](#contributions)
> 
> ## Dataset Description
> 
> - **Homepage:** [https://scienceqa.github.io/index.html#home](https://scienceqa.github.io/index.html#home)
> - **Repository:** [https://github.com/lupantech/ScienceQA](https://github.com/lupantech/ScienceQA)
> - **Paper:** [https://arxiv.org/abs/2209.09
> ...[card continues at the source link]
