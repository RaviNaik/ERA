# bigcode/commitpackft

Source: https://huggingface.co/datasets/bigcode/commitpackft

## Summary

| Field | Value |
|---|---|
| Category | coding |
| Training stage | SFT |
| Licence tag (Hub, read 29 Oct 2026) | mit |
| Our decision | **use, check** |
| Gated | False |
| Last modified | 2023-08-20 |
| Downloads (30 d) | 33186 |
| Languages | code |
| Task categories | not stated |
| Size category | not stated |
| Total rows (all configs) | unknown |

## Caveats and open checks

- Commit data; repository licences were filtered upstream. Also in Amit's merged licence table.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Benchmark overlap

- Risk: **Low** (evidence: not stated)
- GitHub commits. The same group released HumanEvalPack, so a quick check is sensible; overlap is unlikely but untested.
- Card-level check only, no text-overlap test run. See `../BENCHMARK_OVERLAP.md`.

## Configs and splits

Not available through the datasets API.

## Fields (from the sampled config)

Not available.

## Sample

`sample_20.jsonl` holds the first 20 rows read directly from the raw file(s) `data/python/data.jsonl`, `data/go/data.jsonl` (the datasets API had no preview for this dataset). Each line is one row; `_file` says where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> ![Octopack](https://github.com/bigcode-project/octopack/blob/31f3320f098703c7910e43492c39366eeea68d83/banner.png?raw=true)
> 
> # Dataset Card for CommitPackFT
> 
> ## Table of Contents
> - [Table of Contents](#table-of-contents)
> - [Dataset Description](#dataset-description)
>   - [Dataset Summary](#dataset-summary)
>   - [Languages](#languages)
> - [Dataset Structure](#dataset-structure)
>   - [Data Instances](#data-instances)
>   - [Data Fields](#data-fields)
>   - [Data Splits](#data-splits)
> - [Dataset Creation](#dataset-creation)
>   - [Curation Rationale](#curation-rationale)
>   - [Source Data](#source-data)
>   - [Annotations](#annotations)
> - [Additional Information](#additional-information)
>   - [Licensing Information](#licensing-information)
>   - [Citation Information](#citation-information)
>   - [Contributions](#contributions)
> 
> ## Dataset Description
> 
> - **Repository:** https://github.com/bigcode-project/octopack
> - **Paper:** [OctoPack: Instruction Tuning Code Large Language Models](https://arxiv.org/abs/2308.07124)
> - **Point of Contact:** [Niklas Muennighoff](mailto:n.muennighoff@gmail.com)
> 
> ### Dataset Summary
> 
> > CommitPackFT is a 2GB filtered version of [CommitPack](https://huggingface.co/datasets/bigcode/commitpack) to contain only high-quality commit messages that resemble natural language instructions.
> > 
> - **Creation:** The dataset can be recreated using instructions available [here](https://github.com/bigcode-project/octopack).
> - **Languages:** 277
> - **OctoPack🐙🎒:**
> 
> 
> 
> Data 
> CommitPack
> 4TB of GitHub commits across 350 programming languages
> 
> 
>  
> CommitPackFT
> Filtered version of CommitPack for high-quality commit messages that resemble instructions
> 
> 
> Model 
> OctoCoder
> StarCoder (16B parameters) instruction tuned on CommitPackFT + OASST
> 
> 
>  
> OctoGeeX
> CodeGeeX2 (6B parameters) instruction t
> ...[card continues at the source link]
