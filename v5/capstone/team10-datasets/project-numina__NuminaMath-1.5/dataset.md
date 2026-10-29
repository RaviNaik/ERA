# project-numina/NuminaMath-1.5

Source: https://huggingface.co/datasets/project-numina/NuminaMath-1.5

## Summary

| Field | Value |
|---|---|
| Category | math |
| Training stage | SFT |
| Licence tag (Hub, read 29 Oct 2026) | apache-2.0 |
| Our decision | **use, check** |
| Gated | False |
| Last modified | 2026-01-29 |
| Downloads (30 d) | 55478 |
| Languages | en |
| Task categories | text-generation |
| Size category | not stated |
| Total rows (all configs) | 896215 |

## Caveats and open checks

- Renamed from AI-MO/NuminaMath-1.5. Problems were collected from exam PDFs and maths forums; the Apache tag covers the dataset, not necessarily the underlying problem text.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Benchmark overlap

- Risk: **High** (evidence: card)
- Sources include amc_aime (5,872 rows) and olympiads. No decontamination is stated. Decontaminate against AIME/AMC and MATH-500 before use.
- Card-level check only, no text-overlap test run. See `../BENCHMARK_OVERLAP.md`.

## Configs and splits

- `default` / `train`

## Fields (from the sampled config)

- `problem`: {"dtype": "string", "_type": "Value"}
- `solution`: {"dtype": "string", "_type": "Value"}
- `answer`: {"dtype": "string", "_type": "Value"}
- `problem_type`: {"dtype": "string", "_type": "Value"}
- `question_type`: {"dtype": "string", "_type": "Value"}
- `problem_is_valid`: {"dtype": "string", "_type": "Value"}
- `solution_is_valid`: {"dtype": "string", "_type": "Value"}
- `source`: {"dtype": "string", "_type": "Value"}
- `synthetic`: {"dtype": "bool", "_type": "Value"}

## Sample

`sample_20.jsonl` holds the first 20 rows from `default/train`. Each line is one row; `_config` and `_split` say where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> # Dataset Card for NuminaMath 1.5
> 
> ## Dataset Description
> 
> - **Homepage:** https://projectnumina.ai
> - **Repository:** 
> - **Paper:** https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf
> - **Leaderboard:** 
> - **Point of Contact:** [Jia Li](jia@projectnumina.ai)
> 
> 
> ### Dataset Summary
> 
> This is the second iteration of the popular [NuminaMath](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) dataset, bringing high quality post-training data for approximately 900k competition-level math problems.  Each solution is formatted in a Chain of Thought (CoT) manner. The sources of the dataset range from Chinese high school math exercises to US and international mathematics olympiad competition problems. The data were primarily collected from online exam paper PDFs and mathematics discussion forums.
> 
> ### What's new?
> 
> #### Problem metadata
> 
> After understanding the importance of verifiable output for each problem, we have added `answer`, `problem_type`, `question_type` metadata for all problems:
> 
> - `answer`: Final answer of the problem when `question_type` is a "math word problem", i.e. a number-valued output. For problems which do not belong to this category, `answer` takes one of the following special values:
>     - `proof`: When the `question_type` is proof
>     - `notfound`: When we cannot find the answer from the `ref_solution`
> - `problem_type`: The mathematical domain of the problem. See `find_problem_type` for more information. Here are the supported types:
>     - Algebra
>     - Geometry
>     - Number Theory
>     - Combinatorics
>     - Calculus
>     - Inequalities
>     - Logic and Puzzles
>     - Other
> - `question_type`: The form or style of the mathematical problem.
>     - multiple-choice question (MCQ)
>     - proof
>     - math-word-problem (problem w
> ...[card continues at the source link]
