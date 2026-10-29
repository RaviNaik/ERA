# open-r1/codeforces

Source: https://huggingface.co/datasets/open-r1/codeforces

## Summary

| Field | Value |
|---|---|
| Category | coding |
| Training stage | RL |
| Licence tag (Hub, read 29 Oct 2026) | cc-by-4.0 |
| Our decision | **use, check** |
| Gated | False |
| Last modified | 2025-05-19 |
| Downloads (30 d) | 51236 |
| Languages | not stated |
| Task categories | not stated |
| Size category | not stated |
| Total rows (all configs) | 29286 |

## Caveats and open checks

- Verifiable Codeforces problems with tests. The card header says cc-by-4.0 while the card body says ODC-By 4.0 (both permissive). Problem statements are Codeforces content: confirm that redistribution for training is acceptable. Check overlap with other Codeforces-derived sets.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Benchmark overlap

- Risk: **Low** (evidence: card)
- Has a `test` split, and Codeforces problems are also used by LiveCodeBench and CodeElo. Use `train` only.
- Card-level check only, no text-overlap test run. See `../BENCHMARK_OVERLAP.md`.

## Configs and splits

- `default` / `train`
- `default` / `test`
- `verifiable` / `train`
- `verifiable` / `test`
- `verifiable-prompts` / `train`
- `verifiable-prompts` / `test`

## Fields (from the sampled config)

- `id`: {"dtype": "string", "_type": "Value"}
- `aliases`: {"feature": {"dtype": "string", "_type": "Value"}, "_type": "List"}
- `contest_id`: {"dtype": "string", "_type": "Value"}
- `contest_name`: {"dtype": "string", "_type": "Value"}
- `contest_type`: {"dtype": "string", "_type": "Value"}
- `contest_start`: {"dtype": "int64", "_type": "Value"}
- `contest_start_year`: {"dtype": "int64", "_type": "Value"}
- `index`: {"dtype": "string", "_type": "Value"}
- `time_limit`: {"dtype": "float64", "_type": "Value"}
- `memory_limit`: {"dtype": "float64", "_type": "Value"}
- `title`: {"dtype": "string", "_type": "Value"}
- `description`: {"dtype": "string", "_type": "Value"}
- `input_format`: {"dtype": "string", "_type": "Value"}
- `output_format`: {"dtype": "string", "_type": "Value"}
- `interaction_format`: {"dtype": "string", "_type": "Value"}
- `note`: {"dtype": "string", "_type": "Value"}
- `examples`: {"feature": {"input": {"dtype": "string", "_type": "Value"}, "output": {"dtype": "string", "_type": "Value"}}, "_type": 
- `editorial`: {"dtype": "string", "_type": "Value"}
- `rating`: {"dtype": "int64", "_type": "Value"}
- `tags`: {"feature": {"dtype": "string", "_type": "Value"}, "_type": "List"}
- `testset_size`: {"dtype": "int64", "_type": "Value"}
- `official_tests`: {"feature": {"input": {"dtype": "string", "_type": "Value"}, "output": {"dtype": "string", "_type": "Value"}}, "_type": 
- `official_tests_complete`: {"dtype": "bool", "_type": "Value"}
- `input_mode`: {"dtype": "string", "_type": "Value"}
- `generated_checker`: {"dtype": "string", "_type": "Value"}
- `executable`: {"dtype": "bool", "_type": "Value"}
- `generated_tests`: {"dtype": "int64", "_type": "Value"}

## Sample

`sample_20.jsonl` holds the first 20 rows from `default/train`. Each line is one row; `_config` and `_split` say where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> # Dataset Card for CodeForces
> 
> ## Dataset description
> [CodeForces](https://codeforces.com/) is one of the most popular websites among competitive programmers, hosting regular contests where participants must solve challenging algorithmic optimization problems. The challenging nature of these problems makes them an interesting dataset to improve and test models’ code reasoning capabilities. This dataset includes more than **10k unique problems** covering the very first contests all the way to 2025. Additionally, we generated and extensively validated custom checker code for problems with multiple solutions, and generated additional challenging test cases.
> 
> ## Why most datasets are actually not "verifiable"
> 
> Competitive programming problems require solutions to solve any valid test case within the input constraints within certain time and memory limits. Usually, the most challenging test cases requiring truly optimized (and correct) solutions consist of **very large test inputs**.
> 
> While contest platforms like CodeForces let users see test cases, they **truncate** them to at most \~400 characters. This means that the problems on "verifiable" datasets often only contain the most simple, trivial test cases that can be solved with an easy brute force solution.
> Additionally, many problems allow for multiple correct answers, and therefore require a special program, a "checker" to validate if a user submitted answer is correct (~30% of problems).
> Fully verifable problems thus require two things: being able to properly validate test cases (checkers), and having challenging test cases that ensure solutions are correct.
> 
> ## What sets this dataset apart
> 
> Besides covering more problems than previous efforts, this dataset:
> 
> - **editorials**: includes *editorials*, an explanation, wri
> ...[card continues at the source link]
