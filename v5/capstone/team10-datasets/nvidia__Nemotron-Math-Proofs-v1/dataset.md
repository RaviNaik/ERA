# nvidia/Nemotron-Math-Proofs-v1

Source: https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v1

## Summary

| Field | Value |
|---|---|
| Category | math |
| Training stage | SFT |
| Licence tag (Hub, read 29 Oct 2026) | cc-by-sa-4.0 |
| Our decision | **allowed (CC-BY-SA, attribution needed)** (TSAI update, 30 Oct). Generator: gpt-oss-120b (open weights, Apache-2.0), confirmed allowed by TSAI on 30 Oct (Math-Proofs also uses Goedel-Prover-V2-32B for proofs) |
| Gated | False |
| Last modified | 2026-01-05 |
| Downloads (30 d) | 1293 |
| Languages | en |
| Task categories | text-generation |
| Size category | not stated |
| Total rows (all configs) | 243314 |

## Caveats and open checks

- cc-by-sa-4.0 (share-alike): needs a team decision.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Benchmark overlap

- Risk: **Medium** (evidence: not stated)
- AoPS, Math StackExchange and MathOverflow problems; no decontamination statement found. Check against competition benchmarks.
- Card-level check only, no text-overlap test run. See `../BENCHMARK_OVERLAP.md`.

## Configs and splits

- `default` / `lean`

## Fields (from the sampled config)

- `problem`: {"dtype": "string", "_type": "Value"}
- `source`: {"dtype": "string", "_type": "Value"}
- `formal_statement`: {"dtype": "string", "_type": "Value"}
- `lean_header`: {"dtype": "string", "_type": "Value"}
- `url`: {"dtype": "string", "_type": "Value"}
- `user_name`: {"dtype": "string", "_type": "Value"}
- `user_url`: {"dtype": "string", "_type": "Value"}
- `sft_line_number`: {"dtype": "int64", "_type": "Value"}
- `messages`: {"feature": {"role": {"dtype": "string", "_type": "Value"}, "content": {"dtype": "string", "_type": "Value"}, "reasoning
- `uuid`: {"dtype": "string", "_type": "Value"}
- `used_in`: {"feature": {"dtype": "string", "_type": "Value"}, "_type": "List"}
- `tools`: {"feature": {"dtype": "null", "_type": "Value"}, "_type": "List"}
- `license`: {"dtype": "string", "_type": "Value"}

## Sample

`sample_20.jsonl` holds the first 20 rows from `default/lean`. Each line is one row; `_config` and `_split` say where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> # Nemotron-Math-Proofs-v1
> 
> Paper: [Nemotron-Math: Efficient Long-Context Distillation of Mathematical Reasoning from Multi-Mode Supervision](https://huggingface.co/papers/2512.15489)  
> Code: [https://github.com/NVIDIA/NeMo-Skills](https://github.com/NVIDIA/NeMo-Skills)  
> Documentation: [Nemotron-MathProofs-v1 documentation](https://nvidia-nemo.github.io/Skills/releases/nemotronmathproofs/)  
> 
> ## Dataset Description:
> 
> Nemotron-Math-Proofs-v1 is a large-scale mathematical reasoning dataset containing ~580k natural language proof problems, ~550k formalizations into theorem statements in Lean 4, and ~900k model-generated reasoning trajectories culminating in Lean 4 proofs. The dataset integrates human-authored problems with systematically generated formalizations and solution traces.
> 
> Each natural language problem is formalized by [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) into Lean 4 theorem statements with placeholders for the proof. Reasoning traces and proofs are attempted by [Goedel-Prover-V2-32B](https://huggingface.co/Goedel-LM/Goedel-Prover-V2-32B). Proofs are verified by the Lean 4 compiler, with errors passed back and self-correction attempted.
> 
> This dataset is ready for commercial use.  
>   
> 
> ## Dataset Owner(s):
> NVIDIA Corporation
> 
> ## Dataset Creation Date:
> Created on: Dec 3, 2025  
> Last Modified on: Dec 15, 2025
> 
> ## License/Terms of Use: 
> This dataset is governed by the [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).
> 
> ## Intended Usage:
> This dataset is intended for:
> * Training LLMs to perform structured mathematical reasoning, in particular generation of Lean 4 formal proofs.
> 
> * Building long-context or multi-trajectory reasoning systems
> 
> * Research on reasoning
> ...[card continues at the source link]
