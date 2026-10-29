# nvidia/OpenMathInstruct-2

Source: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2

## Summary

| Field | Value |
|---|---|
| Category | math |
| Training stage | SFT |
| Licence tag (Hub, read 29 Oct 2026) | cc-by-4.0 |
| Our decision | **use, check** |
| Gated | False |
| Last modified | 2024-11-25 |
| Downloads (30 d) | 105403 |
| Languages | en |
| Task categories | question-answering, text-generation |
| Size category | 10M<n<100M |
| Total rows (all configs) | 21972791 |

## Caveats and open checks

- Generated with Llama-3.1-405B: check the Llama licence. Also in Amit's merged licence table.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Configs and splits

- `default` / `train`
- `default` / `train_1M`
- `default` / `train_2M`
- `default` / `train_5M`

## Fields (from the sampled config)

- `problem`: {"dtype": "string", "_type": "Value"}
- `generated_solution`: {"dtype": "string", "_type": "Value"}
- `expected_answer`: {"dtype": "string", "_type": "Value"}
- `problem_source`: {"dtype": "string", "_type": "Value"}

## Sample

`sample_20.jsonl` holds the first 20 rows from `default/train`. Each line is one row; `_config` and `_split` say where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> # OpenMathInstruct-2
> 
> OpenMathInstruct-2 is a math instruction tuning dataset with 14M problem-solution pairs 
> generated using the [Llama3.1-405B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct) model.
> 
> The training set problems of [GSM8K](https://github.com/openai/grade-school-math)
> and [MATH](https://github.com/hendrycks/math) are used for constructing the dataset in the following ways: 
> - *Solution augmentation*: Generating chain-of-thought solutions for training set problems in GSM8K and MATH. 
> - *Problem-Solution augmentation*: Generating new problems, followed by solutions for these new problems.      
> 
> 
>  
> 
> 
> OpenMathInstruct-2 dataset contains the following fields:
> 
> - **problem**: Original problem from either the GSM8K or MATH training set or augmented problem from these training sets.
> - **generated_solution**: Synthetically generated solution.
> - **expected_answer**: For problems in the training set, it is the ground-truth answer provided in the datasets. **For augmented problems, it is the majority-voting answer.**
> - **problem_source**: Whether the problem is taken directly from GSM8K or MATH or is an augmented version derived from either dataset.
> 
>   
> 
>  
> 
> 
> We also release the 1M, 2M, and 5M, *fair-downsampled* versions of the entire training set corresponding to points in the above scaling plot. 
> These splits are referred to as **train_1M**, **train_2M**, and **train_5M**. 
> To use these subsets, just specify one of these subsets as split while downloading the data:
> ```python
> from datasets import load_dataset
> 
> # Download only the 1M training split
> dataset = load_dataset('nvidia/OpenMathInstruct-2', split='train_1M', streaming=True)
> ```  
> 
> To download the entire training set and to convert it into the jsonl format, use the following code snippet
> ...[card continues at the source link]
