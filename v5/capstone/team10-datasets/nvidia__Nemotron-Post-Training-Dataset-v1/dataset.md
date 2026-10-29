# nvidia/Nemotron-Post-Training-Dataset-v1

Source: https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1

## Summary

| Field | Value |
|---|---|
| Category | math, science, coding |
| Training stage | SFT |
| Licence tag (Hub, read 29 Oct 2026) | cc-by-4.0 |
| Our decision | **use, check** |
| Gated | False |
| Last modified | 2025-08-25 |
| Downloads (30 d) | 18790 |
| Languages | not stated |
| Task categories | not stated |
| Size category | not stated |
| Total rows (all configs) | 25659642 |

## Caveats and open checks

- Math, code, STEM and chat splits; the sample takes math and stem. Generator models carry their own licences.
- A Hub licence tag is not legal clearance. Before the licence-table decision, check the dataset card, the licences of the source data, and the terms of any model that generated the responses.

## Configs and splits

- `default` / `chat`
- `default` / `code`
- `default` / `math`
- `default` / `stem`
- `default` / `tool_calling`

## Fields (from the sampled config)

- `uuid`: {"dtype": "string", "_type": "Value"}
- `license`: {"dtype": "string", "_type": "Value"}
- `generator`: {"dtype": "string", "_type": "Value"}
- `version`: {"dtype": "string", "_type": "Value"}
- `category`: {"dtype": "string", "_type": "Value"}
- `reasoning`: {"dtype": "string", "_type": "Value"}
- `messages`: {"feature": {"role": {"dtype": "string", "_type": "Value"}, "content": {"dtype": "string", "_type": "Value"}, "tool_call
- `metadata`: {"dtype": "string", "_type": "Value"}

## Sample

`sample_20.jsonl` holds the first 20 rows from `default/math`, `default/stem`. Each line is one row; `_config` and `_split` say where it came from. Strings longer than 3000 characters are cut and marked `...[truncated N chars]`.

Quick look:

```python
import json
rows = [json.loads(l) for l in open('sample_20.jsonl', encoding='utf-8')]
print(len(rows), list(rows[0]))
```

## Dataset card excerpt (start of the card, for checking)

> # Nemotron-Post-Training-Dataset-v1 Release
> This dataset is a compilation of SFT data that supports improvements of math, code, stem, general reasoning, and tool calling capabilities of the original Llama instruct model [Llama-3.3-Nemotron-Super-49B-v1.5](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5). 
> 
> Llama-3.3-Nemotron-Super-49B-v1.5 is an LLM which is a derivative of [Meta Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) (AKA the *reference model*).
> 
> Llama-3.3-Nemotron-Super-49B-v1.5 offers a great tradeoff between model accuracy and efficiency. Efficiency (throughput) directly translates to savings. Using a novel Neural Architecture Search (NAS) approach, we greatly reduce the model’s memory footprint and enable larger workloads. This NAS approach enables the selection of a desired point in the accuracy-efficiency tradeoff. The model supports a context length of 128K.
> 
> This dataset release represents a significant move forward in openness and transparency in model development and improvement. By releasing the complete training set, in addition to the training technique, tools and final model weights, NVIDIA supports both the re-creation and the improvement of our approach.
> 
> This dataset is ready for commercial/non-commercial use.
> 
> ## Data Distribution
> 
> | Category      | Value          |
> |---------------|---------------:|
> | chat          |      746,622   |
> | code          |    1,896,395   |
> | math          |    2,044,407   |
> | stem          |   20,662,167   |
> | tool_calling  |      310,051   |
> | **Total**     | **25,659,642** |
> 
> 
> ## Filtering the data
> Users can download subsets of the data based on the metadata schema described above. Example script for downloading code and math as follows:
> ```
> from datasets import
> ...[card continues at the source link]
