# Session 13: Reversible LLM Experiments

Assignment: train a roughly 20M parameter LLM for 50M tokens, repeat with reversible layers, then push the reversible variant to the maximum stable batch size. Report final loss, tokens/s, peak memory, and findings.

## Project Shape

- `src/revllm/`: reusable data, model, reversible-block, and training-loop code.
- `notebooks/reversible_llm.ipynb`: owns all experiment execution.
- `results/runs.jsonl`: durable summaries written by the trainer.
- `.aim/`: Aim experiment database created by runs.
- `../webapp/`: static visualization populated by notebook-exported `data.js`.

## Run

```bash
uv sync
uv run pytest -q
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/reversible_llm.ipynb
uv run aim up
```

The default model uses GPT-2 BPE tokenization (`vocab_size=50257`) and `n_embd=256`, `n_layer=9`, `n_head=8`, `block_size=512`, which lands near 20M parameters with tied input/output embeddings.