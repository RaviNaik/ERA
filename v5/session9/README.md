# Session 9 — The Loss Harness

Making next-token cross-entropy **correct and observable**, then adding a second
output head that predicts `t+2`.

**📓 Notebook:** [`session9_loss_harness.ipynb`](./session9_loss_harness.ipynb) — runs top to bottom, real outputs already saved in place.

---

## The assignment

Starting from:

```python
hidden = model(tokens)
logits = output_head(hidden)
loss = cross_entropy(
    logits[:, :-1].reshape(-1, vocab_size),
    tokens[:, 1:].reshape(-1),
)
```

**Part 1 — the harness.** Make it correct and observable: print every tensor shape
with what each dimension means; verify the input/target shift by printing decoded
token *strings*, not ids; mask padding and confirm the contributing-token count
changes; pack two documents into one sequence, mask the boundary, and show the loss
before and after; compute perplexity and confirm an untrained model sits near
vocabulary size; compare tied vs. untied output-head parameter counts; measure peak
memory for ordinary cross-entropy against a hand-written chunked version.

**Part 2 — one extra head.** Add a second output head predicting `t+2`. Report both
losses separately, their sum, and what happens to the second head's loss over
training relative to the first.

## Stack

- **Project & dependencies:** [`uv`](https://docs.astral.sh/uv/) — `pyproject.toml` / `uv.lock`, reproduced with `uv sync`.
- **Model:** a small decoder-only transformer written from scratch (`src/harness/`) — RMSNorm, pre-norm residual stream, SwiGLU FFN, causal multi-head self-attention. The trunk returns hidden states only; the output head(s) are separate modules attached afterward, exactly as in the harness above.
- **Tokenizer:** GPT-2 BPE via `tiktoken` — a real 50,257-token vocabulary, so "print the strings" is checking something real.
- **Dataset:** [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) (raw).
- **Experiment tracking:** [Aim](https://aimstack.io/) — params, per-step metrics, and logs for every run, in a local `.aim` repo.
- **Hardware:** trained locally on an NVIDIA RTX 500 Ada (4 GB laptop GPU).

## Project layout

```
session9/
├── session9_loss_harness.ipynb   # the whole assignment, executed top to bottom
├── src/harness/                  # model, data, loss-harness utilities the notebook imports
│   ├── model.py                  #   TinyGPT trunk + OutputHead (tied/untied)
│   ├── data.py                   #   WikiText-2 loading, tokenization, packing, padding
│   ├── losses.py                 #   ordinary + chunked cross-entropy, perplexity, peak-memory measurement
│   └── utils.py                  #   seeding, device, shape-printing, param dedup
├── assets/                       # saved plots + results.json written by the notebook
├── .aim/                         # Aim tracking repo (created on first run)
├── pyproject.toml / uv.lock      # uv-managed dependencies
└── build_notebook.py             # generates the notebook from its cell source (for reproducibility/diffing)
```

## Reproducing this

```bash
cd session9
uv sync                     # installs torch (CUDA), tiktoken, datasets, aim, jupyter, ...
uv run jupyter nbconvert --to notebook --execute --inplace session9_loss_harness.ipynb
uv run aim up                # open http://localhost:43800 to browse tracked runs
```

---

## Part 1 — The seven numbers

<!-- SEVEN_NUMBERS_TABLE -->

## Part 2 — The two losses

<!-- MTP_RESULTS -->

---

## Notes on a few decisions

- **Why WikiText-2 + GPT-2 BPE, not a toy vocabulary.** The whole point of "print the
  strings, not the ids" is that misalignment should be visually obvious. A real
  sub-word vocabulary makes that check meaningful; a char-level toy vocabulary would
  hide it.
- **Why the output head is a separate module from the trunk.** `TinyGPT.forward`
  never touches the vocabulary — it returns hidden states only. That's what makes
  tied-vs-untied and the Part-2 second head drop-in changes rather than trunk
  surgery.
- **The tied-weight double-update bug.** Tying reuses `tok_emb.weight` as the head's
  weight — the *same* `nn.Parameter` object. Naively building an optimizer from
  `list(model.parameters()) + list(head.parameters())` lists that tensor twice, and
  Adam would apply two updates to it per step instead of one. `collect_params()`
  (`src/harness/utils.py`) dedups by tensor identity before constructing every
  optimizer in this notebook.
- **Peak memory is measured, not estimated.** `torch.cuda.reset_peak_memory_stats` /
  `max_memory_allocated` bracket each implementation individually, isolated from the
  rest of the notebook's GPU usage.
- **Document-boundary masking is demonstrated in isolation (Part 1, §3.4) but not
  wired into the Part 2 training loops.** The packed training corpus does insert
  `<|endoftext|>` between documents but does not additionally mask the join in the
  loss — a common simplification once a run spans thousands of steps, since that
  noise averages out (unlike padding, which is systematic and always excluded).

## What to read first

If you only read one section of the notebook, read **3.2 (verify the shift)** and
**3.4 (boundary masking)** — they're the two places a training bug can produce a
loss curve that looks perfectly healthy while learning nothing real, which is the
assignment's central warning.
