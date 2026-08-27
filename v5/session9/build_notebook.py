"""Generates session9_loss_harness.ipynb from the CELLS list below.

Run with plain python (stdlib json only, no project deps needed):
    python build_notebook.py
"""
import json

NB_PATH = "session9_loss_harness.ipynb"

CELLS: list[tuple[str, str]] = []


def md(text: str):
    CELLS.append(("markdown", text.strip("\n") + "\n"))


def code(text: str):
    CELLS.append(("code", text.strip("\n") + "\n"))


# ============================================================================
# 0. Title & framing
# ============================================================================

md(r"""
# Session 9 — The Loss Harness: Making Next-Token Cross-Entropy Correct and Observable

**Starting point:**

```python
hidden = model(tokens)
logits = output_head(hidden)
loss = cross_entropy(
    logits[:, :-1].reshape(-1, vocab_size),
    tokens[:, 1:].reshape(-1),
)
```

This notebook takes that four-line snippet and makes it **correct and observable**:
every tensor shape is printed and explained, the input/target shift is verified by
reading actual token *strings* (not ids), padding and document boundaries are
explicitly masked with the contributing-token count shown to change, perplexity is
checked against the vocabulary size as a sanity anchor, tied vs. untied head
parameter counts are compared, and peak memory is measured for an ordinary vs. a
hand-written chunked cross-entropy.

Part 2 adds a second output head that predicts `t+2`, trains it jointly with a
`t+1` head on the same shared trunk, and compares how the two losses evolve.

**Stack:** a `uv`-managed project, a decoder-only transformer (~50M-parameter
trunk: `d_model=512`, 8 layers, 8 heads) trained from scratch, **WikiText-103-raw**
(~117M tokens) tokenized with the real GPT-2 BPE vocabulary (50,257 tokens), and
[Aim](https://aimstack.io/) for tracking every run's params, metrics, and logs.

> **Two run scales.** The defaults below are the *decent proxy scale* meant for a
> GPU server. Setting the environment variable `S9_SMOKE=1` before launching
> shrinks the model, dataset, and step count so the whole notebook runs
> top-to-bottom on a CPU or a tiny GPU in a couple of minutes — that is what
> `dry_run.py` uses to verify the notebook executes end-to-end before the real run.

> The one rule this notebook takes seriously: **a target shift in the wrong
> direction can produce a beautiful loss curve.** Every claim below is checked by
> printing the actual numbers and strings involved, not by trusting that the code
> "looks right."
""")

# ============================================================================
# 1. Setup
# ============================================================================

md(r"""
## 0. Setup & Environment

Standard library, PyTorch, and this project's own `src/harness` package (the model,
data, and loss-harness utilities). `RESULTS` is a running dict we fill in as we go —
it becomes the final "seven numbers" summary at the end of the notebook.
""")

code(r"""
import sys, os, time, math, json
sys.path.insert(0, os.path.abspath("src"))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

from harness.model import ModelConfig, TinyGPT, OutputHead, count_parameters
from harness.data import get_tokenizer, load_and_tokenize, pack_documents, make_batches, build_padded_batch, EOT_ID
from harness.losses import (
    ordinary_cross_entropy, chunked_cross_entropy, chunked_cross_entropy_backward,
    perplexity, measure_peak_memory_bytes,
)
from harness.utils import set_seed, get_device, describe_shape, human_bytes, collect_params

set_seed(1337)
DEVICE = get_device()
RESULTS = {}  # accumulates the headline numbers for the final summary

# ---------------------------------------------------------------------------
# Run configuration. Two scales, selected by the S9_SMOKE environment variable.
#   S9_SMOKE unset / "0"  -> decent proxy scale, for a GPU server
#   S9_SMOKE = "1"         -> tiny smoke scale, runs on CPU in ~2 min (dry_run.py)
# Only the scale changes; every check, mask, and measurement below is identical.
# ---------------------------------------------------------------------------
SMOKE = os.environ.get("S9_SMOKE", "0") == "1"

if SMOKE:
    DATASET_CONFIG = "wikitext-2-raw-v1"
    MAX_DOCS = 4000
    D_MODEL, N_LAYERS, N_HEADS, MAX_SEQ_LEN = 128, 2, 2, 256
    SEQ_LEN, BATCH_SIZE, N_STEPS = 64, 8, 20
    MEM_N_TOKENS, MEM_CHUNK = 2048, 256
else:
    DATASET_CONFIG = "wikitext-103-raw-v1"
    MAX_DOCS = None
    D_MODEL, N_LAYERS, N_HEADS, MAX_SEQ_LEN = 512, 8, 8, 512
    # batch 12 x seq 512 keeps the two full [B*T, V] logits tensors (one per head)
    # within ~5 GB in fp32; raise BATCH_SIZE if the GPU server has room.
    SEQ_LEN, BATCH_SIZE, N_STEPS = 512, 12, 4000
    MEM_N_TOKENS, MEM_CHUNK = 16384, 1024

print(f"run scale              : {'SMOKE' if SMOKE else 'FULL (decent proxy scale)'}")
print(f"dataset config         : {DATASET_CONFIG}")
print(f"model                  : d_model={D_MODEL} n_layers={N_LAYERS} n_heads={N_HEADS} max_seq_len={MAX_SEQ_LEN}")
print(f"training               : seq_len={SEQ_LEN} batch_size={BATCH_SIZE} n_steps={N_STEPS}")

# A small, fixed categorical palette used for every chart in this notebook:
# blue for the t+1 head, amber for the t+2 head, gray for derived/sum series.
# Chosen for light/dark legibility and blue/amber colorblind-safe separation.
COLORS = dict(head1="#2563EB", head2="#F59E0B", sum="#6B7280", grid="#E5E7EB")
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#9CA3AF", "axes.grid": True,
    "grid.color": COLORS["grid"], "grid.linewidth": 0.8, "font.size": 11,
})
os.makedirs("assets", exist_ok=True)

print("=" * 70)
print("ENVIRONMENT")
print("=" * 70)
print(f"torch version        : {torch.__version__}")
print(f"cuda available        : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu                    : {torch.cuda.get_device_name(0)}")
    print(f"total gpu memory       : {human_bytes(torch.cuda.get_device_properties(0).total_memory)}")
print(f"device in use          : {DEVICE}")
print(f"numpy version          : {np.__version__}")
""")

# ============================================================================
# 2. Dataset & tokenizer
# ============================================================================

md(r"""
## 1. Dataset & Tokenizer

**Dataset:** [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext)
(raw) at full scale (~117M GPT-2 tokens), or WikiText-2 (~2M tokens) under
`S9_SMOKE=1`. Both are genuine Wikipedia prose with real document/section
boundaries, so the padding, packing, and shift checks below are checking something
real rather than a synthetic toy. Tokenized ids are cached under `.cache/` so
re-running the notebook skips re-encoding.

**Tokenizer:** GPT-2's BPE encoding via `tiktoken` — a real sub-word vocabulary of
**50,257 tokens**. This matters for the "print the actual strings" check: with a
real BPE tokenizer, misalignment is visually obvious (you'll see whole sub-words
shifted by one position), which a toy char-level vocabulary would hide.
""")

code(r"""
print(f"Loading {DATASET_CONFIG} and tokenizing with GPT-2 BPE (tiktoken)...")
enc = get_tokenizer()
VOCAB_SIZE = enc.n_vocab
print(f"tokenizer            : gpt2 (tiktoken)")
print(f"vocab_size (V)        : {VOCAB_SIZE:,}")

t0 = time.time()
train_docs = load_and_tokenize(split="train", config=DATASET_CONFIG, max_docs=MAX_DOCS)
print(f"\nloaded {len(train_docs):,} documents in {time.time()-t0:.1f}s")

total_tokens = sum(len(d) for d in train_docs)
print(f"total tokens           : {total_tokens:,}")
print(f"avg tokens / doc        : {total_tokens/len(train_docs):.1f}")
print(f"shortest / longest doc  : {min(len(d) for d in train_docs)} / {max(len(d) for d in train_docs)}")

print("\nSample document, decoded back to text (round-trip check that the")
print("tokenizer + dataset pipeline is sane):")
sample = train_docs[10][:30]
print(f"  ids  : {sample}")
print(f"  text : {enc.decode(sample)!r}")

RESULTS["vocab_size"] = VOCAB_SIZE
RESULTS["num_docs"] = len(train_docs)
RESULTS["total_tokens"] = total_tokens
""")

# ============================================================================
# 3. Model
# ============================================================================

md(r"""
## 2. Model

A decoder-only transformer: RMSNorm, pre-norm residual stream, a SwiGLU FFN, and
ordinary causal multi-head self-attention. At the full scale this is a ~50M-param
trunk (`d_model=512`, 8 layers) — a real proxy-scale model, not a toy — while the
assignment's focus stays on the harness around the model.

Critically, **the trunk returns hidden states only.** `TinyGPT.forward` never
touches the vocabulary. The output head is a separate module attached afterward,
exactly as in the assignment's harness (`hidden = model(tokens); logits =
output_head(hidden)`), which is what lets us swap tied/untied heads and add a
second head in Part 2 without touching the trunk at all.
""")

code(r"""
cfg = ModelConfig(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS,
                  n_heads=N_HEADS, max_seq_len=MAX_SEQ_LEN)

model = TinyGPT(cfg).to(DEVICE)
head = OutputHead(cfg.vocab_size, cfg.d_model, tied_weight=model.tok_emb.weight).to(DEVICE)

trunk_params = count_parameters(model)
head_params_tied = head.num_parameters()

print("Model configuration:")
print(f"  d_model    = {cfg.d_model}")
print(f"  n_layers   = {cfg.n_layers}")
print(f"  n_heads    = {cfg.n_heads}")
print(f"  d_ff       = {cfg.d_ff}   (SwiGLU, derived so 3 matrices cost ~= a plain 4x two-matrix FFN)")
print(f"  max_seq_len= {cfg.max_seq_len}")
print(f"  vocab_size = {cfg.vocab_size:,}")
print()
print(f"trunk parameters (embeddings + blocks + norms)  : {trunk_params:,}")
print(f"  of which tok_emb table [V, D]                  : {model.tok_emb.weight.numel():,}")
print(f"output head parameters (tied -> reuses tok_emb)  : {head_params_tied:,} (0 NEW parameters)")
""")

# ============================================================================
# 4. Part 1 intro
# ============================================================================

md(r"""
## 3. Part 1 — The Loss Harness

Working through the assignment's checklist, in order. Each subsection prints the
evidence for its own claim rather than asserting it.
""")

md(r"""
### 3.1 Tensor Shapes — Every Dimension, Named

`hidden = model(tokens)` then `logits = output_head(hidden)`. Below, every tensor's
shape is printed with a one-line explanation of what each dimension actually
indexes.
""")

code(r"""
B, T = 4, 32
long_docs = [d for d in train_docs if len(d) >= T]  # documents are different lengths; need >= T tokens each
tokens = torch.stack([torch.tensor(d[:T], dtype=torch.long) for d in long_docs[:B]]).to(DEVICE)

print("Running the harness forward pass:\n")
hidden = model(tokens)
logits = head(hidden)

describe_shape("tokens", tokens, [
    "batch: which example in the batch",
    "sequence position: which token in the context window",
])
print()
describe_shape("hidden", hidden, [
    "batch: same batch axis as tokens",
    "sequence position: one hidden vector per input position",
    "hidden width D: the model's internal representation size (d_model)",
])
print()
describe_shape("logits", logits, [
    "batch: same batch axis as tokens",
    "sequence position: one prediction per input position",
    "vocabulary V: one raw, unnormalized score per possible next token",
])

RESULTS["shapes"] = {
    "tokens": list(tokens.shape), "hidden": list(hidden.shape), "logits": list(logits.shape),
}
""")

# ============================================================================
# 5. Shift verification
# ============================================================================

md(r"""
### 3.2 Verify the Shift — Print the Strings, Not the IDs

The whole trick of language modelling is that position `i`'s target is position
`i+1`'s token: `logits[:, :-1]` (drop the last position — nothing follows it)
against `tokens[:, 1:]` (drop the first position — nothing predicts it). An
off-by-one here hands the model the answer and produces a beautiful, wrong loss
curve. The only real defense is reading the actual decoded strings.
""")

code(r"""
tokens_in = tokens[:, :-1]
targets_out = tokens[:, 1:]

print(f"tokens_in   shape: {tuple(tokens_in.shape)}  (dropped LAST position: nothing follows it)")
print(f"targets_out shape: {tuple(targets_out.shape)}  (dropped FIRST position: nothing predicts it)")

print("\nInputs vs. targets, decoded to strings, example 0, first 12 positions:")
print(f"{'pos':>4} | {'input token':<18} | {'target token':<18} | check")
print("-" * 60)
n_mismatches = 0
for i in range(12):
    in_tok = enc.decode([tokens_in[0, i].item()])
    tgt_tok = enc.decode([targets_out[0, i].item()])
    expected_tgt = enc.decode([tokens[0, i + 1].item()])
    ok = tgt_tok == expected_tgt
    n_mismatches += (not ok)
    print(f"{i:>4} | {in_tok!r:<18} | {tgt_tok!r:<18} | {'OK' if ok else 'MISMATCH'}")

print(f"\nmismatches found: {n_mismatches} / 12  -> shift is {'CORRECT' if n_mismatches == 0 else 'BROKEN'}")
print("Read position 0 by eye: its input is the sequence's first token, and its")
print("target is the sequence's SECOND token. target[i] really is input[i+1].")

print("\nFor contrast, here is what the (incorrect) UNSHIFTED alignment would print —")
print("the exact bug the assignment warns about, made visible instead of theoretical:")
print(f"{'pos':>4} | {'input token':<18} | {'WRONG target (=input)':<18}")
for i in range(4):
    in_tok = enc.decode([tokens_in[0, i].item()])
    print(f"{i:>4} | {in_tok!r:<18} | {in_tok!r:<18}   <- model would just learn to copy its input")

RESULTS["shift_mismatches"] = n_mismatches
""")

# ============================================================================
# 6. Padding masking
# ============================================================================

md(r"""
### 3.3 Mask Padding — Confirm the Contributing-Token Count Changes

Sequences in a batch have different lengths, so short ones get padded. A padding
position is not a prediction; training on it teaches the model to predict padding
(trivially easy, so the loss *looks* better than it is). The fix is `ignore_index`
on the padded target positions — and the check is that the count of tokens
contributing to the mean actually changes.
""")

code(r"""
print("Building a padded batch from documents of different lengths...\n")
pad_inputs, pad_targets, valid_mask, key_padding_mask, lengths = build_padded_batch(
    train_docs, n_docs=6, pad_to=40
)
print(f"document lengths in this batch : {lengths}")
print(f"pad_inputs shape                : {tuple(pad_inputs.shape)}  (batch, padded sequence length)")

n_total_positions = valid_mask.numel()
n_valid_positions = int(valid_mask.sum().item())
n_pad_positions = n_total_positions - n_valid_positions
print(f"\ntotal (batch x seq) target positions         : {n_total_positions}")
print(f"positions that are REAL next-token targets    : {n_valid_positions}")
print(f"positions that are PADDING (must be excluded)  : {n_pad_positions}")

pad_inputs_d = pad_inputs.to(DEVICE)
pad_targets_d = pad_targets.to(DEVICE)
input_key_padding_mask = key_padding_mask[:, :-1].to(DEVICE)  # matches pad_inputs' length (tokens[:, :-1])
hidden_pad = model(pad_inputs_d, key_padding_mask=input_key_padding_mask)
logits_pad = head(hidden_pad)

# (a) wrong: no masking -- every position, including padding, counted in the mean
loss_no_mask = F.cross_entropy(logits_pad.reshape(-1, VOCAB_SIZE), pad_targets_d.reshape(-1))

# (b) right: padding targets set to ignore_index
targets_masked = pad_targets_d.clone()
targets_masked[~valid_mask.to(DEVICE)] = -100
loss_masked = F.cross_entropy(logits_pad.reshape(-1, VOCAB_SIZE), targets_masked.reshape(-1), ignore_index=-100)

contributing_no_mask = n_total_positions
contributing_masked = int((targets_masked != -100).sum().item())

print(f"\nloss WITHOUT padding mask : {loss_no_mask.item():.4f}  (averaged over {contributing_no_mask} positions, "
      f"{n_pad_positions} of them padding)")
print(f"loss WITH padding mask    : {loss_masked.item():.4f}  (averaged over {contributing_masked} positions, all real)")
print(f"\n==> contributing-token count changed from {contributing_no_mask} to {contributing_masked} "
      f"({n_pad_positions} padding positions removed from the mean).")

RESULTS["padding"] = {
    "contributing_before": contributing_no_mask,
    "contributing_after": contributing_masked,
    "loss_before": loss_no_mask.item(),
    "loss_after": loss_masked.item(),
}
""")

# ============================================================================
# 7. Document boundary masking
# ============================================================================

md(r"""
### 3.4 Pack Two Documents, Mask the Boundary

Packing avoids wasting compute on padding by concatenating documents into one
sequence. But the last token of one document has no relationship to the first
token of the next — training that pair teaches the model that unrelated things
follow each other. Below, two real documents are packed **without** a separator
(to isolate the exact bad pair), the boundary target position is identified, and
the loss is shown before and after masking it, with the arithmetic checked
explicitly rather than just asserted.
""")

code(r"""
long_docs2 = [d for d in train_docs if len(d) >= 24]
doc_a = long_docs2[3][:24]
doc_b = long_docs2[7][:24]
seq = torch.tensor(doc_a + doc_b, dtype=torch.long).unsqueeze(0).to(DEVICE)  # [1, T]

inputs_b = seq[:, :-1]
targets_b = seq[:, 1:].clone()
boundary_idx = len(doc_a) - 1  # last token of doc_a (as input) predicts first token of doc_b (as target)

print(f"doc_a length = {len(doc_a)}   doc_b length = {len(doc_b)}   packed sequence length = {seq.shape[1]}")
print(f"boundary target index = {boundary_idx}")
print(f"  input  at boundary : {enc.decode([inputs_b[0, boundary_idx].item()])!r}   (last real token of doc_a)")
print(f"  target at boundary : {enc.decode([targets_b[0, boundary_idx].item()])!r}   (first real token of doc_b -- UNRELATED to the input)")

hidden_b = model(inputs_b)
logits_b = head(hidden_b)

per_token_losses = F.cross_entropy(
    logits_b.reshape(-1, VOCAB_SIZE), targets_b.reshape(-1), reduction="none"
)
loss_before = per_token_losses.mean()
boundary_loss_value = per_token_losses[boundary_idx].item()

targets_masked_b = targets_b.clone()
targets_masked_b[0, boundary_idx] = -100
loss_after = F.cross_entropy(
    logits_b.reshape(-1, VOCAB_SIZE), targets_masked_b.reshape(-1), ignore_index=-100
)

n_before = targets_b.numel()
n_after = int((targets_masked_b != -100).sum().item())

print(f"\nloss BEFORE masking the boundary : {loss_before.item():.4f}  (mean over {n_before} positions)")
print(f"loss AFTER  masking the boundary : {loss_after.item():.4f}  (mean over {n_after} positions)")
print(f"difference                        : {loss_after.item() - loss_before.item():+.4f}")

print(f"\nthe boundary position's own loss  : {boundary_loss_value:.4f}")
check = (loss_before.item() * n_before - boundary_loss_value) / n_after
print(f"check -- (sum_before - boundary_loss) / (n-1) = {check:.4f}  (should equal loss_after: {loss_after.item():.4f})")

print("\nExplanation: removing the boundary position removes exactly one term from the")
print("mean. On this UNTRAINED model the difference is small, because every position's")
print("loss is close to ln(V) regardless of whether the pair is real -- an untrained head")
print("can't tell a genuine continuation from an unrelated document join yet. Once")
print("training makes the model good at real continuations, an unmasked boundary keeps")
print("getting scored as a genuine failure (the model correctly has no idea what token")
print("follows an unrelated document), which drags the mean loss up and teaches gradient")
print("signal toward a relationship that does not exist. Masking removes that noise from")
print("both the reported loss and the gradient.")

RESULTS["boundary"] = {
    "loss_before": loss_before.item(), "loss_after": loss_after.item(),
    "boundary_loss": boundary_loss_value, "n_before": n_before, "n_after": n_after,
}
""")

# ============================================================================
# 8. Perplexity sanity check
# ============================================================================

md(r"""
### 3.5 Perplexity — An Untrained Model Should Sit Near Vocabulary Size

`perplexity = exp(mean loss)`, read as "how many equally likely options the model
is effectively choosing between." An untrained model, having learned nothing, should
be about as unsure as guessing uniformly over the whole vocabulary: perplexity ≈ V,
loss ≈ ln(V). If it isn't close, the target alignment is broken and nothing past
this point can be trusted.
""")

code(r"""
print("Untrained-model perplexity sanity check\n")

PPL_SEQ_LEN = min(128, MAX_SEQ_LEN)
packed_check = pack_documents(train_docs, seq_len=PPL_SEQ_LEN, n_sequences=8, seed=42)
check_tokens, check_targets, _ = next(make_batches(packed_check, seq_len=PPL_SEQ_LEN, batch_size=8))
check_tokens = check_tokens.to(DEVICE)
check_targets = check_targets.to(DEVICE)

with torch.no_grad():
    h = model(check_tokens)
    z = head(h)
    loss_untrained = F.cross_entropy(z.reshape(-1, VOCAB_SIZE), check_targets.reshape(-1))

ppl_untrained = perplexity(loss_untrained)
ln_v = math.log(VOCAB_SIZE)
rel_gap = abs(loss_untrained.item() - ln_v) / ln_v

print(f"loss (untrained)        : {loss_untrained.item():.4f} nats")
print(f"perplexity (untrained)   : {ppl_untrained:,.1f}")
print(f"vocabulary size V         : {VOCAB_SIZE:,}")
print(f"ln(V)                     : {ln_v:.4f}")
print(f"relative gap from ln(V)   : {rel_gap*100:.2f}%")
verdict = "PASS" if rel_gap < 0.05 else "CHECK"
print(f"\n{verdict}: untrained loss should sit within a few percent of ln(V) -- a random,")
print("freshly-initialized head scores the vocabulary close to uniformly, before any")
print("training signal has shaped it.")

RESULTS["perplexity"] = {
    "loss_untrained": loss_untrained.item(), "ppl_untrained": ppl_untrained,
    "ln_v": ln_v, "vocab_size": VOCAB_SIZE, "relative_gap_pct": rel_gap * 100,
}
""")

# ============================================================================
# 9. Tied vs untied
# ============================================================================

md(r"""
### 3.6 Tied vs. Untied Head Parameter Counts

Tying reuses the input embedding table `W_embed [V, D]` as the output head, adding
zero new parameters. Untying gives the head its own `[V, D]` matrix. On this
configuration, that matrix is the single largest block in the whole model.
""")

code(r"""
untied_head = OutputHead(cfg.vocab_size, cfg.d_model, tied_weight=None).to(DEVICE)

model_params = count_parameters(model)         # trunk, incl. the tok_emb table -- independent of head choice
tied_head_new_params = 0                        # tied head adds no new parameters
untied_head_new_params = count_parameters(untied_head)

total_tied = model_params + tied_head_new_params
total_untied = model_params + untied_head_new_params

print(f"trunk parameters (embeddings + blocks + norms) : {model_params:,}")
print(f"  of which the [V, D]-sized table                : {model.tok_emb.weight.numel():,}")
print()
print(f"TIED   total parameters : {total_tied:,}   (head adds {tied_head_new_params:,} new params)")
print(f"UNTIED total parameters : {total_untied:,}   (head adds {untied_head_new_params:,} new params)")
print(f"difference                : {total_untied - total_tied:,} parameters")
print(f"ratio untied / tied        : {total_untied / total_tied:.3f}x")

RESULTS["tied_vs_untied"] = {
    "trunk_params": model_params, "total_tied": total_tied, "total_untied": total_untied,
    "ratio": total_untied / total_tied,
}
""")

# ============================================================================
# 10. Peak memory
# ============================================================================

md(r"""
### 3.7 Peak Memory — Ordinary vs. Chunked Cross-Entropy

Two ways to compute exactly the same objective. **Ordinary:** materialize the full
`[N, V]` logits tensor, then cross-entropy. **Chunked:** process `chunk_size` tokens
at a time, accumulate the summed loss, and discard each chunk's logits before
computing the next — at most `[chunk_size, V]` logits ever exist at once. First a
correctness check on a small tensor (the two must agree to float precision), then
the actual peak-memory measurement on a much larger one.
""")

code(r"""
CHUNK = MEM_CHUNK

print("Correctness check -- does chunking change the loss, or the gradients it produces?\n")
check_hidden = torch.randn(4, 16, cfg.d_model, device=DEVICE, requires_grad=True)
check_targets2 = torch.randint(0, VOCAB_SIZE, (4, 16), device=DEVICE)
check_weight = untied_head.weight.detach().clone().requires_grad_(True)

# (a) loss values agree
l_ord = ordinary_cross_entropy(check_hidden, check_weight, check_targets2)
l_chk = chunked_cross_entropy(check_hidden, check_weight, check_targets2, chunk_size=5)
print(f"ordinary loss  : {l_ord.item():.10f}")
print(f"chunked  loss  : {l_chk.item():.10f}")
print(f"abs diff        : {abs(l_ord.item() - l_chk.item()):.2e}  (agree to float precision)")

# (b) gradients agree too -- chunking must not silently change what the model learns.
# chunked_cross_entropy_backward does its own backward (chunk-by-chunk, freeing each
# chunk's graph before the next), so it needs its own fresh leaf tensors to accumulate into.
h_for_ord = check_hidden.detach().clone().requires_grad_(True)
w_for_ord = check_weight.detach().clone().requires_grad_(True)
ordinary_cross_entropy(h_for_ord, w_for_ord, check_targets2).backward()

h_for_chk = check_hidden.detach().clone().requires_grad_(True)
w_for_chk = check_weight.detach().clone().requires_grad_(True)
chunked_cross_entropy_backward(h_for_chk, w_for_chk, check_targets2, chunk_size=5)

grad_diff_h = (h_for_ord.grad - h_for_chk.grad).abs().max().item()
grad_diff_w = (w_for_ord.grad - w_for_chk.grad).abs().max().item()
print(f"max |grad diff| on hidden : {grad_diff_h:.2e}")
print(f"max |grad diff| on head weight : {grad_diff_w:.2e}")
print("PASS -- chunking changes memory, not the objective or its gradient." if max(grad_diff_h, grad_diff_w) < 1e-5
      else "CHECK -- gradients disagree, something in the chunking is wrong.")

print("\nPeak-memory measurement\n")
N_TOKENS = MEM_N_TOKENS  # flattened batch x seq the loss must score at once
MEM_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

hidden_mem = torch.randn(N_TOKENS, cfg.d_model, device=DEVICE, dtype=MEM_DTYPE, requires_grad=True)
head_w_mem = torch.randn(VOCAB_SIZE, cfg.d_model, device=DEVICE, dtype=MEM_DTYPE, requires_grad=True)
targets_mem = torch.randint(0, VOCAB_SIZE, (N_TOKENS,), device=DEVICE)

def run_ordinary():
    # naive path: materialize the full [N, V] logits, backward through all of it at once
    loss = ordinary_cross_entropy(hidden_mem.unsqueeze(0), head_w_mem, targets_mem.unsqueeze(0))
    loss.backward()
    hidden_mem.grad = None
    head_w_mem.grad = None

def run_chunked():
    # chunked path: backward happens per-chunk *inside* this call, so at most one
    # chunk's [chunk_size, V] logits (and its backward graph) exist at a time
    chunked_cross_entropy_backward(hidden_mem.unsqueeze(0), head_w_mem, targets_mem.unsqueeze(0), chunk_size=CHUNK)
    hidden_mem.grad = None
    head_w_mem.grad = None

peak_ordinary = measure_peak_memory_bytes(run_ordinary, DEVICE)
peak_chunked = measure_peak_memory_bytes(run_chunked, DEVICE)

print(f"tokens (N)                : {N_TOKENS:,}")
print(f"vocab size V               : {VOCAB_SIZE:,}")
print(f"dtype                      : {MEM_DTYPE}")
print(f"chunk size                  : {CHUNK}")
print(f"theoretical materialized logits : {human_bytes(N_TOKENS*VOCAB_SIZE*2)}  ({N_TOKENS} x {VOCAB_SIZE})")
print(f"theoretical one-chunk logits     : {human_bytes(CHUNK*VOCAB_SIZE*2)}  ({CHUNK} x {VOCAB_SIZE})")
print()
print(f"measured peak memory, ORDINARY cross-entropy : {human_bytes(peak_ordinary)}")
print(f"measured peak memory, CHUNKED  cross-entropy : {human_bytes(peak_chunked)}")
if peak_chunked > 0:
    ratio_mem = peak_ordinary / peak_chunked
    print(f"ratio (ordinary / chunked)                     : {ratio_mem:.1f}x")
else:
    ratio_mem = None
    print("(CUDA not available -- peak memory not measurable on CPU the same way)")

RESULTS["memory"] = {
    "n_tokens": N_TOKENS, "chunk_size": CHUNK, "dtype": str(MEM_DTYPE),
    "peak_ordinary_bytes": int(peak_ordinary), "peak_chunked_bytes": int(peak_chunked),
    "ratio": ratio_mem,
}
""")

code(r"""
# Part 1's exploratory tensors (the large memory-measurement buffers, the
# untied-head duplicate) are no longer needed. Free them explicitly before
# Part 2 trains three more models on the same small GPU.
del (hidden_mem, head_w_mem, targets_mem, untied_head, model, head,
     check_hidden, check_targets2, check_weight,
     h_for_ord, w_for_ord, h_for_chk, w_for_chk)
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
    print(f"GPU memory freed. currently allocated: {human_bytes(torch.cuda.memory_allocated(DEVICE))}")
""")

# ============================================================================
# 11. Part 1 recap
# ============================================================================

md(r"""
### Part 1 Recap

Every checklist item now has a number behind it: shapes are printed and explained,
the shift is verified by eye against decoded strings, padding and document
boundaries are masked with the contributing-token count shown to move, perplexity
lands near `ln(V)`, tied vs. untied parameter counts are compared, and peak memory
for the two cross-entropy implementations is measured with an explicit ratio. The
full "seven numbers" table is assembled in the final summary section, after Part 2.
""")

# ============================================================================
# 12. Aim setup
# ============================================================================

md(r"""
## 4. Experiment Tracking with Aim

[Aim](https://aimstack.io/) tracks this run's hyperparameters, per-step metrics, and
logs, stored in a local `.aim` repo inside this project. Everything logged below —
losses, perplexities, both MTP heads — is queryable later via `aim up`, without
re-running anything.
""")

code(r"""
from aim import Run, Repo

repo = Repo(".", init=True)
aim_run = Run(repo=repo, experiment="session9-loss-harness")
aim_run["hparams"] = {
    "d_model": cfg.d_model, "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
    "d_ff": cfg.d_ff, "vocab_size": cfg.vocab_size, "max_seq_len": cfg.max_seq_len,
    "tokenizer": "gpt2 (tiktoken)", "dataset": DATASET_CONFIG,
    "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE, "n_steps": N_STEPS,
    "smoke": SMOKE,
}
print(f"Aim run hash   : {aim_run.hash}")
print(f"Aim repo path   : {os.path.abspath('.')}")
print("hparams logged   :", json.dumps(aim_run["hparams"], indent=2))
""")

# ============================================================================
# 13. Baseline training
# ============================================================================

md(r"""
## 5. Baseline Training — Single Head, Predicts t+1

A training run of the ordinary single-head model (one tied `t+1` head, the standard
setup) on the packed corpus, logging loss and perplexity to Aim every step. This is
the reference `t+1` curve; Part 2 then trains two untied heads (`t+1` and `t+2`)
side by side on the same trunk, data, step count, and seed.
""")

code(r"""
print(f"Training baseline (t+1 only): seq_len={SEQ_LEN} batch_size={BATCH_SIZE} steps={N_STEPS}\n")

set_seed(1337)
baseline_model = TinyGPT(cfg).to(DEVICE)
baseline_head = OutputHead(cfg.vocab_size, cfg.d_model, tied_weight=baseline_model.tok_emb.weight).to(DEVICE)
opt = torch.optim.AdamW(collect_params(baseline_model, baseline_head), lr=3e-4)

packed_train = pack_documents(train_docs, seq_len=SEQ_LEN, n_sequences=BATCH_SIZE * (N_STEPS + 8), seed=0)

baseline_losses = []
t0 = time.time()
step = 0
for tokens_i, targets_i, _boundary_i in make_batches(packed_train, SEQ_LEN, BATCH_SIZE):
    if step >= N_STEPS:
        break
    tokens_i = tokens_i.to(DEVICE)
    targets_i = targets_i.to(DEVICE)

    h = baseline_model(tokens_i)
    z = baseline_head(h)
    loss = F.cross_entropy(z.reshape(-1, VOCAB_SIZE), targets_i.reshape(-1))

    opt.zero_grad()
    loss.backward()
    opt.step()

    baseline_losses.append(loss.item())
    aim_run.track(loss.item(), name="loss", context={"subset": "baseline"}, step=step)
    aim_run.track(perplexity(loss.item()), name="perplexity", context={"subset": "baseline"}, step=step)

    if step % 25 == 0 or step == N_STEPS - 1:
        print(f"step {step:4d}/{N_STEPS} | loss {loss.item():.4f} | ppl {perplexity(loss.item()):9.1f} "
              f"| elapsed {time.time()-t0:5.1f}s")
    step += 1

print(f"\nbaseline training done in {time.time()-t0:.1f}s")
print(f"loss: {baseline_losses[0]:.4f} -> {baseline_losses[-1]:.4f}  "
      f"(perplexity {perplexity(baseline_losses[0]):.0f} -> {perplexity(baseline_losses[-1]):.1f})")
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(baseline_losses, color=COLORS["head1"], linewidth=2)
ax.set_xlabel("training step")
ax.set_ylabel("loss (nats)")
ax.set_title("Baseline single-head (t+1) training loss")
plt.tight_layout()
plt.savefig("assets/baseline_loss.png", dpi=130)
plt.show()
""")

# ============================================================================
# 14. Part 2 intro
# ============================================================================

md(r"""
## 6. Part 2 — A Second Output Head Predicting t+2

Same trunk architecture, same packed corpus, same step count and seed as the
baseline above (the only batching difference is a `seq_len+2` window so every input
position has both a `t+1` and a `t+2` target). Two output heads sit on top of the
shared hidden state: **Head 1** predicts `t+1`, **Head 2** predicts `t+2`. Both are
**untied and architecturally identical** — the only difference is the target — so
the loss gap between them measures exactly one thing: how much harder it is to
predict two tokens ahead than one. The two losses are reported separately and
summed; the sum is what gets optimized (added, not averaged, so each head pulls on
the shared trunk with equal weight).
""")

code(r"""
def make_mtp_batches(packed, seq_len, batch_size):
    '''Like make_batches, but yields a seq_len+2 window so we have both
    a t+1 and a t+2 target for every input position.'''
    n_positions = (len(packed.ids) - 2) // seq_len
    ids = packed.ids[: n_positions * seq_len + 2]
    chunks = []
    for i in range(n_positions):
        s = i * seq_len
        chunks.append(ids[s: s + seq_len + 2])
    chunks = torch.stack(chunks)
    for i in range(0, len(chunks) - batch_size + 1, batch_size):
        block = chunks[i:i + batch_size]
        tok = block[:, :-2]
        tgt1 = block[:, 1:-1]
        tgt2 = block[:, 2:]
        yield tok, tgt1, tgt2

print(f"Training MTP model (t+1 and t+2 jointly): seq_len={SEQ_LEN} batch_size={BATCH_SIZE} steps={N_STEPS}\n")

set_seed(1337)
mtp_model = TinyGPT(cfg).to(DEVICE)
# Both heads UNTIED and architecturally identical -- the only thing that differs
# is the target (t+1 vs t+2). Tying Head 1 to the embedding (as the baseline does)
# would confound the comparison: any loss gap could be "t+2 is harder" OR "the
# tied head is more constrained." Untying both isolates the prediction-distance
# effect, which is what Part 2 is actually asking about.
head_t1 = OutputHead(cfg.vocab_size, cfg.d_model, tied_weight=None).to(DEVICE)
head_t2 = OutputHead(cfg.vocab_size, cfg.d_model, tied_weight=None).to(DEVICE)

mtp_params = collect_params(mtp_model, head_t1, head_t2)
opt2 = torch.optim.AdamW(mtp_params, lr=3e-4)

n_mtp_total = count_parameters(mtp_model) + count_parameters(head_t1) + count_parameters(head_t2)
print(f"MTP model total parameters (trunk + untied head1 + untied head2): {n_mtp_total:,}")
print(f"  each output head is a [{cfg.vocab_size:,} x {cfg.d_model}] matrix = {count_parameters(head_t2):,} params\n")

packed_train2 = pack_documents(train_docs, seq_len=SEQ_LEN, n_sequences=BATCH_SIZE * (N_STEPS + 8), seed=0)

loss1_hist, loss2_hist, sum_hist = [], [], []
t0 = time.time()
step = 0
for tokens_i, t1_i, t2_i in make_mtp_batches(packed_train2, SEQ_LEN, BATCH_SIZE):
    if step >= N_STEPS:
        break
    tokens_i = tokens_i.to(DEVICE)
    t1_i = t1_i.to(DEVICE)
    t2_i = t2_i.to(DEVICE)

    h = mtp_model(tokens_i)
    z1 = head_t1(h)
    z2 = head_t2(h)
    loss1 = F.cross_entropy(z1.reshape(-1, VOCAB_SIZE), t1_i.reshape(-1))
    loss2 = F.cross_entropy(z2.reshape(-1, VOCAB_SIZE), t2_i.reshape(-1))
    loss_sum = loss1 + loss2

    opt2.zero_grad()
    loss_sum.backward()
    opt2.step()

    loss1_hist.append(loss1.item())
    loss2_hist.append(loss2.item())
    sum_hist.append(loss_sum.item())

    aim_run.track(loss1.item(), name="loss", context={"subset": "mtp_head1_t+1"}, step=step)
    aim_run.track(loss2.item(), name="loss", context={"subset": "mtp_head2_t+2"}, step=step)
    aim_run.track(loss_sum.item(), name="loss", context={"subset": "mtp_sum"}, step=step)

    if step % 25 == 0 or step == N_STEPS - 1:
        print(f"step {step:4d}/{N_STEPS} | L1(t+1) {loss1.item():.4f} | L2(t+2) {loss2.item():.4f} "
              f"| sum {loss_sum.item():.4f} | gap(L2-L1) {loss2.item()-loss1.item():+.4f}")
    step += 1

print(f"\nMTP training done in {time.time()-t0:.1f}s")
""")

code(r"""
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(loss1_hist, color=COLORS["head1"], linewidth=2, label="Head 1: predicts t+1")
ax.plot(loss2_hist, color=COLORS["head2"], linewidth=2, label="Head 2: predicts t+2")
ax.plot(sum_hist, color=COLORS["sum"], linewidth=1.5, linestyle="--", label="Sum (L1 + L2, the optimized objective)")
ax.set_xlabel("training step")
ax.set_ylabel("loss (nats)")
ax.set_title("Multi-token prediction: t+1 vs. t+2 head losses over training")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("assets/mtp_losses.png", dpi=130)
plt.show()
""")

md(r"""
### 6.1 What Happens to the Second Head's Loss

Both heads share the same trunk, so anything that improves the shared hidden state
helps both objectives at once — which is why the two curves fall *together* rather
than independently. But predicting `t+2` is a strictly harder target than `t+1`: it
has to survive one additional step of genuine uncertainty about what the text does
next, so its irreducible entropy floor is higher than `t+1`'s even for a perfect
model. At step 0 both heads are random, so the gap is ~0 and pure noise; the
expectation is that as training shapes the trunk, `L2` settles **above** `L1` and
stays there. The cell below prints the actual gap over the whole run so you can
read what happened rather than take the claim on trust.
""")

code(r"""
start_gap = loss2_hist[0] - loss1_hist[0]
end_gap = loss2_hist[-1] - loss1_hist[-1]

print(f"Head 1 (t+1): {loss1_hist[0]:.4f} -> {loss1_hist[-1]:.4f}   (drop of {loss1_hist[0]-loss1_hist[-1]:.4f} nats, "
      f"ppl {perplexity(loss1_hist[0]):.0f} -> {perplexity(loss1_hist[-1]):.1f})")
print(f"Head 2 (t+2): {loss2_hist[0]:.4f} -> {loss2_hist[-1]:.4f}   (drop of {loss2_hist[0]-loss2_hist[-1]:.4f} nats, "
      f"ppl {perplexity(loss2_hist[0]):.0f} -> {perplexity(loss2_hist[-1]):.1f})")
print(f"Sum          : {sum_hist[0]:.4f} -> {sum_hist[-1]:.4f}")

gaps = np.array(loss2_hist) - np.array(loss1_hist)
frac_above = float((gaps > 0).mean())
# ignore the first ~10% of steps (random-init transient) when summarising
tail = gaps[max(1, len(gaps) // 10):]
print(f"\ngap (L2 - L1) at step 0        : {start_gap:+.4f} nats")
print(f"gap (L2 - L1) at final step    : {end_gap:+.4f} nats")
print(f"gap (L2 - L1) mean over run    : {gaps.mean():+.4f} nats")
print(f"gap (L2 - L1) mean, after warmup: {tail.mean():+.4f} nats  (min {tail.min():+.4f}, max {tail.max():+.4f})")
print(f"fraction of steps with L2 > L1 : {frac_above*100:.1f}%")
verdict = "as expected: t+2 is the harder target" if tail.mean() > 0 else "NOT as expected -- inspect (too few steps? lr? scale?)"
print(f"==> {verdict}")

RESULTS["mtp"] = {
    "steps": N_STEPS,
    "head1_loss_start": loss1_hist[0], "head1_loss_end": loss1_hist[-1],
    "head2_loss_start": loss2_hist[0], "head2_loss_end": loss2_hist[-1],
    "sum_start": sum_hist[0], "sum_end": sum_hist[-1],
    "gap_start": start_gap, "gap_end": end_gap,
    "gap_mean": float(gaps.mean()), "gap_mean_after_warmup": float(tail.mean()),
    "frac_steps_L2_above_L1": frac_above,
}
""")

# ============================================================================
# 15. Aim inspection
# ============================================================================

md(r"""
## 7. Inspecting the Aim Run

The run is closed, and the on-disk `.aim` repo is checked directly to confirm
everything was actually written to the tracking store (rather than just trusting
that `.track()` calls silently succeeded).
""")

code(r"""
run_hash = aim_run.hash
aim_run.close()

repo_check = Repo(".")
all_hashes = list(repo_check._all_run_hashes())
print(f"Aim repo path         : {os.path.abspath('.aim')}")
print(f"this run's hash        : {run_hash}")
print(f"runs found in the repo  : {len(all_hashes)} -> {all_hashes}")
print(f"this run is present     : {run_hash in all_hashes}")

print("\nTo explore interactively:")
print("  cd session9 && uv run aim up")
print("  then open http://localhost:43800")
""")

# ============================================================================
# 16. Final summary
# ============================================================================

md(r"""
## 8. Final Summary — The Seven Numbers (Part 1) + Two Losses (Part 2)

One row per Part 1 checklist item, plus the Part 2 head comparison. Also written to
`assets/results.json` so the README can quote the exact same numbers this run
produced.
""")

code(r"""
seven_numbers = [
    ("1. Tensor shapes", f"tokens {RESULTS['shapes']['tokens']}, hidden {RESULTS['shapes']['hidden']}, "
                          f"logits {RESULTS['shapes']['logits']}"),
    ("2. Shift verification", f"{RESULTS['shift_mismatches']} mismatches found (0 = shift is correct)"),
    ("3. Padding mask: contributing tokens", f"{RESULTS['padding']['contributing_before']} -> "
                                              f"{RESULTS['padding']['contributing_after']} "
                                              f"(loss {RESULTS['padding']['loss_before']:.4f} -> {RESULTS['padding']['loss_after']:.4f})"),
    ("4. Boundary mask: loss before/after", f"{RESULTS['boundary']['loss_before']:.4f} -> "
                                             f"{RESULTS['boundary']['loss_after']:.4f}"),
    ("5. Perplexity vs. vocab size", f"ppl={RESULTS['perplexity']['ppl_untrained']:.1f} vs. V={RESULTS['perplexity']['vocab_size']:,} "
                                      f"(gap {RESULTS['perplexity']['relative_gap_pct']:.2f}%)"),
    ("6. Tied vs. untied head params", f"{RESULTS['tied_vs_untied']['total_tied']:,} vs. "
                                        f"{RESULTS['tied_vs_untied']['total_untied']:,} "
                                        f"({RESULTS['tied_vs_untied']['ratio']:.3f}x)"),
    ("7. Peak memory: ordinary vs. chunked", f"{human_bytes(RESULTS['memory']['peak_ordinary_bytes'])} vs. "
                                              f"{human_bytes(RESULTS['memory']['peak_chunked_bytes'])} "
                                              + (f"({RESULTS['memory']['ratio']:.1f}x)" if RESULTS['memory']['ratio'] else "")),
]

df = pd.DataFrame(seven_numbers, columns=["Checklist item", "Result"])
print("=" * 100)
print("PART 1 -- THE SEVEN NUMBERS")
print("=" * 100)
for _, row in df.iterrows():
    print(f"{row['Checklist item']:<42} {row['Result']}")

print()
print("=" * 100)
print("PART 2 -- THE TWO LOSSES")
print("=" * 100)
print(f"Head 1 (t+1) final loss : {RESULTS['mtp']['head1_loss_end']:.4f} nats")
print(f"Head 2 (t+2) final loss : {RESULTS['mtp']['head2_loss_end']:.4f} nats")
print(f"Sum (optimized)          : {RESULTS['mtp']['sum_end']:.4f} nats")
print(f"gap (L2 - L1), start -> end : {RESULTS['mtp']['gap_start']:+.4f} -> {RESULTS['mtp']['gap_end']:+.4f}")

RESULTS["smoke"] = SMOKE
RESULTS["config"] = {
    "dataset": DATASET_CONFIG, "d_model": cfg.d_model, "n_layers": cfg.n_layers,
    "n_heads": cfg.n_heads, "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE, "n_steps": N_STEPS,
}
with open("assets/results.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nWrote assets/results.json")
if SMOKE:
    print("\n*** SMOKE RUN — these numbers are from the tiny CPU config, not the real run. ***")
    print("*** Re-run with S9_SMOKE unset on a GPU server for the numbers to report.   ***")
""")

md(r"""
## 9. Conclusion

The harness now does what the assignment asked: every shape is printed and named,
the shift is confirmed by reading real decoded strings rather than trusting integer
arithmetic, padding and document-boundary masking are shown to change the
contributing-token count and the loss, perplexity lands near `ln(V)` as the sanity
anchor it's meant to be, tied vs. untied head parameter counts are compared directly
on this configuration, and peak memory is measured (not estimated) for both an
ordinary and a hand-written chunked cross-entropy. Part 2 adds a second,
architecturally identical untied head predicting `t+2`: once the random-init
transient passes, its loss settles above the `t+1` head's — forecasting two tokens
ahead is a harder, higher-entropy task — while both curves fall together because
they share one trunk. All of the above was tracked step-by-step in Aim, queryable
independently of this notebook via
`aim up`.
""")

def write():
    cells = []
    for ctype, source in CELLS:
        lines = source.splitlines(keepends=True)
        cell = {"cell_type": ctype, "metadata": {}, "source": lines}
        if ctype == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (session9)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {NB_PATH} with {len(cells)} cells")


if __name__ == "__main__":
    write()
