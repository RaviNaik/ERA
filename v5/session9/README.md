# Session 9 — The Loss Harness

**Making next-token cross-entropy correct and observable, then adding a second output head that predicts `t+2`.**

The assignment starts from four lines that *look* right:

```python
hidden = model(tokens)
logits = output_head(hidden)
loss = cross_entropy(
    logits[:, :-1].reshape(-1, vocab_size),
    tokens[:, 1:].reshape(-1),
)
```

…and asks you to turn them into something you can *trust* — every shape printed and named, the input/target shift verified by reading decoded **strings** (not ids), padding and document boundaries masked with the contributing-token count shown to move, perplexity checked against vocabulary size, tied-vs-untied head cost measured, and peak memory for an ordinary vs. a hand-written chunked cross-entropy measured on a real GPU. Then Part 2 bolts on a second head that predicts two tokens ahead and watches what its loss does.

> 📓 **Notebook:** [`loss_harness/session9_loss_harness.ipynb`](./loss_harness/session9_loss_harness.ipynb) — runs top to bottom (real run on an RTX A6000, ~30 min; `S9_SMOKE=1` for a 2-minute CPU pass).
> 🌐 **Web app:** [`webapp/index.html`](./webapp/index.html) — visual walkthrough of the experiment and results.
>
> *Revised after a review pass — see [§9](#9-notes-on-a-few-decisions). Numbers marked "pending re-run" refresh when the notebook is re-executed on GPU.*

---

## 1. The one warning this session is about

> A target shift in the **wrong** direction produces a beautiful, smoothly-falling loss curve. The model just learns to copy its input. It never raises an exception.

Everything in Part 1 is a defense against that class of silent bug — the ones that live in the few lines between the model output and the scalar. The single most important cell in the notebook prints this, for a *human* to read:

```
 pos | input token        | target token       | check
------------------------------------------------------------
   0 | 'Sen'              | 'j'                | OK
   1 | 'j'                | 'ō'                | OK
   2 | 'ō'                | ' no'              | OK
   3 | ' no'              | ' V'               | OK
   ...
```

vs. the bug it rules out:

```
 pos | input token        | WRONG target (=input)
   0 | 'Sen'              | 'Sen'    <- model would just learn to copy its input
```

The `0 / 12` counter beneath the table only checks that `targets_out == tokens[:, 1:]` — true by construction of the slice, so it catches a wrong slice expression but is **not** a mechanical proof that the loss call is aligned. That `logits[:, :-1]` lines up with those targets is a *functional* fact, checked by the **perplexity anchor** (§4, check 5): an untrained model that lands at ≈ `ln V` is aligned; a wrong-way shift collapses it well below, and a forgotten shift makes the trained loss fall to near zero. The strings and the anchor together are the guard.

---

## 2. The loss spine

```mermaid
flowchart LR
    T["tokens<br/>[B, T]"] --> M["decoder trunk<br/>(8 blocks)"]
    M --> H["hidden state h<br/>[B, T, D=512]"]
    H --> O["output head<br/>W_vocab [V, D]"]
    O --> Z["logits z<br/>[B, T, V=50257]"]
    Z --> S["softmax<br/>(gaps → ratios)"]
    S --> P["probabilities p<br/>[B, T, V]"]
    P --> X["cross-entropy<br/>−log p(true next token)"]
    X --> L(["scalar loss L<br/>(nats)"])

    T -. "shift by 1" .-> Y["targets<br/>tokens[:, 1:]"]
    Y --> X
```

Cross-entropy asks exactly one question at every position: **what probability did you assign to the token that actually came next?** Perplexity = `exp(mean loss)` re-expresses that as "how many equally-likely options is the model effectively choosing between."

---

## 3. Experiment setup

| | |
|---|---|
| **Hardware** | NVIDIA RTX A6000 (47.5 GiB), CUDA 12.4, PyTorch 2.6.0 |
| **Dataset** | WikiText-103-raw — **1,153,246** documents, **115,689,651** GPT-2 tokens |
| **Tokenizer** | GPT-2 BPE via `tiktoken` — real sub-word vocab, **V = 50,257** |
| **Model** | decoder-only transformer, from scratch (`loss_harness/src/harness/`) |
| **Trunk** | `d_model=512`, `n_layers=8`, `n_heads=8`, `d_ff=1408` (SwiGLU), `max_seq_len=512` — **51,692,544 params** |
| **Norm / layout** | RMSNorm, pre-norm residual stream |
| **Training** | `seq_len=512`, `batch=12` (6,144 tokens/step), **4,000 steps**, AdamW `lr=3e-4` |
| **Tracking** | [Aim](https://aimstack.io/) — every step's loss & perplexity |
| **Wall clock** | baseline run ≈ 11 min · MTP run ≈ 16 min |

### Model architecture

```mermaid
flowchart TB
    subgraph Block["transformer block  ×8  (pre-norm)"]
      direction TB
      x1["residual stream x"] --> n1["RMSNorm"]
      n1 --> attn["causal multi-head<br/>self-attention (8 heads)"]
      attn --> a1(("+"))
      x1 --> a1
      a1 --> n2["RMSNorm"]
      n2 --> ffn["SwiGLU FFN<br/>down( silu(gate(x)) ⊙ up(x) )"]
      ffn --> a2(("+"))
      a1 --> a2
      a2 --> xo["x out"]
    end

    emb["token emb [V,D]  +  pos emb [512,D]"] --> Block
    Block --> fn["final RMSNorm"]
    fn --> hh["hidden state h [B,T,512]"]
    hh --> head["output head → logits [B,T,50257]"]
```

The trunk **returns hidden states only** — `TinyGPT.forward` never touches the vocabulary. The output head is a separate module attached afterward (`hidden = model(tokens); logits = head(hidden)`), which is exactly what makes tied-vs-untied and the Part-2 second head *drop-in* changes rather than surgery on the trunk.

---

## 4. Part 1 — the seven numbers

| # | Check | Result | What it tells you |
|---|---|---|---|
| **1** | **Tensor shapes** | `tokens [4, 32]` → `hidden [4, 32, 512]` → `logits [4, 32, 50257]` | `[batch, position]` → `+ hidden width D` → `+ one score per vocab token V`. The logits tensor is **98×** bigger than the hidden state that produced it. |
| **2** | **Shift verification** | prints the inputs/targets table for eye-reading; **0 / 12** target-slice mismatches | The table is the deliverable — you read it and confirm each target is the next word. The counter only proves `targets_out` is the right *slice*; loss-call alignment is check 5. |
| **3** | **Padding mask** | contributing tokens **234 → 167** (−67); loss **10.6834 → 10.8603** | The loss went *up* after masking. Padding is trivially predictable, so counting it was **flattering the mean**. The contributing-token count is the tell. |
| **4** | **Document-boundary mask** | *untrained:* loss **10.7940 → 10.7902**, boundary position **10.9666** ≈ the mean (delta from masking is noise) · *trained `t+1` head:* §6.2 re-runs the exact check — ⏳ *pending re-run* | Untrained, the join is indistinguishable from a real continuation. Trained, the model is good at real continuations, so predicting an unrelated document's first token is a genuine high-loss position that drags the mean up and teaches a relationship that doesn't exist. §6.2 (new cell, after Part 2) re-scores the same two documents on the trained head. |
| **5** | **Perplexity anchor** (in nats) | untrained loss **10.9328** vs. `ln V` **10.8249** — **1.0 % gap in nats** (perplexity **55,984** vs. **V = 50,257**, ~11 % high) | ✅ PASS. The right unit here is nats: the model is ~1 % from a uniform distribution. In perplexity that 1 % becomes ~11 %, which is why "sits near the vocabulary size" is doing a little work — read the nats. |
| **6** | **Tied vs. untied head params** | **51,692,544** vs. **77,424,128** — **+25,731,584** params, ratio **1.498×** | Untying gives the head its own `[V, D]` matrix — `50257 × 512 = 25.7M` params, the single largest block in the model, as big as the entire token-embedding table. |
| **7** | **Peak memory: ordinary vs. chunked CE** | **5.99 GiB** vs. **1.83 GiB** — ratio **3.3×** at `N = 16,384` · notebook now measures at `N = 32,768` (ordinary ≈ 12 GiB) — ⏳ *ratio refreshes on re-run* | Same loss to `9.5e-7`, same gradients to `2.6e-8`. The naive `[N, V]` figure (1.53 GiB) is **not** what either path costs — the byte-by-byte reconciliation is §4b below. |

### The chunked cross-entropy (written by hand)

```mermaid
flowchart LR
    subgraph ord["ordinary — one shot"]
      h1["h [N, D]"] --> g1["logits [N, V]<br/>(materialised)"] --> l1["loss"] --> b1["backward<br/>grad [N, V]"]
    end
    subgraph chk["chunked — N / chunk passes"]
      h2["h — one chunk"] --> g2["logits [chunk, V]"] --> l2["+= chunk loss"] --> b2["backward chunk<br/>free graph"] --> h2
    end
    ord -. "5.99 GiB" .-> R["3.3× less"]
    chk -. "1.83 GiB" .-> R
```

The subtlety that actually matters for memory: computing the whole chunked loss and then calling `.backward()` once keeps *every* chunk's graph alive at once and defeats the point. The implementation back-propagates **each chunk immediately** (pre-scaled by `1/total_count`) so autograd frees that chunk before the next is computed.

## 4b. Where the memory goes — reconciling the numbers

The naive `[N, V]` logits figure is not what either path costs. The notebook now prints this account (values shown for the `N = 16,384`, `bf16`, `V = 50,257`, `chunk = 1,024` run):

| | bytes | why |
|---|---:|---|
| **Ordinary** | | |
| logits `[N, V]` bf16 | 1.53 GiB | materialised by the matmul |
| log-softmax `[N, V]` **fp32** | 3.07 GiB | `F.cross_entropy` up-casts to fp32 for numerical stability |
| grad of logits `[N, V]` bf16 | 1.53 GiB | the backward pass needs it too |
| hidden + weight + their grads | ~0.13 GiB | small next to the `[N, V]` tensors |
| **predicted ≈ 6.3 GiB** | | **measured 5.99 GiB** ✓ (logits and log-softmax don't fully coexist) |
| **Chunked** (`chunk = 1,024`) | | |
| one chunk: logits + fp32 log-softmax + grad | ~0.39 GiB | transient — freed after each chunk's backward |
| hidden `[N, D]` + grad, weight `[V, D]` + grad | ~0.13 GiB | **does not shrink with `chunk`** — this is the floor |
| CUDA context + cuBLAS workspace | ~1.3 GiB | fixed cost of touching the GPU at all |
| **predicted ≈ 1.8 GiB** | | **measured 1.83 GiB** ✓ |

So the ordinary path is **~3× the naive figure** (backward gradient + fp32 log-softmax), and the chunked path **can't reach the one-chunk ideal** (~0.1 GiB) because the CUDA context and the `N`-independent hidden/weight tensors dominate at this size. The ratio is only **3.3×** here because 1.53 GiB isn't a wall. Push `N`, the context, or `V` up and the ordinary path's `3·N·V` term runs away while the chunked path stays flat — the notebook now measures at **N = 32,768** (ordinary ≈ 12 GiB) to make that visible; the class notes cite ~64× at frontier `V` and context.

---

## 5. Part 2 — a second head predicting `t+2`

Two **untied, architecturally identical** heads sit on the shared trunk. The only difference is the target: Head 1 predicts `t+1`, Head 2 predicts `t+2`. Keeping them identical is deliberate — if Head 1 were tied to the embedding and Head 2 weren't, a loss gap could be "t+2 is harder" *or* "the tied head is more constrained." Untying both isolates the one effect the assignment asks about.

```mermaid
flowchart LR
    h["shared hidden state h<br/>[B, T, 512]"] --> H1["Head 1 (untied)<br/>[V, 512]"]
    h --> H2["Head 2 (untied)<br/>[V, 512]"]
    H1 --> L1["L1 = CE(·, tokens[:, 1:])<br/>predict t+1"]
    H2 --> L2["L2 = CE(·, tokens[:, 2:])<br/>predict t+2"]
    L1 --> SUM(["L_total = L1 + L2<br/>(added, not averaged)"])
    L2 --> SUM
    SUM --> OPT["one AdamW step"]
```

### The two losses

Single-step training loss is one random minibatch — noisy by **±0.2 nats**. Every "settled" figure below is a **trailing mean over the last 400 steps**, not the final data point.

| Head | Predicts | Loss: start → final → **settled** | Perplexity (settled) |
|---|---|---|---|
| **Head 1** | `t+1` | 10.892 → 4.717 → **4.62** | ~102 |
| **Head 2** | `t+2` | 10.954 → 5.985 → **5.88** | ~359 |
| **Sum** (optimised) | — | 21.845 → 20.365 → **10.51** | — |
| *baseline (single tied `t+1` head)* | `t+1` | 10.939 → 4.651 → *4.55* | *~95* |

**Gap `L2 − L1`:** `+0.06` (step 0, both heads random) → settled **`+1.26 nats`** (final-step `+1.27`; whole-run mean `+1.06`; post-warm-up mean `+1.12`). **`L2 > L1` at 100 % of the 4,000 steps** — the trailing-mean gap is dozens of standard errors above zero.

**Cost to the `t+1` objective:** the MTP `t+1` head settles at **`+0.068 nats`** above the standalone baseline — below the ±0.2-nat single-step swing but ~3× the standard error of the trailing mean, so a small *resolved* cost, not zero. And the comparison is unfair to the two-head model: its `t+1` head is **untied** and also carries the `t+2` gradient, while the baseline's is **tied** (tying usually helps a little).

![t+1 vs t+2 head losses over training](./loss_harness/assets/mtp_losses.png)

![Baseline single-head (t+1) training loss](./loss_harness/assets/baseline_loss.png)

### Tracked in Aim

| Head 1 · `t+1` | Head 2 · `t+2` | Sum · `L1 + L2` |
|---|---|---|
| ![](./loss_harness/assets/mtp_loss_head1_aim.png) | ![](./loss_harness/assets/mtp_loss_head2_aim.png) | ![](./loss_harness/assets/mtp_loss_sum_aim.png) |

| Baseline loss | Baseline perplexity |
|---|---|
| ![](./loss_harness/assets/baseline_loss_aim.png) | ![](./loss_harness/assets/perplexity_baseline_aim.png) |

---

## 6. Commentary — what the result actually says

- **The gap is the finding.** At step 0 both heads are random, so `L2 − L1 ≈ 0` — noise. As training shapes the *shared* trunk, the gap opens to a stable **~1.12–1.26 nats** (trailing mean) and never closes. That is the multi-token-prediction claim made concrete: predicting `t+2` from the same hidden state carries **one extra step of genuine, irreducible uncertainty** about what the text does next. Its entropy floor is simply higher.

- **In perplexity terms:** the model settles at effectively **~100** options for the very next token, but **~360** for the token after that. Same context, same representation — the further-out prediction is a measurably harder question.

- **Both curves fall together, not apart.** They share one trunk, so anything that improves the hidden state helps both objectives at once. The `t+2` head is *extra supervision on the same representation*, not a competing task fighting for capacity.

- **The second head is cheap on the primary objective — but not literally free.** Trailing means: standalone baseline **4.55**, MTP `t+1` head **4.62** — a **+0.068 nat** cost, small enough to sit inside a single step's noise but cleanly resolved in the mean (~3 SE). Read against the confound it *understates* the two-head model: the baseline head is tied (helps), the MTP head is untied and shares its trunk gradient with `t+2`. Net: adding a `t+2` head barely touches next-token quality at this scale.

- **The sum is what's optimised**, and it's roughly `2×` a single head because the two losses are **added, not averaged** — so Head 2 pulls on the shared trunk exactly as hard as Head 1. If you wanted the `t+1` objective to dominate you would weight the sum (`L1 + λ·L2`); here `λ = 1`.

- **The honest cost of MTP** (not paid here, but real at scale): each extra head is another `V × D` matrix — **25.7M params** in this config, as large as the embedding table. Four dense heads would more than double the model. This is the argument for a *factored* output head.

---

## 7. Reproducing this

```bash
cd loss_harness
uv sync                                              # torch (CUDA), tiktoken, datasets, aim, jupyter, ...

# fast correctness pass — tiny model, WikiText-2, 20 steps, CPU, ~2 min
S9_SMOKE=1 uv run python dry_run.py

# the real run (GPU) — ~30 min; §3.7 memory now needs ~14 GiB free (N = 32,768)
uv run jupyter nbconvert --to notebook --execute --inplace session9_loss_harness.ipynb
uv run aim up                                        # browse tracked runs at http://localhost:43800
```

The notebook reads one environment variable, `S9_SMOKE`: unset it for the full config above, set it to `1` for a 2-minute CPU smoke run. Tokenised WikiText ids are cached under `loss_harness/.cache/`.

> **⏳ This notebook has un-executed changes from the review revision** (memory reconciliation + larger `N`, trailing-mean Part 2 reporting, the trained-boundary §6.2 cell). Re-run the GPU command above to refresh `assets/results.json` and the ⏳-marked numbers — the code, structure, and analysis are final.

---

## 8. Project layout

```
session9/
├── README.md                      ← you are here (the submission)
├── webapp/                        ← visual walkthrough (dark/light, no build step)
│   ├── index.html
│   ├── style.css
│   ├── data.js                    ← every number from the run
│   └── app.js
└── loss_harness/
    ├── session9_loss_harness.ipynb   ← the whole assignment, executed top to bottom
    ├── src/harness/
    │   ├── model.py                  ← TinyGPT trunk + OutputHead (tied/untied)
    │   ├── data.py                   ← WikiText load, batched+cached tokenize, packing, padding
    │   ├── losses.py                 ← ordinary + chunked cross-entropy, perplexity, peak-mem
    │   └── utils.py                  ← seeding, device, shape-printer, param dedup
    ├── assets/                       ← plots + results.json written by the notebook
    ├── pyproject.toml / uv.lock      ← uv-managed dependencies
    └── build_notebook.py             ← regenerates the notebook from cell source (dev only, gitignored)
```

## 9. Notes on a few decisions

- **Why WikiText-103 + GPT-2 BPE, not a toy vocabulary.** "Print the strings, not the ids" only means something with a real sub-word vocabulary — misalignment shows up as whole sub-words shifted by one. A char-level toy vocab would hide it.
- **The tied-weight double-update trap.** Tying reuses `tok_emb.weight` as the head weight — the *same* `nn.Parameter`. Building an optimizer from `list(model.parameters()) + list(head.parameters())` lists that tensor twice and AdamW updates it twice per step. `collect_params()` dedups by tensor identity before every optimizer in the notebook.
- **Peak memory is measured *and* reconciled** — `torch.cuda.max_memory_allocated` / `max_memory_reserved` bracket each path, and the notebook prints a byte-by-byte theoretical account (§4b) so the measured number is explained, not just reported.
- **The boundary demo diverges slightly from the real packer.** The isolated §3.4 demo concatenates two documents *raw*; `pack_documents` (used for the training corpus) *does* insert `<|endoftext|>` at joins but still adds no loss mask there. The demo drops the EOT only to make the bad pair a single unambiguous position. It is **not** wired into the Part 2 training loops — a common simplification once a run spans thousands of steps — but §6.2 re-checks it on the trained head to prove the point isn't hypothetical.
- **This submission was revised after a review pass.** Changes: the memory section now reconciles measured-vs-theoretical and measures at a larger `N`; Part 2 reports trailing-window means with standard errors rather than final-step losses; the boundary check is re-run on the trained model (§6.2); the shift check's automated line is framed as a slice check, not a loss-alignment guarantee. **The notebook must be re-executed on GPU** to refresh `assets/results.json` and the §7 / §6.2 numbers — the analysis and structure are final, the pending numbers are marked in the tables above.
