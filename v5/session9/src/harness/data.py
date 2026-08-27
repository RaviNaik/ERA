"""Dataset loading, tokenization, packing, and batching utilities.

Dataset: WikiText-2 (raw), via Hugging Face `datasets`. It's small enough
to tokenize and pack in seconds on a laptop CPU, real enough (Wikipedia
prose, genuine document boundaries) to make the padding/packing/shift
demonstrations meaningful rather than synthetic.

Tokenizer: tiktoken's GPT-2 encoding (50,257 tokens, real subword vocab).
Using a real tokenizer -- not a toy char-level one -- is what makes
"print the actual token strings" a real check rather than a formality:
GPT-2 BPE tokens are genuine sub-words, so misalignment is obvious when
you read them.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import tiktoken
import torch


EOT_ID = 50256  # GPT-2's <|endoftext|>, doubles as our document separator


def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


@dataclass
class PackedCorpus:
    """One long stream of token ids with <|endoftext|> between documents,
    plus a boolean tensor marking which positions are the first token of a
    new document (i.e. immediately after an EOT) -- the boundary map used
    for masking the join between packed documents.
    """
    ids: torch.Tensor          # [N] int64
    is_doc_start: torch.Tensor  # [N] bool


def load_and_tokenize(
    split: str = "train",
    config: str = "wikitext-103-raw-v1",
    max_docs: int | None = None,
    cache_dir: str | None = ".cache",
) -> list[list[int]]:
    """Load a WikiText config and tokenize each non-empty line as one document.

    Returns a list of token-id lists (one per document), each ending
    implicitly at EOT when packed (EOT itself is inserted at pack time).

    WikiText-103-raw (~117M GPT-2 tokens) is the default "decent size"
    corpus; pass ``config="wikitext-2-raw-v1"`` for the ~2M-token smoke
    corpus. Tokenized ids are cached under ``cache_dir`` keyed by
    (config, split, max_docs) so repeat runs skip re-encoding.
    """
    import pickle

    key = f"{config}__{split}__{max_docs}"
    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        digest = hashlib.md5(key.encode()).hexdigest()[:12]
        cache_path = os.path.join(cache_dir, f"tok_{config}_{split}_{digest}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    from datasets import load_dataset

    # "Salesforce/wikitext" is the current parquet-backed mirror of the classic
    # WikiText dataset; the original "wikitext" repo's loading script is no
    # longer compatible with recent huggingface_hub releases.
    ds = load_dataset("Salesforce/wikitext", config, split=split)
    enc = get_tokenizer()

    # WikiText marks section headings like " = Title = " -- treat every
    # non-empty line as its own "document" for packing purposes. This gives
    # us plenty of genuine document boundaries to pack and mask. Encode in
    # batches so a 117M-token corpus tokenizes in ~a minute, not many.
    texts = [t.strip() for t in ds["text"]]
    texts = [t for t in texts if t]

    docs: list[list[int]] = []
    BATCH = 8192
    for i in range(0, len(texts), BATCH):
        for ids in enc.encode_ordinary_batch(texts[i : i + BATCH]):
            if len(ids) < 4:
                continue
            docs.append(ids)
            if max_docs is not None and len(docs) >= max_docs:
                break
        if max_docs is not None and len(docs) >= max_docs:
            break

    if cache_path is not None:
        with open(cache_path, "wb") as f:
            pickle.dump(docs, f, protocol=pickle.HIGHEST_PROTOCOL)
    return docs


def pack_documents(docs: list[list[int]], seq_len: int, n_sequences: int, seed: int = 0) -> PackedCorpus:
    """Concatenate documents with EOT separators into `n_sequences` chunks
    of exactly `seq_len` tokens each (the standard packing scheme used to
    avoid wasting compute on padding). Returns ids plus a doc-start mask.
    """
    import random
    rng = random.Random(seed)
    order = list(range(len(docs)))
    rng.shuffle(order)

    stream: list[int] = []
    starts: list[bool] = []
    di = 0
    # +2 so both a t+1 and a t+2 target exist for the last input position
    # (make_batches only needs +1; make_mtp_batches needs +2).
    target_len = seq_len * n_sequences + 2
    while len(stream) < target_len:
        doc = docs[order[di % len(order)]]
        di += 1
        stream.append(EOT_ID)
        starts.append(True)  # EOT itself begins a fresh "slot"; real first token flagged below
        for j, tok in enumerate(doc):
            stream.append(tok)
            starts.append(j == 0)
    stream = stream[:target_len]
    starts = starts[:target_len]
    return PackedCorpus(
        ids=torch.tensor(stream, dtype=torch.long),
        is_doc_start=torch.tensor(starts, dtype=torch.bool),
    )


def make_batches(packed: PackedCorpus, seq_len: int, batch_size: int):
    """Yield [B, seq_len+1] slices (input+target share this via shifting)
    from a packed corpus, purely by reshaping -- no padding needed since
    the corpus was packed to an exact multiple of seq_len.
    """
    n_positions = (len(packed.ids) - 1) // seq_len
    ids = packed.ids[: n_positions * seq_len + 1]
    starts_full = packed.is_doc_start[: n_positions * seq_len + 1]

    chunks = []
    boundary_chunks = []
    for i in range(n_positions):
        s = i * seq_len
        chunks.append(ids[s: s + seq_len + 1])
        boundary_chunks.append(starts_full[s + 1: s + seq_len + 1])  # aligned to targets
    chunks = torch.stack(chunks)              # [n_positions, seq_len+1]
    boundary_chunks = torch.stack(boundary_chunks)  # [n_positions, seq_len] aligned with targets

    for i in range(0, len(chunks) - batch_size + 1, batch_size):
        block = chunks[i:i + batch_size]
        bmask = boundary_chunks[i:i + batch_size]
        tokens = block[:, :-1]
        targets = block[:, 1:]
        yield tokens, targets, bmask


def build_padded_batch(docs: list[list[int]], n_docs: int, pad_to: int | None = None):
    """Take `n_docs` short, *unpacked* documents of differing lengths and
    left-pack + right-pad them into one [B, T] batch, to demonstrate
    padding masking. Returns tokens, targets, and a boolean `valid` mask
    aligned with targets (True = real next-token prediction to train on).
    """
    chosen = docs[:n_docs]
    max_len = max(len(d) for d in chosen)
    T = pad_to or max_len

    tokens = torch.full((n_docs, T), EOT_ID, dtype=torch.long)
    valid = torch.zeros((n_docs, T - 1), dtype=torch.bool)
    lengths = []
    for i, d in enumerate(chosen):
        d = d[:T]
        tokens[i, :len(d)] = torch.tensor(d, dtype=torch.long)
        lengths.append(len(d))
        # position j (0-indexed) in the *targets* array predicts token j+1;
        # it's valid iff both token j and token j+1 are real (non-pad).
        n_valid_targets = max(len(d) - 1, 0)
        valid[i, :n_valid_targets] = True

    targets = tokens[:, 1:].clone()
    inputs = tokens[:, :-1].clone()
    key_padding_mask = torch.zeros((n_docs, T), dtype=torch.bool)
    for i, L in enumerate(lengths):
        key_padding_mask[i, :L] = True
    return inputs, targets, valid, key_padding_mask, lengths
