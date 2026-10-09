"""BPE-tokenised datasets cached as flat ``uint16`` token files.

* ``wikitext103`` — Salesforce/wikitext (wikitext-103-raw-v1), GPT-2 BPE, ~120M train tokens.
* ``synthetic``   — a tiny seeded Markov-style corpus; only used by the smoke test.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .utils import work_root


def default_data_dir(dataset: str) -> Path:
    return Path(work_root()) / "data" / dataset


@dataclass
class TokenDataset:
    data_dir: Path
    vocab_size: int = 50257

    def split_data(self, split: str) -> np.memmap:
        path = self.data_dir / ("train.bin" if split == "train" else "val.bin")
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; call prepare_dataset() first")
        return np.memmap(path, dtype=np.uint16, mode="r")

    def num_tokens(self, split: str) -> int:
        return len(self.split_data(split))

    def get_batch(self, split: str, batch_size: int, block_size: int, device,
                  generator: torch.Generator | None = None):
        """Uniformly random contiguous windows (sampling with replacement)."""
        data = self.split_data(split)
        hi = len(data) - block_size - 1
        if hi <= 0:
            raise ValueError(f"{split} split too short ({len(data)} tokens) for block {block_size}")
        ix = torch.randint(hi, (batch_size,), generator=generator).numpy()
        offs = ix[:, None] + np.arange(block_size + 1, dtype=np.int64)
        chunk = torch.from_numpy(data[offs].astype(np.int64))
        x, y = chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()
        if str(device).startswith("cuda"):
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)


def _prepare_synthetic(data_dir: Path, vocab: int = 512, n_train=200_000, n_val=20_000, seed=0) -> dict:
    """Order-1 Markov chain over ``vocab`` tokens: learnable, so loss visibly drops."""
    rng = np.random.default_rng(seed)
    trans = rng.dirichlet(np.full(vocab, 0.05), size=vocab)       # peaky rows
    cdf = trans.cumsum(1)

    def gen(n):
        out = np.empty(n, dtype=np.uint16)
        s = 0
        u = rng.random(n)
        for i in range(n):
            s = int(np.searchsorted(cdf[s], u[i]))
            s = min(s, vocab - 1)
            out[i] = s
        return out

    tr, va = gen(n_train), gen(n_val)
    tr.tofile(data_dir / "train.bin")
    va.tofile(data_dir / "val.bin")
    return {"dataset": "synthetic-markov", "tokenizer": "none", "vocab_size": vocab,
            "train_tokens": int(tr.size), "val_tokens": int(va.size)}


def _prepare_wikitext103(data_dir: Path) -> dict:
    from datasets import load_dataset
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

    def encode_split(split: str) -> np.ndarray:
        chunks, total = [], 0
        for row in ds[split]:
            text = row.get("text", "")
            if not text.strip():
                continue
            ids = enc.encode_ordinary(text)
            ids.append(enc.eot_token)
            chunks.append(np.asarray(ids, dtype=np.uint16))
            total += len(ids)
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint16)

    tr, va = encode_split("train"), encode_split("validation")
    tr.tofile(data_dir / "train.bin")
    va.tofile(data_dir / "val.bin")
    return {"dataset": "wikitext-103-raw-v1", "tokenizer": "gpt2-bpe", "vocab_size": enc.n_vocab,
            "train_tokens": int(tr.size), "val_tokens": int(va.size)}


def prepare_dataset(dataset: str = "wikitext103", data_dir: str | Path | None = None,
                    force: bool = False) -> dict:
    data_dir = Path(data_dir) if data_dir else default_data_dir(dataset)
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_path = data_dir / "meta.json"
    if not force and meta_path.exists() and (data_dir / "train.bin").exists():
        meta = json.loads(meta_path.read_text())
        print(f"[data] cached {dataset}: {meta['train_tokens']:,} train / {meta['val_tokens']:,} val tokens")
        return meta
    print(f"[data] preparing {dataset} -> {data_dir} (one-off)")
    meta = _prepare_synthetic(data_dir) if dataset == "synthetic" else _prepare_wikitext103(data_dir)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[data] {meta['train_tokens']:,} train / {meta['val_tokens']:,} val tokens")
    return meta


def load_dataset_tokens(dataset: str = "wikitext103", data_dir: str | Path | None = None) -> TokenDataset:
    data_dir = Path(data_dir) if data_dir else default_data_dir(dataset)
    prepare_dataset(dataset, data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    return TokenDataset(data_dir=data_dir, vocab_size=int(meta["vocab_size"]))
