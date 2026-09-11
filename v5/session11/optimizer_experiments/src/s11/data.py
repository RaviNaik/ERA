"""Char-level Tiny Shakespeare loader.

A tiny, fully-deterministic dataset: 1,115,394 characters, 65 unique symbols.
We keep a 90/10 train/val split (the nanoGPT convention) and draw random
contiguous windows of ``block_size+1`` for each batch.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "tinyshakespeare.txt")


@dataclass
class CharDataset:
    text: str
    stoi: dict
    itos: dict
    train_ids: np.ndarray
    val_ids: np.ndarray

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def get_batch(self, split: str, batch_size: int, block_size: int,
                  device: str = "cpu", generator: torch.Generator | None = None):
        data = self.train_ids if split == "train" else self.val_ids
        ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
        x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
        if device.startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


def _ensure_data(path: str):
    path = os.path.abspath(path)
    if os.path.exists(path):
        return path
    import urllib.request
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"downloading Tiny Shakespeare -> {path}")
    urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    return path


def load_char_dataset(path: str = DATA_PATH) -> CharDataset:
    path = _ensure_data(path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.uint16)
    n = int(0.9 * len(ids))
    return CharDataset(text, stoi, itos, ids[:n], ids[n:])
