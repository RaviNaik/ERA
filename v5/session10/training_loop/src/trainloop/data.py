"""Char-level Tiny Shakespeare, plus a variable-length micro-batch sampler.

Char-level on purpose: no tokenizer download, no network needed after the first
run, and the vocabulary is ~65 symbols so an untrained model's loss sits near
ln(65) which is a clean sanity anchor.

The variable-length sampler is what Part 3 (breaking gradient accumulation)
needs: micro-batches that hold *different numbers of real tokens*, so that
"average of the averages" and "total loss over total tokens" actually diverge.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# The 65 distinct characters of Tiny Shakespeare. Prepended to the offline
# fallback so `sorted(set(text))` still yields vocab_size == 65 and the
# ln(65) = 4.174 sanity anchor holds even with no network.
_VOCAB_PRIMER = (
    "\n !$&',-.3:;?"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz\n"
)

# A tiny offline fallback so the notebook still runs with no network. It is real
# Shakespeare (Sonnet 18 + a Hamlet fragment), repeated to a usable length.
_FALLBACK = (
    "Shall I compare thee to a summer's day?\n"
    "Thou art more lovely and more temperate:\n"
    "Rough winds do shake the darling buds of May,\n"
    "And summer's lease hath all too short a date;\n"
    "Sometime too hot the eye of heaven shines,\n"
    "And often is his gold complexion dimm'd;\n"
    "And every fair from fair sometime declines,\n"
    "By chance or nature's changing course untrimm'd;\n"
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles\n"
    "And by opposing end them.\n"
)


def _load_text() -> str:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _DATA_DIR / "tinyshakespeare.txt"
    if not path.exists():
        try:
            print(f"downloading Tiny Shakespeare -> {path}")
            urllib.request.urlretrieve(_URL, path)  # noqa: S310
        except Exception as exc:  # offline: use the bundled fallback
            print(f"download failed ({exc}); using bundled fallback corpus")
            path.write_text(_VOCAB_PRIMER + _FALLBACK * 400, encoding="utf-8")
    return path.read_text(encoding="utf-8")


@dataclass
class CharDataset:
    """Holds the encoded corpus and the char<->id maps."""

    train_ids: np.ndarray
    val_ids: np.ndarray
    stoi: dict[str, int]
    itos: dict[int, str]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]


def load_char_dataset(smoke: bool = False) -> CharDataset:
    text = _load_text()
    if smoke:
        text = text[:20000]
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    n = int(0.9 * len(ids))
    return CharDataset(train_ids=ids[:n], val_ids=ids[n:], stoi=stoi, itos=itos)


def get_batch(
    ds: CharDataset,
    split: str,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A plain fixed-length batch. Returns (x, y), each [batch_size, seq_len]."""
    data = ds.train_ids if split == "train" else ds.val_ids
    hi = len(data) - seq_len - 1
    ix = torch.randint(hi, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy(data[i : i + seq_len].copy()) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + seq_len].copy()) for i in ix])
    return x.to(device), y.to(device)


def get_var_micro_batch(
    ds: CharDataset,
    split: str,
    batch_size: int,
    real_len: int,
    pad_to: int,
    device: torch.device,
    pad_id: int = 0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A micro-batch whose sequences carry only `real_len` valid tokens each,
    padded out to `pad_to`. Returns (x, y, loss_mask), each [batch_size, pad_to].

    `loss_mask` is 1 on the `real_len` predicted positions and 0 on padding, so
    the number of tokens that actually contribute to the loss is
    `batch_size * real_len` -- deliberately different across micro-batches.
    """
    data = ds.train_ids if split == "train" else ds.val_ids
    hi = len(data) - real_len - 1
    ix = torch.randint(hi, (batch_size,), generator=generator)
    x = torch.full((batch_size, pad_to), pad_id, dtype=torch.long)
    y = torch.full((batch_size, pad_to), pad_id, dtype=torch.long)
    mask = torch.zeros((batch_size, pad_to), dtype=torch.float32)
    for b, i in enumerate(ix):
        i = int(i)
        x[b, :real_len] = torch.from_numpy(data[i : i + real_len].copy())
        y[b, :real_len] = torch.from_numpy(data[i + 1 : i + 1 + real_len].copy())
        mask[b, :real_len] = 1.0
    return x.to(device), y.to(device), mask.to(device)
