"""BPE-tokenized WikiText data pipeline for the session 13 LLM experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class TokenDataset:
    data_dir: Path = DATA_DIR
    train_file: str = "train.bin"
    val_file: str = "val.bin"
    vocab_size: int = 50257

    @property
    def train_path(self) -> Path:
        return self.data_dir / self.train_file

    @property
    def val_path(self) -> Path:
        return self.data_dir / self.val_file

    def split_data(self, split: str) -> np.memmap:
        path = self.train_path if split == "train" else self.val_path
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist; run prepare_wikitext() first"
            )
        return np.memmap(path, dtype=np.uint16, mode="r")

    def num_tokens(self, split: str) -> int:
        return len(self.split_data(split))

    def windows(
        self,
        split: str,
        starts: np.ndarray | torch.Tensor,
        block_size: int,
        device: str | torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gathers (x, y) windows beginning at `starts` in one vectorized read."""
        data = self.split_data(split)
        starts = np.asarray(starts, dtype=np.int64)
        offsets = starts[:, None] + np.arange(block_size + 1, dtype=np.int64)
        chunk = torch.from_numpy(data[offsets].astype(np.int64))
        x, y = chunk[:, :-1], chunk[:, 1:]
        device = torch.device(device)
        if device.type == "cuda":
            return (
                x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True),
            )
        return x.contiguous().to(device), y.contiguous().to(device)

    def get_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        device: str | torch.device,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniformly random windows (sampling with replacement)."""
        hi = self.num_tokens(split) - block_size - 1
        if hi <= 0:
            raise ValueError(
                f"{split} split has only {hi + block_size + 1} tokens for block_size={block_size}"
            )
        ix = torch.randint(hi, (batch_size,), generator=generator)
        return self.windows(split, ix.numpy(), block_size, device)

    def eval_starts(
        self, split: str, block_size: int, max_windows: int | None, seed: int = 0
    ) -> np.ndarray:
        """Fixed evaluation windows, identical for every run.

        `max_windows=None` returns every non-overlapping window of the split
        (a full pass). Otherwise a fixed-seed random subset of those windows
        is returned, so all experiments are scored on exactly the same tokens.
        """
        n_windows = (self.num_tokens(split) - 1) // block_size
        starts = np.arange(n_windows, dtype=np.int64) * block_size
        if max_windows is not None and max_windows < n_windows:
            rng = np.random.default_rng(seed)
            starts = np.sort(rng.choice(starts, size=max_windows, replace=False))
        return starts


def prepare_wikitext(
    data_dir: str | Path = DATA_DIR,
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    tokenizer_name: str = "gpt2",
    force: bool = False,
) -> dict:
    """Download WikiText, tokenize with GPT-2 BPE, and cache uint16 token files."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_path = data_dir / "meta.json"
    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"
    if not force and train_path.exists() and val_path.exists() and meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))

    from datasets import load_dataset
    import tiktoken

    enc = tiktoken.get_encoding(tokenizer_name)
    try:
        ds = load_dataset(dataset_name, dataset_config)
    except Exception:
        if dataset_name != "wikitext":
            raise
        dataset_name = "Salesforce/wikitext"
        ds = load_dataset(dataset_name, dataset_config)

    def encode_split(split: str) -> np.ndarray:
        token_chunks: list[list[int]] = []
        total = 0
        for row in ds[split]:
            text = row.get("text", "")
            if not text.strip():
                continue
            ids = enc.encode_ordinary(text)
            ids.append(enc.eot_token)
            token_chunks.append(ids)
            total += len(ids)
        out = np.empty(total, dtype=np.uint16)
        pos = 0
        for chunk in token_chunks:
            out[pos : pos + len(chunk)] = np.array(chunk, dtype=np.uint16)
            pos += len(chunk)
        return out

    train_ids = encode_split("train")
    val_ids = encode_split("validation")
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)
    meta = {
        "dataset": f"{dataset_name}/{dataset_config}",
        "tokenizer": tokenizer_name,
        "vocab_size": enc.n_vocab,
        "train_tokens": int(train_ids.size),
        "val_tokens": int(val_ids.size),
        "dtype": "uint16",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_token_dataset(data_dir: str | Path = DATA_DIR) -> TokenDataset:
    data_dir = Path(data_dir)
    meta_path = data_dir / "meta.json"
    vocab_size = 50257
    if meta_path.exists():
        vocab_size = int(
            json.loads(meta_path.read_text(encoding="utf-8"))["vocab_size"]
        )
    return TokenDataset(data_dir=data_dir, vocab_size=vocab_size)
