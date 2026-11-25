"""Decoding speed of the evaluation path on one GPU, so CUDA_LAUNCH_BLOCKING=1 and =0 can be compared directly.

    CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=0 .venv/bin/python tools/gen_bench.py --model_dir ... --data_dir ...

Loads the model on one card, generates greedy replies for the first --n_rows GSM8K test questions (chat template) in
batches of --batch with the same code as the evaluation (dtphi.eval_gsm8k.generate_gsm8k), after one warm-up batch.
Prints one JSON line: seconds, new tokens per second and a hash of all replies. The hash must be the same for both flag
settings, otherwise the faster setting changes the results and must not be used."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from dtphi import data  # noqa: E402
from dtphi.eval_gsm8k import generate_gsm8k  # noqa: E402
from dtphi.modeling import load_model, load_tokenizer  # noqa: E402
from dtphi.phimoe import modeling_phimoe  # noqa: E402


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model_dir", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--n_gpus", type=int, default=1, help="0 = CPU (tests)")
    p.add_argument("--n_rows", type=int, default=64)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--fast_moe_tokens", type=int, default=1024)
    a = p.parse_args(argv)
    modeling_phimoe.FAST_MOE_MAX_TOKENS = a.fast_moe_tokens
    tok = load_tokenizer(a.model_dir)
    rows = data.load_gsm8k(a.data_dir, "test")[:a.n_rows]
    model = load_model(a.model_dir, a.n_gpus)
    generate_gsm8k(model, tok, rows[:a.batch], data.chat_prompt_ids, a.batch, a.max_new_tokens)   # warm-up
    if a.n_gpus:
        torch.cuda.synchronize()
    t0 = time.time()
    recs = generate_gsm8k(model, tok, rows, data.chat_prompt_ids, a.batch, a.max_new_tokens)
    if a.n_gpus:
        torch.cuda.synchronize()
    secs = time.time() - t0
    new = sum(len(r["response_ids"]) for r in recs)
    h = hashlib.sha256("\n".join(f"{r['idx']}:{r['text']}" for r in recs).encode()).hexdigest()[:16]
    out = {"CUDA_LAUNCH_BLOCKING": os.environ.get("CUDA_LAUNCH_BLOCKING"), "rows": len(rows), "batch": a.batch,
           "max_new_tokens": a.max_new_tokens, "seconds": round(secs, 1), "reply_tokens": new,
           "reply_tokens_per_s": round(new / secs, 1), "replies_sha": h, "correct": sum(r["correct"] for r in recs)}
    print(json.dumps(out), flush=True)
    return out


if __name__ == "__main__":
    main()
