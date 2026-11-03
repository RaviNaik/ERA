#!/usr/bin/env python3
"""Generation speed of one GGUF (tokens per second), for the gen_tok_s column of runs.csv.

    python speed.py model.gguf --out speed_Q4_K_M.json

Greedy, 256 new tokens from a fixed prompt, every layer on the GPU(s); the first (warm-up) call is not timed.
"""
import argparse, json, time
from pathlib import Path

from llama_cpp import Llama

PROMPT = "Explain in simple words why the sky is blue and why sunsets are red."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-ctx", type=int, default=2048)
    a = ap.parse_args()
    llm = Llama(model_path=str(a.gguf), n_ctx=a.n_ctx, n_gpu_layers=-1, seed=42, verbose=False)
    llm(PROMPT, max_tokens=16, temperature=0.0)
    runs = []
    for _ in range(3):
        t = time.perf_counter()
        out = llm(PROMPT, max_tokens=256, temperature=0.0)
        dt = time.perf_counter() - t
        n = out["usage"]["completion_tokens"]
        runs.append(n / dt)
    res = {"gguf": a.gguf.name, "tok_s_runs": [round(x, 1) for x in runs], "gen_tok_s": round(sorted(runs)[1], 1)}
    a.out.write_text(json.dumps(res, indent=1))
    print(res)


if __name__ == "__main__":
    main()
