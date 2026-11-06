#!/usr/bin/env python3
"""Check a GGUF file against G12's quantization rules and record what it holds.

    python check_gguf.py model.gguf --format Q4_K_M --out results/gguf_check_Q4_K_M.json

G12's rules (teams/12-quantization README): every quantized format keeps output.weight at Q8_0 and the router
(ffn_gate_inp) in F32 or BF16. Sarvam-30B also carries a per-expert routing bias (exp_probs_b); it is checked the same way.
Exits non-zero when a rule is broken. Needs the `gguf` package (llama.cpp's gguf-py, installed in the convert venv).
"""
from __future__ import annotations

import argparse, collections, json, sys
from pathlib import Path

from gguf import GGUFReader


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--format", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    r = GGUFReader(str(args.gguf), "r")
    types: dict[str, str] = {}
    n_params = 0
    hist: collections.Counter = collections.Counter()
    for t in r.tensors:
        types[t.name] = t.tensor_type.name
        n_params += int(t.n_elements)
        hist[t.tensor_type.name] += int(t.n_elements)

    def kind(substr: str) -> set[str]:
        return {v for k, v in types.items() if substr in k}

    # exact name: a substring test would also match every blk.N.attn_output.weight (Q4_K), a false failure
    router, bias, output = kind("ffn_gate_inp"), kind("exp_probs_b"), {v for k, v in types.items() if k == "output.weight"}
    arch = next((bytes(f.parts[f.data[0]]).decode() for k, f in r.fields.items() if k == "general.architecture"), "?")
    problems = []
    if not router:
        problems.append("no ffn_gate_inp (router) tensor found: is this an MoE GGUF?")
    elif not router <= {"F32", "BF16"}:
        problems.append(f"router is {sorted(router)}, must be F32 or BF16")
    if bias and not bias <= {"F32", "BF16"}:
        problems.append(f"expert bias is {sorted(bias)}, must be F32 or BF16")
    want_output = {"BF16"} if args.format == "bf16" else {"Q8_0"}
    if output != want_output:
        problems.append(f"output.weight is {sorted(output)}, expected {sorted(want_output)}")

    res = {"file": args.gguf.name, "format": args.format, "architecture": arch, "bytes": args.gguf.stat().st_size,
           "n_params": n_params, "bits_per_weight": round(args.gguf.stat().st_size * 8 / n_params, 2),
           "router_type": "/".join(sorted(router)), "expert_bias_type": "/".join(sorted(bias)),
           "output_type": "/".join(sorted(output)),
           "params_by_type": dict(hist), "ok": not problems, "problems": problems}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "params_by_type"}, indent=1))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
