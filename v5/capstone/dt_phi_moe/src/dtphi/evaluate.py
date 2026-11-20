"""Evaluate the base model (--tag base) or the base model with a saved Deletion Tuning mask (--tag dt).

Parts (--parts, comma separated):
  gsm8k       GSM8K test, the training format (question + newline, greedy): the number the SFT run optimises
  gsm8k_chat  GSM8K test with the chat template and the '#### <number>' instruction (how the instruct model is used)
  kl          forward KL base -> masked on the base model's own gsm8k replies (dt only; needs a base run with gsm8k)
  team10      Team 10 Indic and safety row sets
  prior       six prior-task benchmarks with lm-evaluation-harness (retention)
The mask is applied with the official apply_mask.py; the KL log-probabilities come from the official
retention_forward_kl.py. Results go to <out_dir>/eval_<tag>/."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from . import data, official
from .eval_gsm8k import forward_kl, generate_gsm8k, gsm8k_metrics
from .modeling import load_model, load_tokenizer

PARTS = ("gsm8k", "gsm8k_chat", "kl", "team10", "prior")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tag", required=True, choices=["base", "dt"])
    p.add_argument("--model_dir", required=True)
    p.add_argument("--data_dir", required=True, help="folder holding main/test-*.parquet")
    p.add_argument("--team10_dir", default="", help="folder with indic_v1.jsonl and safety_v1.jsonl")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--mask_dir", default="", help="official mask folder (model.safetensors + config.json), --tag dt")
    p.add_argument("--n_gpus", type=int, default=2)
    p.add_argument("--parts", default="gsm8k,gsm8k_chat,kl,team10,prior")
    p.add_argument("--gsm8k_limit", type=int, default=0)
    p.add_argument("--gen_batch", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=400)
    p.add_argument("--kl_limit", type=int, default=0)
    p.add_argument("--team10_max_new_tokens", type=int, default=200)
    p.add_argument("--prior_tasks", default="hellaswag,truthfulqa_mc2,mmlu,ifeval,winogrande,humaneval_instruct")
    p.add_argument("--prior_limit", type=int, default=0)
    p.add_argument("--prior_batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fast_moe_tokens", type=int, default=1024,
                   help="dense MoE path for forwards of at most this many tokens (decoding); 0 = original expert loop")
    return p.parse_args(argv)


def _dump(path: Path, recs: list[dict]) -> None:
    with path.open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def run(args: argparse.Namespace) -> dict:
    official.verify_vendored()
    import apply_mask  # official
    import retention_forward_kl  # official
    torch.manual_seed(args.seed)
    parts = [x for x in args.parts.split(",") if x]
    bad = set(parts) - set(PARTS)
    if bad:
        raise ValueError(f"unknown parts {sorted(bad)}")
    if args.tag == "base" and "kl" in parts:
        parts.remove("kl")
    out = Path(args.out_dir) / f"eval_{args.tag}"
    base_gen_path = Path(args.out_dir) / "eval_base" / "gsm8k_generations.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(args.model_dir)
    all_rows = data.load_gsm8k(args.data_dir, "test")
    rows = all_rows[:args.gsm8k_limit] if args.gsm8k_limit else all_rows
    from .phimoe import modeling_phimoe
    modeling_phimoe.FAST_MOE_MAX_TOKENS = args.fast_moe_tokens
    model = load_model(args.model_dir, args.n_gpus)
    summary: dict = {"tag": args.tag, "args": vars(args)}
    pad_id = tok.pad_token_id

    if args.tag == "dt":
        masks = apply_mask.load_mask(mask_dir=args.mask_dir)
        if "kl" in parts:
            if not base_gen_path.exists():
                raise FileNotFoundError(f"{base_gen_path} missing: run the base evaluation (gsm8k part) first")
            recs = [json.loads(l) for l in base_gen_path.read_text().splitlines()]
            recs = [r for r in recs if r["response_ids"]]
            if args.kl_limit:
                recs = recs[:args.kl_limit]
            prompts, resps = [r["prompt_ids"] for r in recs], [r["response_ids"] for r in recs]
            t0 = time.time()
            base_lp = retention_forward_kl.response_logprobs(model=model, pad_id=pad_id, prompt_ids=prompts,
                                                             response_ids=resps)
            summary["kl_base_seconds"] = round(time.time() - t0, 1)
        deleted = apply_mask.apply_mask(model=model, masks=masks)
        summary["weights_zeroed"] = deleted
        print(f"mask applied: {deleted:,d} weights zeroed", flush=True)
        if "kl" in parts:
            masked_lp = retention_forward_kl.response_logprobs(model=model, pad_id=pad_id, prompt_ids=prompts,
                                                               response_ids=resps)
            summary["forward_kl"] = forward_kl(base_lp, masked_lp)
            print("forward KL:", summary["forward_kl"], flush=True)

    for part, fn in (("gsm8k", data.plain_prompt_ids), ("gsm8k_chat", data.chat_prompt_ids)):
        if part in parts:
            t0 = time.time()
            recs = generate_gsm8k(model, tok, rows, fn, args.gen_batch, args.max_new_tokens)
            summary[part] = gsm8k_metrics(recs) | {"seconds": round(time.time() - t0, 1),
                                                   "max_new_tokens": args.max_new_tokens}
            _dump(out / f"{part}_generations.jsonl", recs)
            print(f"{part}:", summary[part], flush=True)

    if "team10" in parts:
        from .eval_team10 import run_team10
        res = run_team10(model, tok, args.team10_dir, args.gen_batch, args.team10_max_new_tokens)
        summary["team10"] = {k: {x: v[x] for x in ("n", "exact", "by_kind", "by_lang") if x in v}
                             | ({"refusal_rate": v["refusal_rate"], "helped_rate": v["helped_rate"]}
                                if "refusal_rate" in v else {}) for k, v in res.items()}
        (out / "team10_reports.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print("team10:", json.dumps(summary["team10"], ensure_ascii=False), flush=True)

    if "prior" in parts:
        from .eval_prior import run_prior_tasks
        t0 = time.time()
        pr = run_prior_tasks(model, tok, [t for t in args.prior_tasks.split(",") if t], args.prior_batch,
                             args.prior_limit)
        summary["prior"] = pr | {"seconds": round(time.time() - t0, 1)}
        print("prior tasks:", pr["scores"], "mean", round(pr["prior_task_mean"], 4), flush=True)

    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
