"""What does a model actually write for the HumanEval prompts of the evaluation? Same lm-evaluation-harness task as the prior-task
evaluation (humaneval_instruct, chat template, bf16, greedy), but with the replies saved and shown.

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/humaneval_check.py --tag ft --ckpt_dir runs/<run>/train_ft --model_dir ... --out he_ft.json

--tag base | ft | lora | dt (dt needs --mask_dir). Prints pass@1 of the first --n problems and, for each, whether the reply has a
code block, a `def`, a `####` answer, plus the first characters. A model that fine-tuning pushed to answer in the GSM8K
format shows `####` and no code; a harness/extraction fault shows normal code and still a failing score."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

from dtphi import official  # noqa: E402,F401
from dtphi.modeling import load_model, load_tokenizer  # noqa: E402
from dtphi.phimoe import modeling_phimoe  # noqa: E402


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tag", required=True, choices=["base", "ft", "lora", "dt"])
    p.add_argument("--model_dir", required=True)
    p.add_argument("--ckpt_dir", default="", help="train_ft or train_lora folder (--tag ft / lora)")
    p.add_argument("--mask_dir", default="", help="official mask folder (--tag dt)")
    p.add_argument("--n", type=int, default=20, help="first n HumanEval problems")
    p.add_argument("--batch", type=int, default=0, help="lm-eval batch size, default n (one batch)")
    p.add_argument("--n_gpus", type=int, default=1, help="0 = CPU (tests)")
    p.add_argument("--out", default="")
    p.add_argument("--show", type=int, default=6, help="replies printed in full")
    p.add_argument("--decode_fix", choices=["on", "off"], default="on",
                   help="on: keep the first space of the decoded reply (the harness decode loses it, so every body starts with "
                        "3 spaces instead of 4 and fails); off: the evaluation's own decode")
    a = p.parse_args(argv)
    modeling_phimoe.FAST_MOE_MAX_TOKENS = 1024
    tok = load_tokenizer(a.model_dir)
    model = load_model(a.model_dir, a.n_gpus)
    if a.tag in ("ft", "lora"):
        from dtphi.train_baseline import apply_ft, apply_lora
        print("weights:", (apply_ft if a.tag == "ft" else apply_lora)(model, a.ckpt_dir), flush=True)
    elif a.tag == "dt":
        import apply_mask  # official
        print("mask:", apply_mask.apply_mask(model=model, masks=apply_mask.load_mask(mask_dir=a.mask_dir)), flush=True)
    from lm_eval import evaluator
    from dtphi.eval_prior import make_lm
    lm = make_lm(model, tok, a.batch or a.n, keep_leading_space=a.decode_fix == "on")
    res = evaluator.simple_evaluate(model=lm, tasks=["humaneval_instruct"], apply_chat_template=True, limit=a.n,
                                    confirm_run_unsafe_code=True, log_samples=True, bootstrap_iters=0)
    pass1 = res["results"]["humaneval_instruct"]["pass@1,create_test"] if "pass@1,create_test" in res["results"]["humaneval_instruct"] \
        else {k: v for k, v in res["results"]["humaneval_instruct"].items() if "pass" in k}
    rows = []
    for s in res["samples"]["humaneval_instruct"]:
        raw = s["resps"][0][0] if s["resps"] and s["resps"][0] else ""
        rows.append({"task_id": s["doc"].get("task_id"), "reply": raw, "has_code_block": "```" in raw,
                     "has_def": "def " in raw, "has_####": "####" in raw, "chars": len(raw)})
    summary = {"tag": a.tag, "decode_fix": a.decode_fix, "n": len(rows), "pass@1": pass1, "with_code_block": sum(r["has_code_block"] for r in rows),
               "with_def": sum(r["has_def"] for r in rows), "with_####": sum(r["has_####"] for r in rows)}
    print(json.dumps(summary), flush=True)
    for r in rows[:a.show]:
        print(f"\n--- {r['task_id']} ({r['chars']} chars) code_block={r['has_code_block']} def={r['has_def']} ####={r['has_####']}\n"
              f"{r['reply'][:700]}")
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
