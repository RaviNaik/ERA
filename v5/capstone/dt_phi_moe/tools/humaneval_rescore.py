"""Rescore the replies saved by humaneval_check.py (--out he_*.json) two ways, with the same code_eval metric as lm-eval:
  as_is        prompt + reply (cut at the first ```), what the evaluation scores
  first_line   the same, with the indentation of the reply's first line set to 4 spaces (diagnostic only: shows how many failures
               are only the first-line indentation; it must not replace the evaluation number)
    HF_ALLOW_CODE_EVAL=1 .venv/bin/python tools/humaneval_rescore.py he_ft.json he_base.json"""
from __future__ import annotations

import json
import os
import re
import sys

os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")


def _cut(reply: str) -> str:
    i = reply.find("```")
    return reply if i == -1 else reply[:i]


def fix_first_line(reply: str) -> str:
    return re.sub(r"^[ \t]*", "    ", reply, count=1)


def score(docs_by_id: dict, rows: list[dict], fix: bool) -> tuple[float, list[bool]]:
    import evaluate
    code_eval = evaluate.load("code_eval")
    ok = []
    for r in rows:
        d = docs_by_id[r["task_id"]]
        body = _cut(r["reply"])
        body = fix_first_line(body) if fix else body
        res, _ = code_eval.compute(references=[d["test"] + "\n" + f"check({d['entry_point']})"],
                                   predictions=[[d["prompt"] + body]], k=[1], timeout=10.0)
        ok.append(res["pass@1"] == 1.0)
    return sum(ok) / len(ok), ok


def main(paths: list[str]) -> None:
    from datasets import load_dataset
    docs = {d["task_id"]: d for d in load_dataset("openai/openai_humaneval", split="test")}
    for p in paths:
        rows = json.load(open(p))["rows"]
        a, oa = score(docs, rows, False)
        b, ob = score(docs, rows, True)
        print(json.dumps({"file": p, "n": len(rows), "as_is": a, "first_line_fixed": b,
                          "fixed_only": [r["task_id"] for r, x, y in zip(rows, oa, ob) if y and not x],
                          "first_line_indents": sorted({len(re.match(r"[ \t]*", r["reply"]).group()) for r in rows})}))


if __name__ == "__main__":
    main(sys.argv[1:])
