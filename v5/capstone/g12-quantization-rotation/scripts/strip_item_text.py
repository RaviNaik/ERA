#!/usr/bin/env python3
"""Make the committable copy of the run folder: per-question rows without benchmark text or model output.

    python strip_item_text.py <raw runs dir> <results/runs>

TSAI's eval firewall (4 Nov 2026): benchmark item text never goes into a repo, fork or PR, and per-item logs with
text never go in Git. G12's lead also asked for no `target` or `response` in the jsonl files (PR #272). The kept
fields are task, doc_id, filter and the metric values (acc, acc_norm, exact_match, ...), which is what
paired.py and combine.py need. summary.json (scores, run metadata) and paired.md are copied unchanged.
"""
from __future__ import annotations

import json, shutil, sys
from pathlib import Path

DROP = {"target", "response", "prompt", "question", "context", "arguments", "resps", "filtered_resps", "doc"}


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    n_files = n_rows = 0
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        out = dst / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        if p.name.endswith(".samples.jsonl"):
            with open(p) as fi, open(out, "w") as fo:
                for line in fi:
                    row = {k: v for k, v in json.loads(line).items() if k not in DROP}
                    fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_rows += 1
            n_files += 1
        elif p.suffix in {".json", ".md", ".csv"}:
            shutil.copy2(p, out)
    # the check that matters: nothing with text is left in the committable copy
    for p in dst.rglob("*.samples.jsonl"):
        for line in open(p):
            bad = DROP & json.loads(line).keys()
            if bad:
                raise SystemExit(f"{p}: still has {sorted(bad)}")
    print(f"stripped {n_rows} rows in {n_files} files -> {dst}")


if __name__ == "__main__":
    main()
