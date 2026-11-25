"""Paired comparison of the tuned variants with each other (DT vs LoRA, DT vs full FT, LoRA vs full FT) on the saved GSM8K replies.
dtphi.paired compares each variant with the base model; this does the pairs between variants, with the same test (paired bootstrap
over questions, exact McNemar on the discordant questions).

    .venv/bin/python tools/paired_variants.py runs/<run> --suffix _full
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dtphi.paired import PARTS, VARIANTS, compare  # noqa: E402


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dir")
    p.add_argument("--suffix", default="", help="the evaluate --eval_suffix (_full for the larger evaluation)")
    p.add_argument("--resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    d = Path(a.run_dir)
    out, rows = {}, []
    for (ta, la), (tb, lb) in itertools.combinations(VARIANTS, 2):
        for part, pname in PARTS:
            pa = d / f"eval_{ta}{a.suffix}" / f"{part}_generations.jsonl"
            pb = d / f"eval_{tb}{a.suffix}" / f"{part}_generations.jsonl"
            if not (pa.exists() and pb.exists()):
                continue
            r = compare(pa, pb, a.resamples, a.seed)
            out[f"{ta}_vs_{tb}/{part}"] = r
            rows.append(f"| {lb} against {la}, {pname} | {r['n']} | {100 * r['acc_a']:.1f} | {100 * r['acc_b']:.1f} | "
                        f"{100 * r['diff']:+.1f} | [{100 * r['ci95'][0]:+.1f}, {100 * r['ci95'][1]:+.1f}] | "
                        f"{r['only_a']} / {r['only_b']} | {r['mcnemar_p']:.2g} |")
    print("| pair (second minus first) | n | first % | second % | change (points) | 95% interval | first-only / second-only | McNemar p |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    print("\n".join(rows) if rows else "(no pair of generation files found)")
    (d / f"paired_variants{a.suffix}.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
