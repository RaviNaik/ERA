"""Inject the numbers from assets/results.json into README.md.

Run after executing the notebook:
    python update_readme.py

Fills the two marker blocks:
    <!-- BEGIN SEVEN_NUMBERS --> ... <!-- END SEVEN_NUMBERS -->
    <!-- BEGIN MTP_RESULTS -->   ... <!-- END MTP_RESULTS -->
"""
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "assets" / "results.json"
README = HERE / "README.md"


def human_bytes(n) -> str:
    n = float(n or 0)
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TiB"


def seven_numbers_table(r: dict) -> str:
    sh = r["shapes"]
    pad = r["padding"]
    bnd = r["boundary"]
    ppl = r["perplexity"]
    tvu = r["tied_vs_untied"]
    mem = r["memory"]
    ratio = f"{mem['ratio']:.1f}x" if mem.get("ratio") else "n/a (CPU)"
    rows = [
        ("1. Tensor shapes",
         f"tokens `{sh['tokens']}`, hidden `{sh['hidden']}`, logits `{sh['logits']}` "
         f"(batch, sequence position, and — for hidden/logits — width `D` / vocab `V`)"),
        ("2. Shift verification",
         f"{r['shift_mismatches']} string mismatches (0 ⇒ `target[i] == input[i+1]`, shift correct)"),
        ("3. Padding mask — contributing tokens",
         f"{pad['contributing_before']} → {pad['contributing_after']} "
         f"(loss {pad['loss_before']:.4f} → {pad['loss_after']:.4f})"),
        ("4. Boundary mask — loss before / after",
         f"{bnd['loss_before']:.4f} → {bnd['loss_after']:.4f} "
         f"(boundary position's own loss {bnd['boundary_loss']:.4f})"),
        ("5. Perplexity vs. vocab size",
         f"ppl ≈ {ppl['ppl_untrained']:,.0f} vs. V = {ppl['vocab_size']:,} "
         f"(loss {ppl['loss_untrained']:.4f} vs. ln V {ppl['ln_v']:.4f}, gap {ppl['relative_gap_pct']:.2f}%)"),
        ("6. Tied vs. untied head params",
         f"{tvu['total_tied']:,} vs. {tvu['total_untied']:,} ({tvu['ratio']:.3f}×)"),
        ("7. Peak memory — ordinary vs. chunked",
         f"{human_bytes(mem['peak_ordinary_bytes'])} vs. {human_bytes(mem['peak_chunked_bytes'])} "
         f"(ratio {ratio}; N={mem['n_tokens']:,} tokens, chunk={mem['chunk_size']})"),
    ]
    out = ["| # | Checklist item | Result |", "|---|---|---|"]
    for i, (item, res) in enumerate(rows, 1):
        out.append(f"| {i} | {item[3:]} | {res} |")
    return "\n".join(out)


def mtp_table(r: dict) -> str:
    m = r["mtp"]
    warm = m.get("gap_mean_after_warmup", m["gap_end"])
    frac = m.get("frac_steps_L2_above_L1")
    frac_txt = f" ({frac * 100:.0f}% of steps have L2 > L1)" if frac is not None else ""
    if warm > 0:
        verdict = (
            f"After the random-init transient, `L2` sits **above** `L1` "
            f"(mean gap {warm:+.4f} nats{frac_txt}) — predicting `t+2` carries one extra "
            f"step of genuine uncertainty, so its entropy floor is higher. Both curves "
            f"fall together because they share one trunk."
        )
    else:
        verdict = (
            f"The mean post-warmup gap is {warm:+.4f} nats{frac_txt} — the two heads track "
            f"very closely at this scale/step count; see notebook §6.1 for discussion."
        )
    return "\n".join([
        f"Two untied, architecturally identical heads trained jointly for {m['steps']} "
        f"steps; the optimized objective is `L1 + L2`.",
        "",
        "| Head | Predicts | Loss (start → end) | Perplexity (end) |",
        "|---|---|---|---|",
        f"| Head 1 | `t+1` | {m['head1_loss_start']:.4f} → {m['head1_loss_end']:.4f} | "
        f"{math.exp(m['head1_loss_end']):,.1f} |",
        f"| Head 2 | `t+2` | {m['head2_loss_start']:.4f} → {m['head2_loss_end']:.4f} | "
        f"{math.exp(m['head2_loss_end']):,.1f} |",
        f"| Sum | — | {m['sum_start']:.4f} → {m['sum_end']:.4f} | — |",
        "",
        f"Gap `L2 − L1`: {m['gap_start']:+.4f} (start) → {m['gap_end']:+.4f} (end) nats. {verdict}",
    ])


def replace_block(text: str, name: str, body: str) -> str:
    pat = re.compile(
        rf"(<!-- BEGIN {name} -->\n).*?(\n<!-- END {name} -->)",
        re.DOTALL,
    )
    if not pat.search(text):
        raise SystemExit(f"marker block {name} not found in README.md")
    return pat.sub(lambda mm: mm.group(1) + body + mm.group(2), text)


def main():
    if not RESULTS.exists():
        raise SystemExit(f"{RESULTS} not found — run the notebook first.")
    r = json.loads(RESULTS.read_text())
    text = README.read_text()
    text = replace_block(text, "SEVEN_NUMBERS", seven_numbers_table(r))
    text = replace_block(text, "MTP_RESULTS", mtp_table(r))
    README.write_text(text)
    print("README.md updated from assets/results.json")


if __name__ == "__main__":
    main()
