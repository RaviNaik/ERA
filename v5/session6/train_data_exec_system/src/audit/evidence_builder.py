"""
Evidence Builder - generates the complete submission evidence bundle.
Produces: run.log, evidence.json, evidence.md, performance.json
All evidence is derived from actual run artifacts (not hardcoded) — including
the Mermaid diagrams and bar/sparkline visuals in evidence.md, which are
rendered from the same numbers written to evidence.json / the ledgers, not
from a separate hand-maintained copy.
"""
from __future__ import annotations
import json, time, hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ── Small rendering helpers (pure functions, no state) ─────────────────────

_BLOCKS = " ▏▎▍▌▋▊▉█"  # 1/8-width steps for a smooth unicode bar


def _bar(fraction: float, width: int = 20) -> str:
    """Render `fraction` (0..1, values outside are clamped) as a unicode bar."""
    fraction = max(0.0, min(1.0, fraction))
    filled_eighths = round(fraction * width * 8)
    full_cells, remainder = divmod(filled_eighths, 8)
    bar = "█" * full_cells
    if full_cells < width:
        bar += _BLOCKS[remainder]
        bar += " " * (width - full_cells - 1)
    return bar


_SPARK = "▁▂▃▄▅▆▇█"


def _sparkline(values: Sequence[float]) -> str:
    """Render a short numeric series as a unicode sparkline."""
    values = list(values)
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo
    if span <= 0:
        return _SPARK[0] * len(values)
    return "".join(_SPARK[min(7, int((v - lo) / span * 7))] for v in values)


def _mermaid_pie(title: str, parts: Dict[str, int]) -> List[str]:
    lines = ["```mermaid", f"pie title {title}"]
    for label, value in parts.items():
        if value > 0:
            lines.append(f'    "{label}" : {value}')
    lines.append("```")
    return lines


def _mermaid_checkpoint_gitgraph(main_checkpoints: List[dict], fork_checkpoints: List[dict],
                                  crash_at_step: Optional[int], resume_from_step: Optional[int],
                                  fork_source_checkpoint_id: Optional[str] = None) -> List[str]:
    """
    Build a gitGraph diagram from the *actual* checkpoints this run produced —
    not a fixed illustration. main_checkpoints/fork_checkpoints come straight
    off CheckpointManager.list_checkpoints(), split by branch_id.
    """
    lines = ["```mermaid", "gitGraph"]
    lines.append('    commit id: "run_start"')

    fork_point_ckpt_id = fork_source_checkpoint_id
    branched = False

    for ckpt in main_checkpoints:
        label = f"step {ckpt['step']:02d}"
        lines.append(f'    commit id: "{ckpt["checkpoint_id"]}" tag: "{label}"')
        if (not branched and fork_point_ckpt_id and
                ckpt["checkpoint_id"] == fork_point_ckpt_id):
            lines.append("    branch experiment_v2")
            lines.append("    checkout main")
            branched = True
        if crash_at_step is not None and ckpt["step"] == resume_from_step:
            lines.append(f'    commit id: "crash_at_step_{crash_at_step}" type: REVERSE')
            lines.append(f'    commit id: "resume_from_step_{resume_from_step}" type: HIGHLIGHT')

    if fork_checkpoints:
        lines.append("    checkout experiment_v2")
        for ckpt in fork_checkpoints:
            lines.append(f'    commit id: "{ckpt["checkpoint_id"]}" tag: "forked"')

    lines.append("```")
    return lines


class EvidenceBuilder:
    """Collects evidence from run artifacts and writes the evidence bundle."""

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._log_lines: List[str] = []
        self._evidence: Dict[str, Any] = {}

    def log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self._log_lines.append(line)
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode('ascii', errors='replace').decode('ascii'))

    def record_pass(self, key: str, description: str, evidence_path: str) -> None:
        self._evidence[key] = {"result": "PASS", "description": description,
                                "evidence": evidence_path}
        self.log(f"[PASS] {key}: {description}")

    def record_fail(self, key: str, description: str, reason: str) -> None:
        self._evidence[key] = {"result": "FAIL", "description": description,
                                "reason": reason}
        self.log(f"[FAIL] {key}: {description} — {reason}")

    def add_section(self, title: str) -> None:
        self.log(f"\n{'='*60}")
        self.log(f"  {title}")
        self.log(f"{'='*60}")

    def write_run_log(self) -> Path:
        log_path = self.artifacts_dir / "run.log"
        log_path.write_text("\n".join(self._log_lines), encoding="utf-8")
        return log_path

    def write_evidence_json(self) -> Path:
        evidence_path = self.artifacts_dir / "evidence.json"
        evidence_path.write_text(json.dumps(self._evidence, indent=2), encoding="utf-8")
        return evidence_path

    # ── evidence.md ──────────────────────────────────────────────────────

    _LABELS = {
        "tokenizer_integrity": "Tokenizer integrity",
        "eval_firewall": "Evaluation firewall",
        "packing_correctness": "Packing correctness",
        "mixture_compliance": "Mixture compliance",
        "opus_audit_trail": "OPUS audit trail",
        "crash_recovery": "Crash recovery",
        "replay": "Replay hash match",
        "learning_trace": "Learning trace",
        "throughput": "Throughput & utilization",
    }

    def write_evidence_md(self, context: Optional[Dict[str, Any]] = None) -> Path:
        """
        Write the human-readable evidence bundle. `context` (optional, see
        run_demo.py's call site for the exact shape) supplies the real
        numbers behind the extra tables/diagrams below the core requirement
        table — packing stats, OPUS decisions, mixture actuals, the real
        checkpoint list, replay results, learning-trace stats. Every value
        rendered here is read from that context or from self._evidence /
        self._log_lines, never hand-typed.
        """
        ctx = context or {}
        passed = sum(1 for v in self._evidence.values() if v.get("result") == "PASS")
        total = len(self._evidence)
        overall_badge = "✅ ALL PASS" if total and passed == total else f"⚠️ {passed}/{total} PASS"

        lines = [
            "# Training Data Execution System — Evidence Bundle",
            "",
            f"**Generated:** {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} · "
            f"**Result:** {overall_badge} · **Run:** `{ctx.get('run_id', '—')}`",
            "",
            "> Every number, table and diagram on this page is generated by "
            "`run_demo.py` from the real `submission_artifacts/` this run "
            "produced — nothing here is hand-typed. Re-run "
            "`uv run python run_demo.py` and this file is rewritten from "
            "scratch.",
            "",
        ]

        # ── Pipeline at a glance ────────────────────────────────────────
        lines += self._pipeline_diagram(ctx)

        # ── Requirement results ────────────────────────────────────────
        lines += [
            "## Requirement Results",
            "",
            "| Requirement | Result | Evidence |",
            "|-------------|--------|----------|",
        ]
        for key, label in self._LABELS.items():
            e = self._evidence.get(key, {})
            result = e.get("result", "UNKNOWN")
            badge = "✅ PASS" if result == "PASS" else "❌ FAIL"
            evidence = e.get("evidence", e.get("reason", "—"))
            lines.append(f"| {label} | {badge} | `{evidence}` |")
        lines.append("")

        if ctx:
            lines += self._shards_section(ctx)
            lines += self._packing_section(ctx)
            lines += self._opus_section(ctx)
            lines += self._mixture_section(ctx)
            lines += self._checkpoint_section(ctx)
            lines += self._learning_section(ctx)
            lines += self._performance_section(ctx)

        lines += self._log_excerpt()

        md_path = self.artifacts_dir / "evidence.md"
        md_path.write_text("\n".join(lines), encoding="utf-8")
        return md_path

    def _pipeline_diagram(self, ctx: dict) -> List[str]:
        def ok(key: str) -> bool:
            return self._evidence.get(key, {}).get("result") == "PASS"

        style = lambda n, good: f"style {n} fill:{'#1a7f37' if good else '#8b1a1a'},color:#fff"
        lines = [
            "## Pipeline At A Glance",
            "",
            "```mermaid",
            "flowchart LR",
            "    A[Documents] --> B[Tokenized Shards]",
            "    B --> C[Manifests]",
            "    C --> D{Eval Firewall}",
            "    D -->|admitted| E[Packing]",
            "    D -->|held out| X[Blocked — never trained]",
            "    E --> F[Mixture + OPUS]",
            "    F --> G[Training]",
            "    G --> H[Consumption + Learning Ledgers]",
            "    H --> I[Checkpoint]",
            "    I --> J((Crash))",
            "    J --> K[Resume]",
            "    K --> L[Replay]",
            "    L --> M[Fork]",
            "    M --> N[Evidence Bundle]",
            "",
            f"    {style('C', ok('tokenizer_integrity'))}",
            f"    {style('D', ok('eval_firewall'))}",
            f"    {style('E', ok('packing_correctness'))}",
            f"    {style('F', ok('mixture_compliance') and ok('opus_audit_trail'))}",
            f"    {style('H', ok('learning_trace'))}",
            f"    {style('K', ok('crash_recovery'))}",
            f"    {style('L', ok('replay'))}",
            "```",
            "",
        ]
        return lines

    def _shards_section(self, ctx: dict) -> List[str]:
        ms = ctx.get("manifest_summary", {})
        total, admitted, blocked = ms.get("total", 0), ms.get("admitted", 0), ms.get("blocked", 0)
        lines = [
            "## Shards & Tokenizer",
            "",
            f"- Tokenizer: `{ctx.get('tokenizer_model', '—')}` · "
            f"SHA `{ctx.get('tokenizer_sha', '—')}` · "
            f"vocab {ctx.get('vocab_size', 0):,}",
            f"- Shards: **{total}** total → **{admitted}** admitted "
            f"`{_bar(admitted / max(1, total), 24)}` **{blocked}** blocked (held-out eval)",
            "",
        ]
        return lines

    def _packing_section(self, ctx: dict) -> List[str]:
        results: Dict[str, dict] = ctx.get("packing_results", {})
        if not results:
            return []
        prod = ctx.get("production_policy", "structure_preserving")
        lines = [
            "## Packing Policy Comparison",
            "",
            "All 5 policies packed the *same* admitted corpus this run — this is "
            "a real measurement, not the Session 6 widget's reference numbers.",
            "",
            "| Policy | Utilization | | Sequences |",
            "|--------|------------:|---|----------:|",
        ]
        for policy, stats in results.items():
            util = stats.get("utilization_pct", 0) / 100.0
            marker = " **← production**" if policy == prod else ""
            lines.append(
                f"| `{policy}`{marker} | {stats.get('utilization_pct', 0):.1f}% "
                f"| `{_bar(util, 20)}` | {stats.get('num_sequences', 0)} |"
            )
        lines.append("")
        return lines

    def _opus_section(self, ctx: dict) -> List[str]:
        summary = ctx.get("opus_summary", {})
        if not summary or summary.get("total_candidates", 0) == 0:
            return []
        lines = ["## Curriculum & OPUS Selection", ""]
        stages = ctx.get("curriculum_stages", [])
        if stages:
            lines.append("| Stage | Budget | Top Lanes |")
            lines.append("|-------|-------:|-----------|")
            for s in stages:
                top = sorted(s["lane_weights"].items(), key=lambda kv: -kv[1])[:3]
                top_str = ", ".join(f"{lane} {w:.0%}" for lane, w in top)
                lines.append(f"| {s['name']} | {s['token_budget_fraction']:.0%} | {top_str} |")
            lines.append("")
        lines += _mermaid_pie(
            f"OPUS Decisions ({summary.get('total_candidates', 0)} candidates, "
            f"{summary.get('floor_overrides', 0)} floor overrides)",
            {
                "Accepted": summary.get("accepted", 0),
                "Rejected": summary.get("rejected", 0),
                "Deferred": summary.get("deferred", 0),
            },
        )
        lines.append("")
        lines.append(
            f"Keep-fraction target **{summary.get('keep_fraction_target', 0):.0%}**, "
            f"actual **{summary.get('keep_fraction_actual', 0):.0%}** "
            f"(effective token multiplier ×{summary.get('effective_token_multiplier', 0):.2f}). "
            "Curriculum stage weights feed directly into the OPUS proxy score "
            "(see `src/mixture/opus_selector.py:_proxy_score`), so a lane's "
            "weight *this stage* has a real, observable effect on which "
            "batches get accepted."
        )
        lines.append("")
        return lines

    def _mixture_section(self, ctx: dict) -> List[str]:
        actual: Dict[str, float] = ctx.get("mixture_actual", {})
        floors: Dict[str, dict] = ctx.get("floor_compliance", {})
        if not actual:
            return []
        lines = [
            "## Mixture — Planned vs. Actual",
            "",
            "| Lane | Actual Share | |",
            "|------|-------------:|---|",
        ]
        for lane, share in sorted(actual.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {lane} | {share:.1%} | `{_bar(share, 20)}` |")
        lines.append("")
        if floors:
            lines.append("**Protected floors** (hard minimums, enforced by OPUS floor override):")
            lines.append("")
            lines.append("| Lane | Floor | Actual | Status |")
            lines.append("|------|------:|-------:|:------:|")
            for lane, check in floors.items():
                status = "✅ compliant" if check["compliant"] else "❌ violated"
                lines.append(f"| {lane} | {check['floor']:.0%} | {check['actual']:.1%} | {status} |")
            lines.append("")
        return lines

    def _checkpoint_section(self, ctx: dict) -> List[str]:
        all_ckpts: List[dict] = ctx.get("checkpoints", [])
        if not all_ckpts:
            return []
        main_ckpts = [c for c in all_ckpts if c.get("branch_id") == "main"]
        fork_ckpts = [c for c in all_ckpts if c.get("branch_id") != "main"]
        lines = [
            "## Checkpoint · Crash · Resume · Replay · Fork",
            "",
            f"Checkpoint saved every **{ctx.get('checkpoint_every', '—')}** dataloader "
            f"positions, independent of that step's OPUS decision. Crash simulated at "
            f"step **{ctx.get('crash_at_step', '—')}**; resume loaded the checkpoint at "
            f"step **{ctx.get('resume_from_step', '—')}**.",
            "",
        ]
        lines += _mermaid_checkpoint_gitgraph(
            main_ckpts, fork_ckpts, ctx.get("crash_at_step"), ctx.get("resume_from_step"),
            ctx.get("fork_source_checkpoint_id"))
        lines.append("")

        replay = ctx.get("replay_results", {})
        steps = replay.get("steps_replayed", [])
        if steps:
            all_match = replay.get("all_match", False)
            lines.append(
                f"**Replay** of steps `{steps[0]['step']}..{steps[-1]['step']}`: "
                f"{'✅ all batch hashes matched byte-for-byte' if all_match else '❌ mismatch detected'}."
            )
            lines.append("")
        return lines

    def _learning_section(self, ctx: dict) -> List[str]:
        doc_losses: Dict[str, float] = ctx.get("doc_avg_loss", {})
        loss_series: List[float] = ctx.get("loss_series", [])
        if not doc_losses:
            return []
        values = list(doc_losses.values())
        lines = [
            "## Learning Trace",
            "",
            f"Per-document average loss recorded for **{len(doc_losses)}** source "
            f"documents (min `{min(values):.4f}`, mean `{sum(values)/len(values):.4f}`, "
            f"max `{max(values):.4f}`).",
            "",
        ]
        if loss_series:
            lines.append(f"Loss per accepted step: `{_sparkline(loss_series)}` "
                          f"({loss_series[0]:.4f} → {loss_series[-1]:.4f})")
            lines.append("")
        return lines

    def _performance_section(self, ctx: dict) -> List[str]:
        perf = ctx.get("performance", {})
        if not perf:
            return []
        lines = [
            "## Performance",
            "",
            "| Metric | Value |",
            "|--------|------:|",
            f"| Average loss | {perf.get('average_loss', 0):.4f} |",
            f"| Average tokens/sec | {perf.get('average_tokens_per_sec', 0):.1f} |",
            f"| Packing utilization | {perf.get('packing_utilization_pct', 0):.1f}% |",
            f"| Useful tokens | {perf.get('useful_tokens', 0):,} |",
            "",
        ]
        return lines

    def _log_excerpt(self) -> List[str]:
        """
        Trim self._log_lines down to the events that matter: [PASS]/[FAIL]
        markers, checkpoint/crash/resume/replay/fork/stage events, and a
        compact section separator in place of each add_section() call's
        3-line rule/title/rule triplet (which would otherwise print as two
        bare "====" rules with nothing readable between them).
        """
        lines = ["## Log Excerpt", "", "```"]
        src = self._log_lines
        i, n = 0, len(src)
        while i < n:
            line = src[i]
            is_rule = "===" in line
            if is_rule and i + 2 < n and "===" not in src[i + 1] and "===" in src[i + 2]:
                title = src[i + 1].split("] ", 1)[-1].strip()
                lines.append(f"\n-- {title} --")
                i += 3
                continue
            if ("[PASS]" in line or "[FAIL]" in line or "[CHECKPOINT]" in line
                    or "[CRASH]" in line or "[RESUME]" in line or "[REPLAY]" in line
                    or "[FORK]" in line or "[STAGE]" in line):
                lines.append(line)
            i += 1
        lines.append("```")
        return lines

    # ── performance.json ─────────────────────────────────────────────────

    def write_performance_json(self, learning_entries: List[dict],
                                packing_stats: dict) -> Path:
        if not learning_entries:
            avg_loss = 0.0
            avg_tps = 0.0
        else:
            avg_loss = sum(e["loss"] for e in learning_entries) / len(learning_entries)
            avg_tps = sum(e["tokens_per_sec"] for e in learning_entries) / len(learning_entries)

        perf = {
            "total_steps": len(learning_entries),
            "average_loss": round(avg_loss, 6),
            "final_loss": round(learning_entries[-1]["loss"], 6) if learning_entries else 0.0,
            "average_tokens_per_sec": round(avg_tps, 1),
            "packing_policy": packing_stats.get("policy", ""),
            "packing_utilization_pct": packing_stats.get("utilization_pct", 0),
            "useful_tokens_total": packing_stats.get("useful_tokens", 0),
            "num_packed_sequences": packing_stats.get("num_sequences", 0),
            "context_len": packing_stats.get("context_len", 0),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        perf_path = self.artifacts_dir / "performance.json"
        perf_path.write_text(json.dumps(perf, indent=2))
        return perf_path
