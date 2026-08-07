# Session 6 — Training Data Execution System

ERA V5 Session 6 assignment: a small but complete **Training Data Execution
System (TDES)** proving the full pipeline —

```
documents → tokenized shards → manifests → mixture schedule → packing → batches
  → training → consumption ledger → learning ledger → checkpoint → crash
  → resume → replay → audit
```

**🌐 Live dashboard: [ravinaik.github.io/ERA/v5/session6/webapp/](https://ravinaik.github.io/ERA/v5/session6/webapp/)**

## Two parts

| Folder | What it is |
|---|---|
| [`train_data_exec_system/`](train_data_exec_system/) | The submission itself — a `uv`-managed Python project. `uv run python run_demo.py` runs the full demonstration and regenerates `submission_artifacts/` (run.log, evidence.json/md, manifests, ledgers, checkpoints, performance.json). See its README for architecture and design decisions. |
| [`webapp/`](webapp/) | A dashboard that visualizes that run. It renders **only** from `webapp/data.js`, which `run_demo.py` generates from the real `submission_artifacts/` bundle on every run — nothing on the page is hand-typed. Open `webapp/index.html` after running the demo. |

## Quick start

```bash
cd train_data_exec_system
uv run python run_demo.py     # regenerates submission_artifacts/ and ../webapp/data.js
uv run pytest tests/ -v       # 20 invariant tests
```

Then open `webapp/index.html` to see that run visualized.

## Grading

All 9 requirements from `evidence.json` (tokenizer integrity, eval firewall,
packing correctness, mixture compliance, OPUS audit trail, crash recovery,
replay, learning trace, throughput) are produced by the implementation, not
asserted — see `train_data_exec_system/submission_artifacts/evidence.md` for
the human-readable table and `train_data_exec_system/README.md` for the full
scoring-coverage breakdown against the assignment's 1,000-point rubric.
