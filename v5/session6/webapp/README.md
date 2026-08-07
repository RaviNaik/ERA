# TDES Dashboard

A static, dependency-free dashboard for the Session 6 **Training Data
Execution System**. It visualizes shards, packing policies, curriculum
mixture, OPUS decisions, ledgers, checkpoints, crash/resume/replay proofs,
and the evidence board.

## Data source

Every number on this page comes from `data.js`, which is **generated**, not
hand-written. `../train_data_exec_system/run_demo.py` writes the full
`submission_artifacts/` bundle and then calls
`src/audit/webapp_export.py`, which reads those files back off disk and
emits `data.js` as `window.TDES_DATA = { ... }`. `app.js` only renders what
it finds there — if `data.js` is missing, the page shows a warning banner
instead of guessing.

To refresh the dashboard with a new run:

```bash
cd ../train_data_exec_system
uv run python run_demo.py
```

Then reload `index.html`.

## Viewing it

Files are plain HTML/CSS/JS — open `index.html` directly in a browser, or
serve the folder statically if your browser restricts local
`<script src="data.js">` loads:

```bash
python -m http.server 8000   # from this directory
```

## Files

| File | Role |
|------|------|
| `index.html` | Page structure — hero, shard registry, packing, mixture, OPUS, training, ledgers, checkpoint/replay, evidence board |
| `style.css` | Dark-mode design system (no framework) |
| `app.js` | Reads `window.TDES_DATA` and renders every section; no data is embedded here |
| `data.js` | **Auto-generated** — overwritten on every `run_demo.py` run. Do not hand-edit. |
