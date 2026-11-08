# Team 10 SFT decontamination: results and flagged rows

Every SFT set of the Team 10 mix was checked against Team 9's benchmark fingerprints **v2** (full set, no `--benchmarks`,
`--tags` or `--exclude`, `set_sha256` `2826a3b883210c653979103c09ce1c79e40fc4f7034b54027f2836482370d22a`) and scanned for
canaries, on every row of the whole source. This folder keeps the results. It holds no source data, no training rows and no
text of any document or benchmark item: counts, document ids, benchmark item ids, offsets and sha256 values only.

Status: 24 sets done (5,099,163 rows in, 209,211 flagged and removed, 4.10%, 0 canaries). Math-v2, Proofs and
OpenMathReasoning were running when this was written; OpenThoughts3 and Proofs-v3-SFT are not run yet. See `summary.csv`.
`pending_full.json` gives the reason for each set not run.

| Path | What |
|---|---|
| `summary.csv` | One line per finished set: rows in, flagged, canary, removed, kept, run time, ok |
| `flagged_rows/<set>.csv` | Every flagged row of the set: id, source file, row number, benchmarks hit, number of matched items, highest coverage |
| `records/<set>.record.json.gz` | The full record of the run: per flagged document the matched benchmark items with n-gram counts, coverage and character offsets; sha256 of every input and clean part; the reconciliation checks |
| `manifests/` | What the converter wrote for each set run through the pipeline: rows scanned, rows kept, rows skipped and why, effort tiers, content sha256 |
| `logs/` | `progress.log` of each run (one line per event) |
| `report/decontam_report.md` | The report as in lab PR #40 (saitej123/era-v5-lab) |
| `scripts/export_flagged.py` | Rebuilds `flagged_rows/` and `summary.csv` from the records |
| `scripts/collect.sh` | Copies new records, manifests and logs here after a batch (paths of the working machine) |
| `full_sets.txt`, `pending_full.json` | The finished sets, and the sets not run with the reason |

## Reading a flagged row

The id is `<set>/<n>`. For the sets run through the parallel pipeline, `n = unit * 100,000,000 + row`, where `unit` is the
source file and `row` is the 0-based row number inside it (rows the converter later skipped are counted). `flagged_rows`
already shows the file and the row. For the 15 sets of the first batch the file column reads
`(single source file or files in order)`: their row number is the position in the converted stream of that set, which can
span several source files in order, so it is not necessarily a row of one original shard.

In `records/`, `flagged[<id>]` has `benchmarks`, `n_items` and `hits` (benchmark item id, matching n-grams, coverage,
character offset in the document). `by_benchmark` counts flagged documents per benchmark.

## What is not here

- The clean (kept) data and the converted input: they were deleted part by part to save disk. They can be rebuilt: the
  conversion is deterministic and the records hold the sha256 of every part. The kept rows are the converter output minus
  the flagged ids.
- The source data from the Hub (see the licence table in `teams/10-post-training/licences/` of the lab).
- A check for benchmarks that fingerprints v2 does not cover (HumanEval, MBPP, ARC and some others, see the report, section 8).
