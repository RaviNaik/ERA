#!/bin/bash
# Copy new records, manifests, logs and the report from the working folders into this folder (run after a batch finishes).
D=/mnt/Dev/ERA_Submissions/g10-decontam-data
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for f in $D/run/*.record.json; do s=$(basename $f .record.json); case " $(cat $HERE/full_sets.txt) " in *" $s "*) gzip -9 -c $f > $HERE/records/$s.record.json.gz;; esac; done
for d in p3 p4 p5 p5c p6 p7 p8 p9 p10; do cp $D/$d/manifests/*.json $HERE/manifests/ 2>/dev/null; done
cp $D/p4/progress.log $HERE/logs/run4-math-v2-proofs-omr.progress.log 2>/dev/null
cp /mnt/Dev/ERA_Submissions/g10-pr41-wt/teams/10-post-training/sft/decontam_report.md $HERE/report/
mkdir -p $HERE/logs/p
python3 $HERE/scripts/export_flagged.py $HERE/records $HERE/logs $HERE
