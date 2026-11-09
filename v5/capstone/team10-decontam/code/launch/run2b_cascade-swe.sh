#!/bin/bash
export G10_DECONTAM_TOOLS=/mnt/Dev/ERA_Submissions/lab-main-wt/teams/09-evaluation/decontam
cd /mnt/Dev/ERA_Submissions/g10-decontam-all-wt/teams/10-post-training/sft
exec /mnt/Dev/ERA_Submissions/.venv-era/bin/python decontam_pipeline.py \
  --sets sft-code-nemotron-cascade-swe \
  --work /mnt/Dev/ERA_Submissions/g10-decontam-data/p5c --records /mnt/Dev/ERA_Submissions/g10-decontam-data/run \
  --dec-slots 4 --conv-slots 3 --roll-mb 300 --min-free-gb 40 --max-backlog-gb 10 \
  > /mnt/Dev/ERA_Submissions/g10-decontam-data/p5c.out 2>&1
