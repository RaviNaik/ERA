#!/bin/bash
export G10_DECONTAM_TOOLS=/mnt/Dev/ERA_Submissions/lab-main-wt/teams/09-evaluation/decontam
export G10_LOCAL_ROOT=/mnt/Dev/ERA_Submissions/g10-hub-downloads
cd /mnt/Dev/ERA_Submissions/g10-pr41-wt/teams/10-post-training/sft
exec /mnt/Dev/ERA_Submissions/.venv-era/bin/python decontam_pipeline.py \
  --sets sft-indic-indic-align-dolly-t sft-indic-indic-align-oasst-t sft-stem-code-openthoughts3 \
  --work /mnt/Dev/ERA_Submissions/g10-decontam-data/p6 --records /mnt/Dev/ERA_Submissions/g10-decontam-data/p6/records \
  --dec-slots 17 --conv-slots 4 --roll-mb 300 --min-free-gb 40 --max-backlog-gb 25 --throttle \
  > /mnt/Dev/ERA_Submissions/g10-decontam-data/p6/driver.out 2>&1
