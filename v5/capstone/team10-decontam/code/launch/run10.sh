#!/bin/bash
# commitpackft (python subset, 136 MB) read straight from the Hub file
export G10_DECONTAM_TOOLS=/mnt/Dev/ERA_Submissions/lab-main-wt/teams/09-evaluation/decontam
cd /mnt/Dev/ERA_Submissions/g10-pr41-wt/teams/10-post-training/sft
exec /mnt/Dev/ERA_Submissions/.venv-era/bin/python decontam_pipeline.py --sets sft-code-commitpackft \
  --work /mnt/Dev/ERA_Submissions/g10-decontam-data/p10 --records /mnt/Dev/ERA_Submissions/g10-decontam-data/p10/records \
  --dec-slots 4 --conv-slots 1 --roll-mb 300 --min-free-gb 40 --max-backlog-gb 25 --throttle
