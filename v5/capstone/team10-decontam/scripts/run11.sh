#!/bin/bash
# commitpackft, all 277 language files (1.6 GB) read straight from the Hub
export G10_DECONTAM_TOOLS=/mnt/Dev/ERA_Submissions/lab-main-wt/teams/09-evaluation/decontam
cd /mnt/Dev/ERA_Submissions/g10-followup-wt/teams/10-post-training/sft
exec /mnt/Dev/ERA_Submissions/.venv-era/bin/python decontam_pipeline.py --sets sft-code-commitpackft \
  --work /mnt/Dev/ERA_Submissions/g10-decontam-data/p11 --records /mnt/Dev/ERA_Submissions/g10-decontam-data/p11/records \
  --dec-slots 14 --conv-slots 4 --roll-mb 300 --min-free-gb 40 --max-backlog-gb 25 --throttle --resume
