#!/bin/bash
export G10_DECONTAM_TOOLS=/mnt/Dev/ERA_Submissions/lab-main-wt/teams/09-evaluation/decontam
cd /mnt/Dev/ERA_Submissions/g10-decontam-all-wt/teams/10-post-training/sft
exec /mnt/Dev/ERA_Submissions/.venv-era/bin/python decontam_pipeline.py \
  --sets sft-agentic-nemotron-sft-agentic-v2-tool-calling sft-chat-nemotron-if-v2 sft-code-swe-zero-openhands \
  --work /mnt/Dev/ERA_Submissions/g10-decontam-data/p3 --records /mnt/Dev/ERA_Submissions/g10-decontam-data/run \
  --dec-slots 17 --conv-slots 4 --roll-mb 300 --min-free-gb 40 --max-backlog-gb 25 \
  > /mnt/Dev/ERA_Submissions/g10-decontam-data/p3/driver.out 2>&1
