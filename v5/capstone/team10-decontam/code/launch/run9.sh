#!/bin/bash
# Proofs-v3-SFT: start on each file as soon as the downloader has verified it (--wait-ready), delete it when read (--delete-source)
export G10_DECONTAM_TOOLS=/mnt/Dev/ERA_Submissions/lab-main-wt/teams/09-evaluation/decontam
export G10_LOCAL_ROOT=/mnt/Dev/ERA_Submissions/g10-hub-downloads
cd /mnt/Dev/ERA_Submissions/g10-pr41-wt/teams/10-post-training/sft
exec /mnt/Dev/ERA_Submissions/.venv-era/bin/python decontam_pipeline.py --sets sft-math-nemotron-proofs-v3-sft \
  --work /mnt/Dev/ERA_Submissions/g10-decontam-data/p9 --records /mnt/Dev/ERA_Submissions/g10-decontam-data/p9/records \
  --dec-slots 17 --conv-slots 3 --roll-mb 300 --min-free-gb 40 --max-backlog-gb 25 --throttle --wait-ready --delete-source --resume
