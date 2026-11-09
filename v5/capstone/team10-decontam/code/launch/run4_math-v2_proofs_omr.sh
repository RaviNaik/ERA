#!/bin/bash
# batch 4: decontam of the three sets downloaded for the remote server, run here after batch 3 and the downloads finish.
# A set's local source is deleted as soon as its record is written and ok.
D=/mnt/Dev/ERA_Submissions/g10-decontam-data
H=/mnt/Dev/ERA_Submissions/g10-hub-downloads
:  # batch 3 and the downloads are done
export G10_DECONTAM_TOOLS=/mnt/Dev/ERA_Submissions/lab-main-wt/teams/09-evaluation/decontam
export G10_LOCAL_ROOT=$H
touch $D/p4/started   # older sample records of these sets must not trigger the cleanup
cd /mnt/Dev/ERA_Submissions/g10-decontam-all-wt/teams/10-post-training/sft
/mnt/Dev/ERA_Submissions/.venv-era/bin/python decontam_pipeline.py \
  --sets sft-math-nemotron-math-v2 sft-math-nemotron-proofs sft-rl-math-openmathreasoning \
  --work $D/p4 --records $D/run \
  --dec-slots 17 --conv-slots 4 --roll-mb 300 --min-free-gb 15 --max-backlog-gb 20 --throttle > $D/p4/driver.out 2>&1 &
PID=$!
declare -A DIR=( [sft-math-nemotron-math-v2]=nvidia/Nemotron-Math-v2 [sft-math-nemotron-proofs]=nvidia/Nemotron-Math-Proofs-v1 [sft-rl-math-openmathreasoning]=nvidia/OpenMathReasoning )
while kill -0 $PID 2>/dev/null; do
  for s in "${!DIR[@]}"; do
    r=$D/run/$s.record.json
    if [ -f $r ] && [ $r -nt $D/p4/started ] && [ -d $H/${DIR[$s]} ] && python3 -c "import json,sys;sys.exit(0 if json.load(open('$r')).get('ok') and json.load(open('$r')).get('scope')=='full set' else 1)"; then
      rm -rf $H/${DIR[$s]}; echo "$(date '+%F %T') deleted source of $s" >> $D/p4/cleanup.log
    fi
  done
  sleep 60
done
