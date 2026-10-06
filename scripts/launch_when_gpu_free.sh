#!/bin/bash
# Wait for an idle GPU (no compute processes, < 2 GB used), then launch the WCR v2 multi-task training on it;
# then wait for a second idle GPU and score the reference models (EchoNext v6, WCR v1 AF-5y) on the test sets.
# Usage: nohup setsid scripts/launch_when_gpu_free.sh > data/wcrv2_multitask/launch_watcher.log 2>&1 &
set -u
REPO=${REPO:-/volume/DeepECG-SSL-finetune}
RUN=${RUN:-wcrv2_mt_v1}
POLL=${POLL:-60}
cd "$REPO"

free_gpu() {  # prints the index of the first idle GPU, or nothing
  local busy
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader | while IFS=, read -r idx uuid mem; do
    idx=$(echo "$idx" | tr -d ' '); uuid=$(echo "$uuid" | tr -d ' '); mem=$(echo "$mem" | tr -dc '0-9')
    if ! grep -q "$uuid" <<<"$busy" && [ "${mem:-99999}" -lt 2000 ]; then echo "$idx"; return; fi
  done
}

echo "$(date +%F_%T) waiting for an idle GPU for training run $RUN"
while true; do
  g=$(free_gpu | head -1)
  if [ -n "$g" ]; then
    echo "$(date +%F_%T) GPU $g is idle: launching training"
    CUDA_VISIBLE_DEVICES=$g RUN=$RUN setsid scripts/train_wcrv2_multitask.sh > data/wcrv2_multitask/train_launch.log 2>&1 &
    echo "$g" > data/wcrv2_multitask/TRAINING_STARTED_ON_GPU
    break
  fi
  sleep "$POLL"
done

sleep 120  # let training claim its memory before looking for a second GPU
echo "$(date +%F_%T) waiting for an idle GPU for the baselines"
while true; do
  g=$(free_gpu | head -1)
  if [ -n "$g" ] && [ "$g" != "$(cat data/wcrv2_multitask/TRAINING_STARTED_ON_GPU)" ]; then
    echo "$(date +%F_%T) GPU $g is idle: scoring baselines"
    PYTHONPATH=$REPO /volume/venvs/fss39/bin/python scripts/eval_wcrv2_multitask.py --baselines \
      --data data/wcrv2_multitask --subsets echonext_test mhi_test --device "$g" --no-ci \
      > data/wcrv2_multitask/baselines.log 2>&1
    echo "$(date +%F_%T) baselines exit $?"
    break
  fi
  sleep "$POLL"
done
