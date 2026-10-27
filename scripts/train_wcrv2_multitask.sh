#!/bin/bash
# Fine-tune WCR v2 (amp-preserved) jointly on EchoNext SHD (12), low LVEF (2) and incident AF (2)
# with age + sex auxiliary inputs. See docs/WCRV2_MULTITASK_RETRAIN_PLAN.md.
#
# Usage:
#   scripts/train_wcrv2_multitask.sh                      # 1 GPU (CUDA_VISIBLE_DEVICES default 0)
#   CUDA_VISIBLE_DEVICES=0,1,2,3 WORLD_SIZE=4 scripts/train_wcrv2_multitask.sh   # DDP on 4 GPUs
#   RUN=ablation_noaux MODEL=ecg_transformer_classifier scripts/train_wcrv2_multitask.sh  # waveform-only ablation
set -euo pipefail

REPO=${REPO:-/volume/DeepECG-SSL-finetune}
PY=${PY:-/volume/venvs/fss39/bin/python}          # python 3.9 venv (omegaconf<2.1, hydra<1.1, torch cu128)
DATA=${DATA:-$REPO/data/wcrv2_multitask}
RUN=${RUN:-wcrv2_multitask_$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$REPO/data/runs/$RUN}
ENCODER=${ENCODER:-/media/data1/models/DeepECG-SSL/wcr-v2/deepecg-ssl-v2-wcr-amp-preserved-encoder.pt}
MODEL=${MODEL:-ecg_transformer_aux_classifier}
CONFIG=${CONFIG:-diagnosis_wcrv2_multitask}   # or diagnosis_wcrv2_echonext12 / diagnosis_wcrv2_afib
WORLD_SIZE=${WORLD_SIZE:-1}
BATCH=${BATCH:-128}
LR=${LR:-1.0e-5}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export WANDB_MODE=${WANDB_MODE:-offline}
export HYDRA_FULL_ERROR=1

mkdir -p "$OUT/checkpoints" "$OUT/logs"
cp "$DATA/labels.json" "$OUT/labels.json" 2>/dev/null || true
LOG=$OUT/logs/train_$(date +%Y%m%d_%H%M%S).log
echo "run=$RUN data=$DATA out=$OUT model=$MODEL config=$CONFIG world_size=$WORLD_SIZE" | tee "$LOG"

exec "$PY" -m fairseq_cli.hydra_train \
  --config-dir "$REPO/examples/w2v_cmsc/config/finetuning/ecg_transformer" \
  --config-name "$CONFIG" \
  task.data="$DATA/manifests" \
  model._name="$MODEL" \
  model.model_path="$ENCODER" \
  checkpoint.save_dir="$OUT/checkpoints" \
  dataset.batch_size="$BATCH" \
  optimization.lr="[$LR]" \
  distributed_training.distributed_world_size="$WORLD_SIZE" \
  "$@" 2>&1 | tee -a "$LOG"
