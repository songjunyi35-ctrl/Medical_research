#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 GPU ARM_NAME GRAY_WEIGHT" >&2
  exit 2
fi
GPU="$1"
ARM_NAME="$2"
GRAY_WEIGHT="$3"
case "$ARM_NAME:$GRAY_WEIGHT" in
  robust_to_clean_control:0.0|robust_to_clean_graymean_w1:1.0) ;;
  *) echo "arm/weight pair is not preregistered" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/aflow_env.sh"
cd "$REPO_ROOT"
REFINER_CACHE_ROOT="${AFLOW_REFINER_CACHE_ROOT:-$REPO_ROOT/workspace/BraTS/cache/t1ce_wt_refiner_graymean_20260813}"
PAIR_ROOT="$REPO_ROOT/workspace/BraTS/sammed3d_t1ce_wt/residual_refiner_robust_to_clean_graymean_paired_a100_20260813"
OUTPUT="$PAIR_ROOT/$ARM_NAME"
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite existing arm: $OUTPUT" >&2
  exit 2
fi
mkdir -p "$PAIR_ROOT"

env \
  CUDA_VISIBLE_DEVICES="$GPU" \
  LD_LIBRARY_PATH="/tmp/aflow_nvidia_535_161_08:${LD_LIBRARY_PATH:-}" \
  PYTHONDONTWRITEBYTECODE=1 \
  "$AFLOW_PYTHON_RESOLVED" -u \
  -m scripts.train_t1ce_wt_residual_refiner_graymean \
  --train-cache "$REFINER_CACHE_ROOT/train" \
  --validation-cache "$REFINER_CACHE_ROOT/validation" \
  --init-checkpoint "$REPO_ROOT/workspace/BraTS/sammed3d_t1ce_wt/residual_refiner_robustshift4_graymean_paired_a100_20260813/robustshift4_control/refiner_val_dice_best_bridge_for_clean_phase.pth" \
  --expected-init-sha256 cf47f3fe92735f788a78a37cda2bd9d4d2f4d0e8b594f6036596e455316e4b4c \
  --expected-baseline-dsc 0.695636292108921 \
  --baseline-tolerance 0.00001 \
  --output-dir "$OUTPUT" \
  --arm-name "$ARM_NAME" \
  --device cuda:0 \
  --seed 20260813 \
  --epochs 20 \
  --patience 7 \
  --batch-size 2 \
  --num-workers 4 \
  --init-filters 16 \
  --dropout 0.1 \
  --lr 0.00001 \
  --weight-decay 0.00001 \
  --max-shift 12 \
  --gray-weight "$GRAY_WEIGHT" \
  --gray-temperature 1.0 \
  --gray-epsilon 0.000001 \
  >"$PAIR_ROOT/$ARM_NAME.log" 2>&1
