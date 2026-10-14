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
  mixshift4p50_control:0.0|mixshift4p50_graymean_w1:1.0) ;;
  *) echo "arm/weight pair is not preregistered" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/aflow_env.sh"
cd "$REPO_ROOT"
REFINER_CACHE_ROOT="${AFLOW_REFINER_CACHE_ROOT:-$REPO_ROOT/workspace/BraTS/cache/t1ce_wt_refiner_graymean_20260813}"
PAIR_ROOT="$REPO_ROOT/workspace/BraTS/sammed3d_t1ce_wt/residual_refiner_mixshift4p50_graymean_paired_a100_20260813"
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
  -m scripts.train_t1ce_wt_residual_refiner_graymean_robust \
  --train-cache "$REFINER_CACHE_ROOT/train" \
  --validation-cache "$REFINER_CACHE_ROOT/validation" \
  --init-checkpoint "$REPO_ROOT/workspace/BraTS/sammed3d_t1ce_wt/residual_refiner_stage2_gtcrop_h100_train972_val108_e60_s20260810_v2/refiner_val_dice_best.pth" \
  --expected-init-sha256 4513ea4544ae153064575ec2ed5c56e1b78adf5e89e9d094754227b3bd429f21 \
  --expected-base-runner-sha256 b2949927640e2d656cfe2c71a9f53aec491d9edb9dfa071a634ce7b1037133d2 \
  --expected-baseline-dsc 0.6695488602509377 \
  --baseline-tolerance 0.00001 \
  --output-dir "$OUTPUT" \
  --arm-name "$ARM_NAME" \
  --device cuda:0 \
  --seed 20260813 \
  --epochs 30 \
  --patience 10 \
  --batch-size 2 \
  --num-workers 4 \
  --init-filters 16 \
  --dropout 0.1 \
  --lr 0.00005 \
  --weight-decay 0.00001 \
  --max-shift 12 \
  --gray-weight "$GRAY_WEIGHT" \
  --gray-temperature 1.0 \
  --gray-epsilon 0.000001 \
  --coarse-shift-max 4 \
  --coarse-shift-probability 0.5 \
  >"$PAIR_ROOT/$ARM_NAME.log" 2>&1
