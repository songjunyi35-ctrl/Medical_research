#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 GPU {control|boundary} {smoke|formal}" >&2
  exit 2
fi
GPU="$1"
ARM_TYPE="$2"
RUN_MODE="$3"
case "$GPU:$ARM_TYPE" in
  0:control) BOUNDARY_WEIGHT="0.0" ;;
  1:boundary) BOUNDARY_WEIGHT="0.1" ;;
  *) echo "preregistered mapping is GPU0=control, GPU1=boundary" >&2; exit 2 ;;
esac
case "$RUN_MODE" in
  smoke|formal) ;;
  *) echo "run mode must be smoke or formal" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/aflow_env.sh"
cd "$REPO_ROOT"
REFINER_CACHE_ROOT="${AFLOW_REFINER_CACHE_ROOT:-$REPO_ROOT/workspace/BraTS/cache/t1ce_wt_refiner_graymean_20260813}"
if [[ "$RUN_MODE" == "formal" ]]; then
  ROOT_OUT="$REPO_ROOT/workspace/BraTS/sammed3d_t1ce_wt/boundary_safe_adapter_pair_20260814"
  EXTRA_ARGS=(
    --epochs 16
    --patience 8
    --expected-baseline-dsc 0.6990433552267926
    --baseline-tolerance 0.00001
  )
else
  ROOT_OUT="$REPO_ROOT/workspace/BraTS/sammed3d_t1ce_wt/boundary_safe_adapter_smoke_20260814"
  EXTRA_ARGS=(
    --epochs 1
    --patience 1
    --limit-train-cases 4
    --limit-validation-cases 4
  )
fi
ARM="safe_adapter_${ARM_TYPE}_seed20260817"
OUTPUT="$ROOT_OUT/$ARM"
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite existing arm: $OUTPUT" >&2
  exit 2
fi
mkdir -p "$ROOT_OUT"

env \
  CUDA_VISIBLE_DEVICES="$GPU" \
  LD_LIBRARY_PATH="/tmp/aflow_nvidia_535_161_08:${LD_LIBRARY_PATH:-}" \
  PYTHONDONTWRITEBYTECODE=1 \
  "$AFLOW_PYTHON_RESOLVED" -u \
  -m scripts.train_t1ce_wt_boundary_safe_adapter \
  --train-cache "$REFINER_CACHE_ROOT/train" \
  --validation-cache "$REFINER_CACHE_ROOT/validation" \
  --init-checkpoint "$REPO_ROOT/workspace/BraTS/sammed3d_t1ce_wt/residual_refiner_robustshift4_three_seed_ensemble_20260813/robustshift4_seed20260815/refiner_val_dice_best.pth" \
  --expected-init-sha256 c5d2b48db2666f097cb0d86e78c20ea95d163c07f98401714d25d755e99aceb0 \
  --output-dir "$OUTPUT" \
  --arm-name "$ARM" \
  --device cuda:0 \
  --seed 20260817 \
  --batch-size 2 \
  --num-workers 4 \
  --base-init-filters 16 \
  --base-dropout 0.1 \
  --adapter-init-filters 8 \
  --adapter-dropout 0.1 \
  --lr 0.0001 \
  --weight-decay 0.00001 \
  --max-shift 12 \
  --coarse-shift-max 4 \
  --coarse-shift-probability 1.0 \
  --boundary-weight "$BOUNDARY_WEIGHT" \
  --boundary-radius 1 \
  "${EXTRA_ARGS[@]}" \
  >"$ROOT_OUT/$ARM.log" 2>&1
