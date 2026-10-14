#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 GPU SEED {smoke|formal}" >&2
  exit 2
fi
GPU="$1"
SEED="$2"
MODE="$3"
case "$GPU:$SEED" in
  0:20260814|1:20260815) ;;
  *) echo "gpu/seed pair is not preregistered" >&2; exit 2 ;;
esac
case "$MODE" in
  smoke|formal) ;;
  *) echo "mode must be smoke or formal" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/aflow_env.sh"
cd "$REPO_ROOT"
REFINER_CACHE_ROOT="${AFLOW_REFINER_CACHE_ROOT:-$REPO_ROOT/workspace/BraTS/cache/t1ce_wt_refiner_graymean_20260813}"
if [[ "$MODE" == "formal" ]]; then
  ROOT_OUT="$REPO_ROOT/workspace/BraTS/sammed3d_t1ce_wt/error_gated_refiner_a2_two_seed_20260814"
  EXTRA_ARGS=(
    --epochs 16
    --patience 10
    --expected-baseline-dsc 0.6695482504071185
    --baseline-tolerance 0.00001
  )
else
  ROOT_OUT="$REPO_ROOT/workspace/BraTS/sammed3d_t1ce_wt/error_gated_refiner_a2_smoke_20260814"
  EXTRA_ARGS=(
    --epochs 1
    --patience 1
    --limit-train-cases 4
    --limit-validation-cases 4
  )
fi
ARM="elpgr_a2_seed$SEED"
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
  -m scripts.train_t1ce_wt_error_gated_refiner \
  --train-cache "$REFINER_CACHE_ROOT/train" \
  --validation-cache "$REFINER_CACHE_ROOT/validation" \
  --init-checkpoint "$REPO_ROOT/workspace/BraTS/sammed3d_t1ce_wt/residual_refiner_stage2_gtcrop_h100_train972_val108_e60_s20260810_v2/refiner_val_dice_best.pth" \
  --expected-init-sha256 4513ea4544ae153064575ec2ed5c56e1b78adf5e89e9d094754227b3bd429f21 \
  --output-dir "$OUTPUT" \
  --arm-name "$ARM" \
  --device cuda:0 \
  --seed "$SEED" \
  --batch-size 2 \
  --num-workers 4 \
  --init-filters 16 \
  --dropout 0.1 \
  --lr 0.00005 \
  --weight-decay 0.00001 \
  --max-shift 12 \
  --coarse-shift-max 4 \
  --coarse-shift-probability 1.0 \
  --gate-weight 0.1 \
  --gate-dilation-radius 2 \
  "${EXTRA_ARGS[@]}" \
  >"$ROOT_OUT/$ARM.log" 2>&1
