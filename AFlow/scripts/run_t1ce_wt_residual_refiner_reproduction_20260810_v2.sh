#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aflow_env.sh"
ROOT="$AFLOW_REPO_ROOT"
PY="$AFLOW_PYTHON_RESOLVED"
CHECKPOINT="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/sammed3d_native_gtcrop_fullunfreeze_ddp2_train972_val108_e100_s20260805/sam_model_val_dice_best.pth"
CHECKPOINT_SHA="42f96a0a2ade9d98d9db897a11a5a285f7a6351a7693c6884d6ac97a29dfde52"
SOURCE_CACHE="$ROOT/workspace/BraTS/cache/t1ce_wt_gtcrop_n4_128"
FULL_CACHE="/tmp/aflow_refiner_cache_v2/t1ce_wt_gtcrop_stage2_42f96_click3_s20260723"
RUN_ROOT="$ROOT/workspace/BraTS/sammed3d_t1ce_wt"
FULL_RUN="$RUN_ROOT/residual_refiner_stage2_gtcrop_h100_train972_val108_e60_s20260810_v2"
LOG="$RUN_ROOT/RESIDUAL_REFINER_REPRODUCTION_20260810_V2.log"

export PYTHONPATH="$ROOT/workspace/BraTS/env_shims"
export MPLCONFIGDIR="/tmp/matplotlib"
export PYTHONDONTWRITEBYTECODE=1

cd "$ROOT"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -u +%FT%TZ)] amended residual-refiner reproduction start"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader
sha256sum \
  workspace/BraTS/sammed3d_t1ce_wt/RESIDUAL_REFINER_PREREG_20260810.json \
  workspace/BraTS/sammed3d_t1ce_wt/RESIDUAL_REFINER_PREREG_AMENDMENT_H100_20260810.json \
  workspace/BraTS/sammed3d_t1ce_wt/H100_RECHECK_GTCROP_3CLICK_20260810.json \
  scripts/cache_sammed3d_t1ce_wt_refiner_inputs.py \
  scripts/train_sammed3d_t1ce_wt_residual_refiner.py \
  "$CHECKPOINT"

mkdir -p "$FULL_CACHE"
"$PY" scripts/cache_sammed3d_t1ce_wt_refiner_inputs.py \
  --cache-dir "$SOURCE_CACHE/train" \
  --split train \
  --checkpoint "$CHECKPOINT" \
  --expected-checkpoint-sha256 "$CHECKPOINT_SHA" \
  --output-dir "$FULL_CACHE/train" \
  --device cuda:0 \
  --seed 20260723

"$PY" scripts/cache_sammed3d_t1ce_wt_refiner_inputs.py \
  --cache-dir "$SOURCE_CACHE/validation" \
  --split validation \
  --checkpoint "$CHECKPOINT" \
  --expected-checkpoint-sha256 "$CHECKPOINT_SHA" \
  --output-dir "$FULL_CACHE/validation" \
  --device cuda:0 \
  --seed 20260723

if [[ -e "$FULL_RUN" ]]; then
  echo "full v2 training output already exists; refusing to overwrite" >&2
  exit 2
fi

"$PY" scripts/train_sammed3d_t1ce_wt_residual_refiner.py \
  --train-cache "$FULL_CACHE/train" \
  --validation-cache "$FULL_CACHE/validation" \
  --output-dir "$FULL_RUN" \
  --device cuda:0 \
  --seed 20260810 \
  --epochs 60 \
  --patience 12 \
  --batch-size 2 \
  --num-workers 4 \
  --init-filters 16 \
  --dropout 0.1 \
  --lr 2e-4 \
  --weight-decay 1e-5 \
  --max-shift 12

echo "[$(date -u +%FT%TZ)] amended full training complete"
