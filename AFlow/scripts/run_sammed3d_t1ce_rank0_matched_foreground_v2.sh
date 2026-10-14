#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aflow_env.sh"
ROOT="$AFLOW_REPO_ROOT"
PY="$AFLOW_PYTHON_RESOLVED"
OUT_REL="workspace/BraTS/sammed3d_t1ce_wt/fv_rank0_matched_aug_train972_val108_e50_s20260725"
PREREG="workspace/BraTS/sammed3d_t1ce_wt/FV_RANK0_MATCHED_AUG_PREREG_20260724_V2.json"

cd "$ROOT"

if [[ -e "$OUT_REL" ]]; then
  echo "Refusing to reuse existing output path: $OUT_REL" >&2
  exit 2
fi

PYTHONDONTWRITEBYTECODE=1 "$PY" \
  scripts/preflight_sammed3d_t1ce_rank0_matched_v2.py \
  --prereg "$PREREG"

exec env PYTHONDONTWRITEBYTECODE=1 "$PY" -u \
  scripts/train_sammed3d_t1ce_wt.py \
  --train-jsonl \
  data/datasets/brats2024_sammed3d_t1ce_wt_inner90_10_seed20260723_v1/train_t1ce_wt.jsonl \
  --val-jsonl \
  data/datasets/brats2024_sammed3d_t1ce_wt_inner90_10_seed20260723_v1/validation_t1ce_wt.jsonl \
  --checkpoint "$AFLOW_SAMMED3D_CHECKPOINT" \
  --output-dir "$OUT_REL" \
  --strategy decoder_only \
  --encoder-qv-lora-rank 0 \
  --encoder-qv-lora-alpha 8 \
  --spatial-preprocessing whole_resize \
  --train-augmentation affine_elastic_noise \
  --num-clicks 3 \
  --click-method random \
  --epochs 50 \
  --lr 1e-4 \
  --weight-decay 0.1 \
  --batch-size 1 \
  --accumulation-steps 2 \
  --num-workers 2 \
  --seed 20260725 \
  --val-seed 20260723 \
  --selection-metric val_dice \
  --device cuda \
  --amp-dtype bf16
