#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aflow_env.sh"
ROOT="$AFLOW_REPO_ROOT"
PY="$AFLOW_PYTHON_RESOLVED"
OUT_REL="workspace/BraTS/sammed3d_t1ce_wt/fv_rank0_matched_aug_train972_val108_e50_s20260725"
OUT="${ROOT}/${OUT_REL}"
LOG="${OUT}.launch.log"
PID_FILE="${OUT}.pid"
PREREG="workspace/BraTS/sammed3d_t1ce_wt/FV_RANK0_MATCHED_AUG_PREREG_20260724.json"

cd "$ROOT"

status() {
  if [[ ! -f "$PID_FILE" ]]; then
    echo "STATUS=not_started"
    return 1
  fi
  local pid
  pid="$(<"$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "STATUS=running PID=${pid}"
    nvidia-smi
    tail -n 20 "$LOG" || true
    return 0
  fi
  echo "STATUS=not_running PID=${pid}"
  tail -n 40 "$LOG" || true
  return 1
}

case "${1:-status}" in
  preflight)
    PYTHONDONTWRITEBYTECODE=1 "$PY" \
      scripts/preflight_sammed3d_t1ce_rank0_matched.py \
      --prereg "$PREREG"
    ;;
  start)
    if [[ -e "$OUT" || -e "$LOG" || -e "$PID_FILE" ]]; then
      echo "Refusing to reuse existing output, log, or PID path." >&2
      exit 2
    fi
    PYTHONDONTWRITEBYTECODE=1 "$PY" \
      scripts/preflight_sammed3d_t1ce_rank0_matched.py \
      --prereg "$PREREG"
    nohup env PYTHONDONTWRITEBYTECODE=1 "$PY" -u \
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
      --amp-dtype bf16 \
      >"$LOG" 2>&1 </dev/null &
    pid=$!
    printf '%s\n' "$pid" >"$PID_FILE"
    sleep 3
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Training exited during launch." >&2
      tail -n 80 "$LOG" >&2 || true
      exit 1
    fi
    echo "STARTED PID=${pid} OUTPUT=${OUT_REL} LOG=${LOG}"
    ;;
  status)
    status
    ;;
  *)
    echo "Usage: $0 {preflight|start|status}" >&2
    exit 2
    ;;
esac
