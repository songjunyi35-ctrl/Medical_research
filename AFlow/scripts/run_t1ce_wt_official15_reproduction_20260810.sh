#!/usr/bin/env bash
set -euo pipefail

# Strict development-only reproduction of the public SAM-Med3D input pipeline.
# The scientific task remains T1ce -> WT={1,2,3}; sealed test is never read.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aflow_env.sh"
ROOT="$AFLOW_REPO_ROOT"
BASE_PY="$AFLOW_PYTHON_RESOLVED"
CACHE_ROOT="${AFLOW_OFFICIAL15_CACHE:-/tmp/aflow_t1ce_wt_official15_128}"
RUN_ID="${AFLOW_OFFICIAL15_RUN_ID:-sammed3d_official15_gtcrop_fullunfreeze_b200_train972_val108_e200_s20260810}"
OUTPUT="${AFLOW_OFFICIAL15_OUTPUT:-$ROOT/workspace/BraTS/sammed3d_t1ce_wt/$RUN_ID}"
TRAIN_MANIFEST="data/datasets/brats2024_sammed3d_t1ce_wt_inner90_10_seed20260723_v1/train_t1ce_wt.jsonl"
VAL_MANIFEST="data/datasets/brats2024_sammed3d_t1ce_wt_inner90_10_seed20260723_v1/validation_t1ce_wt.jsonl"

cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1

build_split() {
  local split="$1"
  local manifest="$2"
  local workers="$3"
  if [[ -f "$CACHE_ROOT/$split/index.json" ]]; then
    return
  fi
  "$BASE_PY" -u scripts/build_t1ce_wt_official15_cache.py \
    --manifest "$manifest" \
    --output-dir "$CACHE_ROOT/$split" \
    --split "$split" \
    --workers "$workers"
}

build_split train "$TRAIN_MANIFEST" "${AFLOW_CACHE_TRAIN_WORKERS:-8}"
build_split validation "$VAL_MANIFEST" "${AFLOW_CACHE_VAL_WORKERS:-2}"

case "${1:-}" in
  cache)
    exit 0
    ;;
  smoke)
    shift
    export AFLOW_NPROC=1
    export AFLOW_PYTHON="$BASE_PY"
    exec bash scripts/run_sammed3d_t1ce_wt_dpdsam_aligned.sh smoke \
      --cache-root "$CACHE_ROOT" \
      --spatial-preprocessing gt_crop \
      --train-augmentation random_flip \
      --unfreeze full \
      --batch-size 4 \
      --accumulation-steps 3 \
      --output-dir "$OUTPUT-smoke" \
      --run-identity "$RUN_ID-smoke" \
      "$@"
    ;;
  formal)
    shift
    export AFLOW_NPROC=1
    export AFLOW_PYTHON="$BASE_PY"
    exec bash scripts/run_sammed3d_t1ce_wt_dpdsam_aligned.sh run \
      --cache-root "$CACHE_ROOT" \
      --spatial-preprocessing gt_crop \
      --train-augmentation random_flip \
      --unfreeze full \
      --epochs 200 \
      --batch-size 12 \
      --accumulation-steps 1 \
      --encoder-lr 1e-4 \
      --head-lr 1e-5 \
      --lr-milestones 120 180 \
      --lr-gamma 0.1 \
      --min-epochs-before-stop 201 \
      --patience 201 \
      --output-dir "$OUTPUT" \
      --run-identity "$RUN_ID" \
      "$@"
    ;;
  *)
    echo "Usage: $0 {cache|smoke|formal} [extra trainer args]" >&2
    exit 2
    ;;
esac
