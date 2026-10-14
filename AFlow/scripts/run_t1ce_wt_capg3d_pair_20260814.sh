#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 GPU {plain|capg} {smoke|formal} {train|eval}" >&2
  exit 2
fi
GPU="$1"
ARM="$2"
RUN_MODE="$3"
ACTION="$4"
case "$GPU:$ARM" in
  0:plain|1:capg) ;;
  *) echo "preregistered mapping is GPU0=plain, GPU1=capg" >&2; exit 2 ;;
esac
case "$RUN_MODE:$ACTION" in
  smoke:train|smoke:eval|formal:train|formal:eval) ;;
  *) echo "invalid mode/action" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/aflow_env.sh"
cd "$REPO_ROOT"
CACHE_ROOT="${AFLOW_BRAINCROP_CACHE_ROOT:-/tmp/aflow_braincrop_cache/t1ce_wt_n4_braincrop_128}"
SAM_CHECKPOINT="$REPO_ROOT/workspace/BraTS/sammed3d_t1ce_wt/sammed3d_braincrop_fullunfreeze_ddp2_train972_val108_e200_s20260806/sam_model_val_dice_best.pth"
SAM_SHA256="87dad82c67151128c7e99d64ae87f37c68547c769d2e610a56d432b554e0732a"

if [[ ! -f "$CACHE_ROOT/train/index.json" || ! -f "$CACHE_ROOT/validation/index.json" ]]; then
  echo "brain-crop train/validation cache is missing: $CACHE_ROOT" >&2
  exit 2
fi

if [[ "$RUN_MODE" == "formal" ]]; then
  ROOT_OUT="$REPO_ROOT/workspace/BraTS/sammed3d_t1ce_wt/capg3d_dice01_pair_20260814"
  TRAIN_ARGS=(--epochs 100 --min-epochs 30 --patience 20)
else
  ROOT_OUT="$REPO_ROOT/workspace/BraTS/sammed3d_t1ce_wt/capg3d_dice01_smoke_20260814"
  TRAIN_ARGS=(--epochs 1 --min-epochs 1 --patience 1 --limit-train-cases 4 --limit-validation-cases 4)
fi
ARM_NAME="capg3d_${ARM}_seed20260819"
OUTPUT="$ROOT_OUT/$ARM_NAME"
mkdir -p "$ROOT_OUT"

if [[ "$ACTION" == "train" ]]; then
  if [[ -e "$OUTPUT" ]]; then
    echo "refusing to overwrite existing arm: $OUTPUT" >&2
    exit 2
  fi
  env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    LD_LIBRARY_PATH="/tmp/aflow_nvidia_535_161_08:${LD_LIBRARY_PATH:-}" \
    PYTHONDONTWRITEBYTECODE=1 \
    "$AFLOW_PYTHON_RESOLVED" -u -m scripts.train_t1ce_wt_capg3d \
    --train-cache "$CACHE_ROOT/train" \
    --validation-cache "$CACHE_ROOT/validation" \
    --output-dir "$OUTPUT" \
    --arm "$ARM" \
    --device cuda:0 \
    --seed 20260819 \
    --batch-size 8 \
    --validation-batch-size 8 \
    --num-workers 4 \
    --init-filters 16 \
    --dropout 0.1 \
    --lr 0.0002 \
    --weight-decay 0.00001 \
    --max-shift 8 \
    --center-weight 0.1 \
    --center-sigma 6 \
    --ce-weight 0.1 \
    "${TRAIN_ARGS[@]}"
  exit 0
fi

CHECKPOINT="$OUTPUT/locator_val_dice_best.pth"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "locator checkpoint is missing: $CHECKPOINT" >&2
  exit 2
fi
EVAL_ARGS=()
if [[ "$RUN_MODE" == "smoke" ]]; then
  EVAL_ARGS=(--limit-validation-cases 4)
fi
env \
  CUDA_VISIBLE_DEVICES="$GPU" \
  LD_LIBRARY_PATH="/tmp/aflow_nvidia_535_161_08:${LD_LIBRARY_PATH:-}" \
  PYTHONDONTWRITEBYTECODE=1 \
  "$AFLOW_PYTHON_RESOLVED" -u -m scripts.evaluate_t1ce_wt_capg3d_auto_sam \
  --validation-cache "$CACHE_ROOT/validation" \
  --locator-checkpoint "$CHECKPOINT" \
  --sam-checkpoint "$SAM_CHECKPOINT" \
  --expected-sam-sha256 "$SAM_SHA256" \
  --output "$OUTPUT/AUTO_SAM_VALIDATION.json" \
  --device cuda:0 \
  --num-workers 2 \
  "${EVAL_ARGS[@]}"
