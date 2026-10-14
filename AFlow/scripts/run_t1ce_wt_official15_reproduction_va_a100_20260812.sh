#!/usr/bin/env bash
set -euo pipefail

# Official 1.5 mm preprocessing reproduction on exactly 2 x A100 80 GB.
# Development train/validation only; this launcher has no sealed-test path.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aflow_env.sh"
ROOT="$AFLOW_REPO_ROOT"
PY="$AFLOW_PYTHON_RESOLVED"
CACHE_ROOT="${AFLOW_OFFICIAL15_CACHE:-/tmp/aflow_t1ce_wt_official15_va_128}"
RUN_ID="${AFLOW_OFFICIAL15_RUN_ID:-sammed3d_official15_va_gtcrop_fullunfreeze_a100x2_train972_val108_e200_trainseed20260805_r20260812}"
OUTPUT="${AFLOW_OFFICIAL15_OUTPUT:-$ROOT/workspace/BraTS/sammed3d_t1ce_wt/$RUN_ID}"
MANIFEST_ROOT="data/datasets/brats2024_t1ce_va_hdfs_20260812_v1"
TRAIN_MANIFEST="$MANIFEST_ROOT/train_t1ce_wt.jsonl"
VAL_MANIFEST="$MANIFEST_ROOT/validation_t1ce_wt.jsonl"
TRAINER="scripts/train_sammed3d_t1ce_wt_official15_va.py"

cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="workspace/BraTS/env_shims${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

# Some worker images expose stale zero-byte 535.129 symlinks.  Prefer the
# kernel-matched 535.161.08 libraries in an isolated, job-local directory.
if [[ -f /usr/lib/x86_64-linux-gnu/libcuda.so.535.161.08 ]]; then
  DRIVER_DIR=/tmp/aflow_nvidia_535_161_08
  mkdir -p "$DRIVER_DIR"
  ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.535.161.08 "$DRIVER_DIR/libcuda.so.1"
  ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.535.161.08 "$DRIVER_DIR/libcuda.so"
  ln -sfn /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.161.08 "$DRIVER_DIR/libnvidia-ml.so.1"
  ln -sfn /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.535.161.08 "$DRIVER_DIR/libnvidia-ptxjitcompiler.so.1"
  export LD_LIBRARY_PATH="$DRIVER_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

build_split() {
  local split="$1"
  local manifest="$2"
  local workers="$3"
  if [[ -f "$CACHE_ROOT/$split/index.json" ]]; then
    return
  fi
  "$PY" -u scripts/build_t1ce_wt_official15_cache.py \
    --manifest "$manifest" \
    --output-dir "$CACHE_ROOT/$split" \
    --split "$split" \
    --workers "$workers"
}

build_cache() {
  build_split train "$TRAIN_MANIFEST" "${AFLOW_CACHE_TRAIN_WORKERS:-24}"
  build_split validation "$VAL_MANIFEST" "${AFLOW_CACHE_VAL_WORKERS:-6}"
}

launch() {
  exec "$PY" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=2 \
    "$TRAINER" "$@"
}

case "${1:-}" in
  cache)
    build_cache
    ;;
  preflight)
    build_cache
    shift
    launch --mode formal --preflight-only \
      --cache-root "$CACHE_ROOT" \
      --spatial-preprocessing gt_crop --train-augmentation random_flip \
      --unfreeze full --epochs 200 --batch-size 6 --accumulation-steps 1 \
      --encoder-lr 1e-4 --head-lr 1e-5 --weight-decay 0.1 \
      --lr-milestones 120 180 --lr-gamma 0.1 \
      --min-epochs-before-stop 201 --patience 201 \
      --run-identity "$RUN_ID" "$@"
    ;;
  smoke)
    build_cache
    shift
    launch --mode smoke \
      --cache-root "$CACHE_ROOT" \
      --spatial-preprocessing gt_crop --train-augmentation random_flip \
      --unfreeze full --batch-size 2 --accumulation-steps 3 \
      --encoder-lr 1e-4 --head-lr 1e-5 --weight-decay 0.1 \
      --lr-milestones 120 180 --lr-gamma 0.1 \
      --output-dir "$OUTPUT-smoke" --run-identity "$RUN_ID-smoke" "$@"
    ;;
  formal)
    build_cache
    shift
    launch --mode formal \
      --cache-root "$CACHE_ROOT" \
      --spatial-preprocessing gt_crop --train-augmentation random_flip \
      --unfreeze full --epochs 200 --batch-size 6 --accumulation-steps 1 \
      --encoder-lr 1e-4 --head-lr 1e-5 --weight-decay 0.1 \
      --lr-milestones 120 180 --lr-gamma 0.1 \
      --min-epochs-before-stop 201 --patience 201 \
      --output-dir "$OUTPUT" --run-identity "$RUN_ID" "$@"
    ;;
  *)
    echo "Usage: $0 {cache|preflight|smoke|formal} [extra trainer args]" >&2
    exit 2
    ;;
esac
