#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aflow_env.sh"
ROOT="$AFLOW_REPO_ROOT"
PY="$AFLOW_PYTHON_RESOLVED"
CACHE_ROOT="${AFLOW_OFFICIAL15_CACHE:-/tmp/aflow_t1ce_wt_official15_va_128}"
RUN_ID="${AFLOW_UPSTREAM_RUN_ID:-sammed3d_upstreamliteral11_officialval5_va_a100x2_train972_val108_e200_s2023}"
OUTPUT="${AFLOW_UPSTREAM_OUTPUT:-$ROOT/workspace/BraTS/sammed3d_t1ce_wt/$RUN_ID}"

cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="workspace/BraTS/env_shims${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

if [[ -f /usr/lib/x86_64-linux-gnu/libcuda.so.535.161.08 ]]; then
  DRIVER_DIR=/tmp/aflow_nvidia_535_161_08
  mkdir -p "$DRIVER_DIR"
  ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.535.161.08 "$DRIVER_DIR/libcuda.so.1"
  ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.535.161.08 "$DRIVER_DIR/libcuda.so"
  ln -sfn /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.161.08 "$DRIVER_DIR/libnvidia-ml.so.1"
  ln -sfn /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.535.161.08 "$DRIVER_DIR/libnvidia-ptxjitcompiler.so.1"
  export LD_LIBRARY_PATH="$DRIVER_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

launch() {
  exec "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_sammed3d_t1ce_wt_upstream_literal_va.py "$@"
}

case "${1:-}" in
  preflight)
    shift
    launch --mode formal --preflight-only --cache-root "$CACHE_ROOT" \
      --epochs 200 --batch-size 12 --accumulation-steps 20 --num-workers 12 \
      --encoder-lr 8e-4 --head-lr 8e-5 --weight-decay 0.1 \
      --run-identity "$RUN_ID" "$@"
    ;;
  smoke)
    shift
    launch --mode smoke --cache-root "$CACHE_ROOT" \
      --epochs 200 --batch-size 12 --accumulation-steps 20 --num-workers 2 \
      --encoder-lr 8e-4 --head-lr 8e-5 --weight-decay 0.1 \
      --output-dir "$OUTPUT-smoke" --run-identity "$RUN_ID-smoke" "$@"
    ;;
  formal)
    shift
    launch --mode formal --cache-root "$CACHE_ROOT" \
      --epochs 200 --batch-size 12 --accumulation-steps 20 --num-workers 12 \
      --encoder-lr 8e-4 --head-lr 8e-5 --weight-decay 0.1 \
      --output-dir "$OUTPUT" --run-identity "$RUN_ID" "$@"
    ;;
  *)
    echo "Usage: $0 {preflight|smoke|formal} [extra args]" >&2
    exit 2
    ;;
esac
