#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aflow_env.sh"
ROOT="$AFLOW_REPO_ROOT"
PY="$AFLOW_PYTHON_RESOLVED"
TRAIN_OUTPUT="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/sammed3d_upstreamliteral11_officialval5_va_a100x2_train972_val108_e200_s2023"
CACHE="/tmp/aflow_t1ce_wt_official15_va_128/test_authorized_20260812"
CHECKPOINT="$TRAIN_OUTPUT/sam_model_val_official_best5.pth"
FREEZE="$TRAIN_OUTPUT/TEST270_FREEZE_20260812.json"
OUTPUT="$TRAIN_OUTPUT/test270_epoch21_official15_upstreamliteral5_s2023_20260812"

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

check_sha() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA mismatch: $path $actual != $expected" >&2
    exit 1
  fi
}

check_sha scripts/build_t1ce_wt_official15_test_cache.py a65d6d38bcf0493161bbcfd3e7732b238ad22e9eb0572155e04f502ffa8b8099
check_sha scripts/evaluate_t1ce_wt_upstream_literal_test.py dc756b8835dba89dd2c684ca6959a58b5a2ccabf834254891d702c934fdb264c
check_sha "$CHECKPOINT" c0c85a0fb90f6301344053f97cabe8a4972bd9febd4c6ae248f3a0908598fe31
check_sha "$FREEZE" e9be1de3f9ba1848d9c5e90667af2dc77dfe4593809e5c3c571a8df4aa90c806

exec "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/evaluate_t1ce_wt_upstream_literal_test.py \
  --cache-dir "$CACHE" \
  --training-output "$TRAIN_OUTPUT" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT" \
  --freeze "$FREEZE" \
  --authorization USER_AUTHORIZED_270_TEST_2026-08-12
