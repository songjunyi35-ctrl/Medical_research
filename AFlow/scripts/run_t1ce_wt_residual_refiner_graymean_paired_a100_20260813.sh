#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/aflow_env.sh"
cd "$REPO_ROOT"

PYTHON="$AFLOW_PYTHON_RESOLVED"
DRIVER_LIB="/tmp/aflow_nvidia_535_161_08"
REFINER_CACHE_ROOT="${AFLOW_REFINER_CACHE_ROOT:-$REPO_ROOT/workspace/BraTS/cache/t1ce_wt_refiner_graymean_20260813}"
TRAIN_CACHE="$REFINER_CACHE_ROOT/train"
VALIDATION_CACHE="$REFINER_CACHE_ROOT/validation"
INIT_CHECKPOINT="$REPO_ROOT/workspace/BraTS/sammed3d_t1ce_wt/residual_refiner_stage2_gtcrop_h100_train972_val108_e60_s20260810_v2/refiner_val_dice_best.pth"
PAIR_ROOT="$REPO_ROOT/workspace/BraTS/sammed3d_t1ce_wt/residual_refiner_graymean_paired_a100_20260813"
CONTROL_OUTPUT="$PAIR_ROOT/control_continuation"
GRAY_OUTPUT="$PAIR_ROOT/graymean_w1"

if [[ -e "$PAIR_ROOT" ]]; then
  echo "Refusing to overwrite existing paired run: $PAIR_ROOT" >&2
  exit 2
fi
mkdir -p "$PAIR_ROOT"

COMMON_ARGS=(
  --train-cache "$TRAIN_CACHE"
  --validation-cache "$VALIDATION_CACHE"
  --init-checkpoint "$INIT_CHECKPOINT"
  --expected-init-sha256 4513ea4544ae153064575ec2ed5c56e1b78adf5e89e9d094754227b3bd429f21
  --expected-baseline-dsc 0.6695488602509377
  --baseline-tolerance 0.0000001
  --device cuda:0
  --seed 20260813
  --epochs 30
  --patience 10
  --batch-size 2
  --num-workers 4
  --init-filters 16
  --dropout 0.1
  --lr 0.00005
  --weight-decay 0.00001
  --max-shift 12
  --gray-temperature 1.0
  --gray-epsilon 0.000001
)

pids=()
terminate_children() {
  if (( ${#pids[@]} > 0 )); then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap terminate_children INT TERM EXIT

env \
  CUDA_VISIBLE_DEVICES=0 \
  LD_LIBRARY_PATH="$DRIVER_LIB:${LD_LIBRARY_PATH:-}" \
  PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON" -u -m scripts.train_t1ce_wt_residual_refiner_graymean \
  "${COMMON_ARGS[@]}" \
  --arm-name control_continuation \
  --gray-weight 0.0 \
  --output-dir "$CONTROL_OUTPUT" \
  >"$PAIR_ROOT/control_continuation.log" 2>&1 &
pids+=("$!")

env \
  CUDA_VISIBLE_DEVICES=1 \
  LD_LIBRARY_PATH="$DRIVER_LIB:${LD_LIBRARY_PATH:-}" \
  PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON" -u -m scripts.train_t1ce_wt_residual_refiner_graymean \
  "${COMMON_ARGS[@]}" \
  --arm-name graymean_w1 \
  --gray-weight 1.0 \
  --output-dir "$GRAY_OUTPUT" \
  >"$PAIR_ROOT/graymean_w1.log" 2>&1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
trap - INT TERM EXIT
exit "$status"
