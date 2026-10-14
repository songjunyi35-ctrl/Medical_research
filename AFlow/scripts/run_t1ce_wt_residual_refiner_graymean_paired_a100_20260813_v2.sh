#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/aflow_env.sh"
cd "$REPO_ROOT"
REFINER_CACHE_ROOT="${AFLOW_REFINER_CACHE_ROOT:-$REPO_ROOT/workspace/BraTS/cache/t1ce_wt_refiner_graymean_20260813}"
PAIR_ROOT="$REPO_ROOT/workspace/BraTS/sammed3d_t1ce_wt/residual_refiner_graymean_paired_a100_20260813"
if [[ -e "$PAIR_ROOT" ]]; then
  echo "Refusing to overwrite existing paired run: $PAIR_ROOT" >&2
  exit 2
fi
mkdir -p "$PAIR_ROOT"

run_arm() {
  local gpu="$1"
  local arm="$2"
  local gray_weight="$3"
  local output="$4"
  env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    LD_LIBRARY_PATH="/tmp/aflow_nvidia_535_161_08:${LD_LIBRARY_PATH:-}" \
    PYTHONDONTWRITEBYTECODE=1 \
    "$AFLOW_PYTHON_RESOLVED" -u \
    -m scripts.train_t1ce_wt_residual_refiner_graymean \
    --train-cache "$REFINER_CACHE_ROOT/train" \
    --validation-cache "$REFINER_CACHE_ROOT/validation" \
    --init-checkpoint "$REPO_ROOT/workspace/BraTS/sammed3d_t1ce_wt/residual_refiner_stage2_gtcrop_h100_train972_val108_e60_s20260810_v2/refiner_val_dice_best.pth" \
    --expected-init-sha256 4513ea4544ae153064575ec2ed5c56e1b78adf5e89e9d094754227b3bd429f21 \
    --expected-baseline-dsc 0.6695488602509377 \
    --baseline-tolerance 0.00001 \
    --output-dir "$output" \
    --arm-name "$arm" \
    --device cuda:0 \
    --seed 20260813 \
    --epochs 30 \
    --patience 10 \
    --batch-size 2 \
    --num-workers 4 \
    --init-filters 16 \
    --dropout 0.1 \
    --lr 0.00005 \
    --weight-decay 0.00001 \
    --max-shift 12 \
    --gray-weight "$gray_weight" \
    --gray-temperature 1.0 \
    --gray-epsilon 0.000001
}

pids=()
terminate_children() {
  if (( ${#pids[@]} > 0 )); then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap terminate_children INT TERM EXIT

run_arm 0 control_continuation 0.0 "$PAIR_ROOT/control_continuation" \
  >"$PAIR_ROOT/control_continuation.log" 2>&1 &
pids+=("$!")
run_arm 1 graymean_w1 1.0 "$PAIR_ROOT/graymean_w1" \
  >"$PAIR_ROOT/graymean_w1.log" 2>&1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
trap - INT TERM EXIT
exit "$status"
