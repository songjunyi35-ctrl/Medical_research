#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aflow_env.sh"
ROOT="$AFLOW_REPO_ROOT"
PY="$AFLOW_PYTHON_RESOLVED"
SOURCE_CACHE="${AFLOW_REFINER_SOURCE_CACHE:-$ROOT/workspace/BraTS/cache/t1ce_wt_gtcrop_n4_128}"
OUTPUT_ROOT="${AFLOW_REFINER_CACHE_ROOT:-$ROOT/workspace/BraTS/cache/t1ce_wt_refiner_graymean_20260813}"
CHECKPOINT="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/sammed3d_native_gtcrop_fullunfreeze_ddp2_train972_val108_e100_s20260805/sam_model_val_dice_best.pth"
CHECKPOINT_SHA="42f96a0a2ade9d98d9db897a11a5a285f7a6351a7693c6884d6ac97a29dfde52"

cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="workspace/BraTS/env_shims${PYTHONPATH:+:$PYTHONPATH}"

verify() {
  "$PY" scripts/verify_t1ce_wt_refiner_cache.py --cache-root "$OUTPUT_ROOT" "$@"
}

build_split() {
  local split="$1"
  local gpu="$2"
  local max_cases="${3:-}"
  local output="$OUTPUT_ROOT/$split"
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite existing cache split: $output" >&2
    return 2
  fi
  local extra=()
  if [[ -n "$max_cases" ]]; then
    extra+=(--max-cases "$max_cases")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u scripts/cache_sammed3d_t1ce_wt_refiner_inputs.py \
    --cache-dir "$SOURCE_CACHE/$split" \
    --split "$split" \
    --checkpoint "$CHECKPOINT" \
    --expected-checkpoint-sha256 "$CHECKPOINT_SHA" \
    --output-dir "$output" \
    --device cuda:0 \
    --seed 20260723 \
    "${extra[@]}"
}

case "${1:-verify}" in
  verify)
    verify "${@:2}"
    ;;
  build)
    mkdir -p "$(dirname "$OUTPUT_ROOT")"
    build_split train 0 &
    train_pid=$!
    build_split validation 1 &
    val_pid=$!
    status=0
    wait "$train_pid" || status=$?
    wait "$val_pid" || status=$?
    if [[ "$status" -ne 0 ]]; then
      exit "$status"
    fi
    verify --full-hash
    ;;
  smoke)
    OUTPUT_ROOT="${AFLOW_REFINER_SMOKE_CACHE_ROOT:-$ROOT/workspace/BraTS/cache/smoke_single_repo_refiner}"
    mkdir -p "$(dirname "$OUTPUT_ROOT")"
    build_split train 0 1
    build_split validation 1 1
    echo "Smoke cache written to $OUTPUT_ROOT"
    ;;
  *)
    echo "Usage: $0 {verify|build|smoke}" >&2
    exit 2
    ;;
esac
