#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 1 && "$1" == "registry" ]]; then
  ACTION="registry"
elif [[ "$#" -eq 3 ]]; then
  ACTION="train"
  GPU="$1"
  ARM="$2"
  RUN_MODE="$3"
  case "$GPU:$ARM" in
    0:plain|1:domain) ;;
    *) echo "preregistered mapping is GPU0=plain, GPU1=domain" >&2; exit 2 ;;
  esac
  case "$RUN_MODE" in
    smoke|formal) ;;
    *) echo "run mode must be smoke or formal" >&2; exit 2 ;;
  esac
else
  echo "usage: $0 registry | GPU {plain|domain} {smoke|formal}" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/aflow_env.sh"
cd "$REPO_ROOT"
REFINER_CACHE_ROOT="${AFLOW_REFINER_CACHE_ROOT:-$REPO_ROOT/workspace/BraTS/cache/t1ce_wt_refiner_graymean_20260813}"
SOURCE_CHECKPOINT="$REPO_ROOT/workspace/BraTS/sammed3d_t1ce_wt/residual_refiner_robustshift4_three_seed_ensemble_20260813/robustshift4_seed20260815/refiner_val_dice_best.pth"
SOURCE_SHA256="c5d2b48db2666f097cb0d86e78c20ea95d163c07f98401714d25d755e99aceb0"
REGISTRY_ROOT="$REPO_ROOT/workspace/BraTS/sammed3d_t1ce_wt/domain_sage_registry_20260814"

if [[ "$ACTION" == "registry" ]]; then
  if [[ -e "$REGISTRY_ROOT" ]]; then
    echo "Refusing to overwrite existing registry: $REGISTRY_ROOT" >&2
    exit 2
  fi
  env \
    CUDA_VISIBLE_DEVICES=1 \
    LD_LIBRARY_PATH="/tmp/aflow_nvidia_535_161_08:${LD_LIBRARY_PATH:-}" \
    PYTHONDONTWRITEBYTECODE=1 \
    "$AFLOW_PYTHON_RESOLVED" -u \
    -m scripts.build_t1ce_wt_domain_sage_registry \
    --train-cache "$REFINER_CACHE_ROOT/train" \
    --validation-cache "$REFINER_CACHE_ROOT/validation" \
    --init-checkpoint "$SOURCE_CHECKPOINT" \
    --expected-init-sha256 "$SOURCE_SHA256" \
    --output-dir "$REGISTRY_ROOT" \
    --device cuda:0 \
    --seed 20260818 \
    --kmeans-seed 20260818 \
    --domain-count 4 \
    --batch-size 2 \
    --num-workers 4
  exit 0
fi

if [[ ! -f "$REGISTRY_ROOT/registry.json" || ! -f "$REGISTRY_ROOT/registry.npz" ]]; then
  echo "Domain-SAGE registry is missing: $REGISTRY_ROOT" >&2
  exit 2
fi
if [[ "$RUN_MODE" == "formal" ]]; then
  ROOT_OUT="$REPO_ROOT/workspace/BraTS/sammed3d_t1ce_wt/domain_sage_pair_20260814"
  EXTRA_ARGS=(
    --epochs 20
    --patience 8
    --expected-baseline-dsc 0.6990433552267926
    --baseline-tolerance 0.00001
  )
else
  ROOT_OUT="$REPO_ROOT/workspace/BraTS/sammed3d_t1ce_wt/domain_sage_smoke_20260814"
  EXTRA_ARGS=(
    --epochs 1
    --patience 1
    --limit-train-cases 4
    --limit-validation-cases 4
  )
fi
ARM_NAME="domain_sage_${ARM}_seed20260818"
OUTPUT="$ROOT_OUT/$ARM_NAME"
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite existing arm: $OUTPUT" >&2
  exit 2
fi
mkdir -p "$ROOT_OUT"

env \
  CUDA_VISIBLE_DEVICES="$GPU" \
  LD_LIBRARY_PATH="/tmp/aflow_nvidia_535_161_08:${LD_LIBRARY_PATH:-}" \
  PYTHONDONTWRITEBYTECODE=1 \
  "$AFLOW_PYTHON_RESOLVED" -u \
  -m scripts.train_t1ce_wt_domain_sage \
  --train-cache "$REFINER_CACHE_ROOT/train" \
  --validation-cache "$REFINER_CACHE_ROOT/validation" \
  --domain-registry "$REGISTRY_ROOT" \
  --init-checkpoint "$SOURCE_CHECKPOINT" \
  --expected-init-sha256 "$SOURCE_SHA256" \
  --output-dir "$OUTPUT" \
  --arm-name "$ARM_NAME" \
  --arm "$ARM" \
  --device cuda:0 \
  --seed 20260818 \
  --batch-size 2 \
  --num-workers 4 \
  --base-init-filters 16 \
  --base-dropout 0.1 \
  --adapter-init-filters 8 \
  --adapter-dropout 0.1 \
  --domain-count 4 \
  --domain-hidden-dim 64 \
  --film-rank 4 \
  --domain-weight 0.1 \
  --lr 0.0001 \
  --weight-decay 0.00001 \
  --max-shift 12 \
  --coarse-shift-max 4 \
  --coarse-shift-probability 1.0 \
  "${EXTRA_ARGS[@]}" \
  >"$ROOT_OUT/$ARM_NAME.log" 2>&1
