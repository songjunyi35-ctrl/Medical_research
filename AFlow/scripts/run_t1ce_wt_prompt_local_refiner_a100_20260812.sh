#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aflow_env.sh"
ROOT="$AFLOW_REPO_ROOT"
PY="$AFLOW_PYTHON_RESOLVED"
SOURCE_CACHE="$ROOT/workspace/BraTS/cache/t1ce_wt_gtcrop_n4_128"
SAM_CHECKPOINT="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/sammed3d_native_gtcrop_fullunfreeze_ddp2_train972_val108_e100_s20260805/sam_model_val_dice_best.pth"
SAM_SHA="42f96a0a2ade9d98d9db897a11a5a285f7a6351a7693c6884d6ac97a29dfde52"
GLOBAL_CHECKPOINT="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/residual_refiner_stage2_gtcrop_h100_train972_val108_e60_s20260810_v2/refiner_val_dice_best.pth"
GLOBAL_SHA="4513ea4544ae153064575ec2ed5c56e1b78adf5e89e9d094754227b3bd429f21"
PREREG="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/PROMPT_LOCAL_REFINER_PREREG_20260812.json"
CACHE_ROOT="/tmp/aflow_prompt_local_cache_20260812"
RUN_ROOT="$ROOT/workspace/BraTS/sammed3d_t1ce_wt"
SMOKE_RUN="$RUN_ROOT/smoke_prompt_local64_ddp2_a100_e2_s20260812_v3"
FORMAL_RUN="$RUN_ROOT/prompt_local64_ddp2_a100_train972_val108_e80_s20260812"

DRIVER_LIB_DIR="/tmp/aflow_nvidia_535_161_08"
mkdir -p "$DRIVER_LIB_DIR"
ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.535.161.08 "$DRIVER_LIB_DIR/libcuda.so.1"
ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.535.161.08 "$DRIVER_LIB_DIR/libcuda.so"
ln -sfn /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.161.08 "$DRIVER_LIB_DIR/libnvidia-ml.so.1"
ln -sfn /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.535.161.08 "$DRIVER_LIB_DIR/libnvidia-ptxjitcompiler.so.1"
export LD_LIBRARY_PATH="$DRIVER_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT/workspace/BraTS/env_shims${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

cd "$ROOT"

preflight() {
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
  sha256sum -c <<EOF
2e72534e8d9f0947278f2ac0f4a1612cf8f2c1512cdd50556a268db129c58779  scripts/cache_t1ce_wt_prompt_local_refiner_inputs.py
b7055502687caa11cb63a494a29d96d3ebf07aef1647927ec4e2718b2b258b88  scripts/merge_t1ce_wt_prompt_local_refiner_cache.py
ea487b5dc6fbea7ecd238f3d7521d3784726d447e5f1e63bf859a1ec190cf038  scripts/train_t1ce_wt_prompt_local_refiner.py
b80f8d9d4d5f649748f97b50794456f8f512fd1549c633df9687227d7f4752e6  tests/test_prompt_local_refiner.py
99324e6ae3d71f38c058de8cb2c0afc7bee079fbaf6af521257e55d1259f9ebd  workspace/BraTS/cache/t1ce_wt_gtcrop_n4_128/train/index.json
179a5ce1a89bf41b07ae633200950ead3d485e01e49ff8d9494437be531461da  workspace/BraTS/cache/t1ce_wt_gtcrop_n4_128/validation/index.json
$SAM_SHA  workspace/BraTS/sammed3d_t1ce_wt/sammed3d_native_gtcrop_fullunfreeze_ddp2_train972_val108_e100_s20260805/sam_model_val_dice_best.pth
$GLOBAL_SHA  workspace/BraTS/sammed3d_t1ce_wt/residual_refiner_stage2_gtcrop_h100_train972_val108_e60_s20260810_v2/refiner_val_dice_best.pth
EOF
  "$PY" - <<'PY'
import json
from pathlib import Path
p = Path("workspace/BraTS/sammed3d_t1ce_wt/PROMPT_LOCAL_REFINER_PREREG_20260812.json")
d = json.loads(p.read_text())
assert d["status"] == "frozen_before_arm_cache_generation_gpu_smoke_or_training"
assert d["test_access"] is False
assert d["arm"]["single_fixed_geometry"] is True
amendment = json.loads(Path("workspace/BraTS/sammed3d_t1ce_wt/PROMPT_LOCAL_REFINER_PREREG_AMENDMENT_WORKER_EXIT_20260812.json").read_text())
assert amendment["status"] == "frozen_after_engineering_smoke_and_before_formal_training"
assert amendment["test_access"] is False
nccl_amendment = json.loads(Path("workspace/BraTS/sammed3d_t1ce_wt/PROMPT_LOCAL_REFINER_PREREG_AMENDMENT_NCCL_EXIT_20260812.json").read_text())
assert nccl_amendment["status"] == "frozen_after_second_engineering_smoke_and_before_formal_training"
assert nccl_amendment["test_access"] is False
print("preregistration guard: pass")
PY
  "$PY" -m pytest -q tests/test_prompt_local_refiner.py
}

cache_gpu() {
  local gpu="$1"
  local shard="$2"
  for split in train validation; do
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/cache_t1ce_wt_prompt_local_refiner_inputs.py \
      --cache-dir "$SOURCE_CACHE/$split" \
      --split "$split" \
      --sam-checkpoint "$SAM_CHECKPOINT" \
      --expected-sam-sha256 "$SAM_SHA" \
      --global-refiner-checkpoint "$GLOBAL_CHECKPOINT" \
      --expected-global-refiner-sha256 "$GLOBAL_SHA" \
      --output-dir "$CACHE_ROOT/shards/$split-$shard" \
      --device cuda:0 \
      --seed 20260723 \
      --shard-index "$shard" \
      --shard-count 2
  done
}

build_cache() {
  if [[ -e "$CACHE_ROOT/merged/train/index.json" || -e "$CACHE_ROOT/merged/validation/index.json" ]]; then
    echo "merged prompt-local cache already exists; refusing to rebuild" >&2
    exit 2
  fi
  cache_gpu 0 0 &
  local first_pid=$!
  cache_gpu 1 1 &
  local second_pid=$!
  wait "$first_pid"
  wait "$second_pid"
  "$PY" scripts/merge_t1ce_wt_prompt_local_refiner_cache.py \
    --shard-dirs "$CACHE_ROOT/shards/train-0" "$CACHE_ROOT/shards/train-1" \
    --output-dir "$CACHE_ROOT/merged/train"
  "$PY" scripts/merge_t1ce_wt_prompt_local_refiner_cache.py \
    --shard-dirs "$CACHE_ROOT/shards/validation-0" "$CACHE_ROOT/shards/validation-1" \
    --output-dir "$CACHE_ROOT/merged/validation"
}

train_run() {
  local output_dir="$1"
  local epochs="$2"
  local patience="$3"
  local port="$4"
  "$PY" -m torch.distributed.run \
    --nproc_per_node=2 --nnodes=1 \
    --rdzv_backend=c10d --rdzv_endpoint="127.0.0.1:$port" \
    scripts/train_t1ce_wt_prompt_local_refiner.py \
    --train-cache "$CACHE_ROOT/merged/train" \
    --validation-cache "$CACHE_ROOT/merged/validation" \
    --output-dir "$output_dir" \
    --seed 20260812 \
    --epochs "$epochs" \
    --patience "$patience" \
    --batch-size 4 \
    --validation-batch-size 4 \
    --num-workers 4 \
    --patch-size 64 \
    --click-sigma 3 \
    --init-filters 16 \
    --dropout 0.1 \
    --lr 2e-4 \
    --weight-decay 1e-5
}

smoke() {
  train_run "$SMOKE_RUN" 2 2 29541
}

formal() {
  if [[ ! -f "$SMOKE_RUN/TRAIN_DONE.json" ]]; then
    echo "formal run requires completed DDP smoke" >&2
    exit 2
  fi
  train_run "$FORMAL_RUN" 80 15 29542
}

case "${1:-all}" in
  preflight)
    preflight
    ;;
  cache)
    preflight
    build_cache
    ;;
  smoke)
    preflight
    smoke
    ;;
  train)
    preflight
    formal
    ;;
  all)
    preflight
    build_cache
    smoke
    formal
    ;;
  *)
    echo "Usage: $0 {preflight|cache|smoke|train|all}" >&2
    exit 2
    ;;
esac
