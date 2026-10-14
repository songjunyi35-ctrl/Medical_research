#!/usr/bin/env bash
set -euo pipefail

# Active worker entrypoint: BraTS T1ce -> WT={1,2,3} only.
# Retired multi-modal, T2-FLAIR, multi-target, GRPO, MedTAWS, and Paper1913
# modes are preserved under archive/retired_non_t1ce_20260810/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aflow_env.sh"
ROOT="$AFLOW_REPO_ROOT"
PY="$AFLOW_PYTHON_RESOLVED"
T1CE_VA_MANIFEST_ROOT="data/datasets/brats2024_t1ce_va_hdfs_20260812_v1"
T1CE_VA_TRAIN_MANIFEST="$T1CE_VA_MANIFEST_ROOT/train_t1ce_wt.jsonl"
T1CE_VA_VAL_MANIFEST="$T1CE_VA_MANIFEST_ROOT/validation_t1ce_wt.jsonl"
T1CE_VA_VAL_SHA256="f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb"

if [[ -n "${AFLOW_B200_SITE:-}" ]]; then
  export LD_LIBRARY_PATH="${AFLOW_B200_SITE}/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

cd "$ROOT"
mode="${1:-probe}"

case "$mode" in
  doctor)
    bash scripts/doctor_brats_single_repo.sh "${@:2}"
    ;;

  probe)
    nvidia-smi
    "$PY" -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda); print("available", torch.cuda.is_available(), "devices", torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]'
    ;;

  sammed3d-t1ce-wt)
    "$PY" scripts/train_sammed3d_t1ce_wt.py \
      --train-jsonl "$T1CE_VA_TRAIN_MANIFEST" \
      --val-jsonl "$T1CE_VA_VAL_MANIFEST" \
      "${@:2}"
    ;;

  eval-sammed3d-t1ce-wt)
    "$PY" scripts/evaluate_sammed3d_t1ce_wt_checkpoint.py \
      --manifest "$T1CE_VA_VAL_MANIFEST" \
      --expected-manifest-sha256 "$T1CE_VA_VAL_SHA256" \
      --expected-split validation \
      "${@:2}"
    ;;

  dpdsam-t1ce-b0)
    "$PY" scripts/train_dpdsam_t1ce_b0.py "${@:2}"
    ;;

  dpdsam-t1ce-a2)
    "$PY" scripts/train_dpdsam_t1ce_a2.py "${@:2}"
    ;;

  dpdsam-t1ce-d1)
    "$PY" scripts/train_dpdsam_t1ce_d1.py "${@:2}"
    ;;

  partial-unfreeze-t1ce-wt)
    "$PY" scripts/train_sammed3d_t1ce_wt_partial_unfreeze.py "${@:2}"
    ;;

  cache-refiner-t1ce-wt)
    "$PY" scripts/cache_sammed3d_t1ce_wt_refiner_inputs.py "${@:2}"
    ;;

  rebuild-refiner-cache)
    bash scripts/rebuild_t1ce_wt_refiner_cache.sh "${2:-verify}" "${@:3}"
    ;;

  train-refiner-t1ce-wt)
    "$PY" scripts/train_sammed3d_t1ce_wt_residual_refiner.py "${@:2}"
    ;;

  cache-mcts-atoms-t1ce-wt)
    "$PY" scripts/cache_t1ce_wt_robust_inference_atoms.py "${@:2}"
    ;;

  promote-mcts-atoms-t1ce-wt)
    "$PY" scripts/promote_t1ce_wt_robust_mcts_atoms.py "${@:2}"
    ;;

  search-mcts-workflow-t1ce-wt)
    "$PY" scripts/search_t1ce_wt_inference_workflow_mcts.py "${@:2}"
    ;;

  reproduce-refiner-t1ce-wt-v2)
    bash scripts/run_t1ce_wt_residual_refiner_reproduction_20260810_v2.sh "${@:2}"
    ;;

  reproduce-official15-t1ce-wt)
    bash scripts/run_t1ce_wt_upstream_literal_va_a100_20260812.sh "${@:2}"
    ;;

  *)
    echo "Unknown T1ce worker mode: $mode" >&2
    echo "Usage: $0 {doctor|probe|sammed3d-t1ce-wt|eval-sammed3d-t1ce-wt|dpdsam-t1ce-b0|dpdsam-t1ce-a2|dpdsam-t1ce-d1|partial-unfreeze-t1ce-wt|cache-refiner-t1ce-wt|rebuild-refiner-cache|train-refiner-t1ce-wt|cache-mcts-atoms-t1ce-wt|promote-mcts-atoms-t1ce-wt|search-mcts-workflow-t1ce-wt|reproduce-refiner-t1ce-wt-v2|reproduce-official15-t1ce-wt} [args...]" >&2
    exit 2
    ;;
esac
