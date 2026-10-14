#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aflow_env.sh"
ROOT="$AFLOW_REPO_ROOT"
PY="$AFLOW_PYTHON_RESOLVED"
MANIFEST="data/datasets/brats2024_sammed3d_t1ce_wt_inner90_10_seed20260723_v1/validation_t1ce_wt.jsonl"
MANIFEST_SHA="fc750ab15266b8a53c6d9a7ff62bd043b10cc21733e9b68354a2a989fd1f488c"

RANK0_DIR="workspace/BraTS/sammed3d_t1ce_wt/fv_rank0_matched_aug_train972_val108_e50_s20260725"
RANK0_CHECKPOINT="${RANK0_DIR}/sam_model_val_dice_best.pth"
RANK0_SHA="7a25d0268799bdfffc626eacecb2a319a51d8127460143e7c6af3e8d4b532cf9"
RANK0_INDEPENDENT="${RANK0_DIR}/INDEPENDENT_RELOAD_VAL108_E32_S20260723_H100_PAIR.json"
RANK0_REPLAY="${RANK0_DIR}/TRAINER_RNG_REPLAY_VAL108_E32_S20260723_H100_PAIR.json"

LORA_DIR="workspace/BraTS/sammed3d_t1ce_wt/fv_lora_r4_aug_train972_val108_e50_s20260725"
LORA_CHECKPOINT="${LORA_DIR}/sam_model_val_dice_best.pth"
LORA_SHA="c1d1bd79791aa7beeffc541095472dec968630cc70ea7967128c86552bd7f1ec"
LORA_INDEPENDENT="${LORA_DIR}/INDEPENDENT_RELOAD_VAL108_E33_S20260723_H100_PAIR.json"
LORA_REPLAY="${LORA_DIR}/TRAINER_RNG_REPLAY_VAL108_E33_S20260723_H100_PAIR.json"

cd "$ROOT"

for output in \
  "$RANK0_INDEPENDENT" \
  "$RANK0_REPLAY" \
  "$LORA_INDEPENDENT" \
  "$LORA_REPLAY"; do
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite an existing validation report: $output" >&2
    exit 2
  fi
done

evaluate_independent() {
  local checkpoint="$1"
  local checkpoint_sha="$2"
  local output="$3"
  PYTHONDONTWRITEBYTECODE=1 "$PY" \
    scripts/evaluate_sammed3d_t1ce_wt_checkpoint.py \
    --manifest "$MANIFEST" \
    --expected-manifest-sha256 "$MANIFEST_SHA" \
    --expected-split validation \
    --checkpoint "$checkpoint" \
    --expected-checkpoint-sha256 "$checkpoint_sha" \
    --spatial-preprocessing whole_resize \
    --device cuda \
    --seed 20260723 \
    --output "$output"
}

evaluate_replay() {
  local checkpoint="$1"
  local checkpoint_sha="$2"
  local output="$3"
  PYTHONDONTWRITEBYTECODE=1 "$PY" \
    scripts/replay_sammed3d_t1ce_wt_val_trainer_rng.py \
    --checkpoint "$checkpoint" \
    --expected-checkpoint-sha256 "$checkpoint_sha" \
    --device cuda \
    --seed 20260723 \
    --output "$output"
}

evaluate_independent "$RANK0_CHECKPOINT" "$RANK0_SHA" "$RANK0_INDEPENDENT"
evaluate_independent "$LORA_CHECKPOINT" "$LORA_SHA" "$LORA_INDEPENDENT"
evaluate_replay "$RANK0_CHECKPOINT" "$RANK0_SHA" "$RANK0_REPLAY"
evaluate_replay "$LORA_CHECKPOINT" "$LORA_SHA" "$LORA_REPLAY"

echo "MATCHED_H100_VALIDATION_DONE rank0=${RANK0_INDEPENDENT} lora=${LORA_INDEPENDENT}"
