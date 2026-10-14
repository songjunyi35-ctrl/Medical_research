#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aflow_env.sh"
ROOT="$AFLOW_REPO_ROOT"
PY="$AFLOW_PYTHON_RESOLVED"
MANIFEST="data/datasets/brats2024_sammed3d_t1ce_wt_inner90_10_seed20260723_v1/validation_t1ce_wt.jsonl"
MANIFEST_SHA="fc750ab15266b8a53c6d9a7ff62bd043b10cc21733e9b68354a2a989fd1f488c"
RUN_DIR="workspace/BraTS/sammed3d_t1ce_wt/fv_rank0_matched_aug_train972_val108_e50_s20260725"
CHECKPOINT="${RUN_DIR}/sam_model_val_dice_best.pth"
CHECKPOINT_SHA="7a25d0268799bdfffc626eacecb2a319a51d8127460143e7c6af3e8d4b532cf9"
INDEPENDENT="${RUN_DIR}/INDEPENDENT_RELOAD_VAL108_E32_S20260723.json"
REPLAY="${RUN_DIR}/TRAINER_RNG_REPLAY_VAL108_E32_S20260723.json"

cd "$ROOT"

if [[ -e "$INDEPENDENT" || -e "$REPLAY" ]]; then
  echo "Refusing to overwrite an existing validation report." >&2
  exit 2
fi

PYTHONDONTWRITEBYTECODE=1 "$PY" \
  scripts/evaluate_sammed3d_t1ce_wt_checkpoint.py \
  --manifest "$MANIFEST" \
  --expected-manifest-sha256 "$MANIFEST_SHA" \
  --expected-split validation \
  --checkpoint "$CHECKPOINT" \
  --expected-checkpoint-sha256 "$CHECKPOINT_SHA" \
  --spatial-preprocessing whole_resize \
  --device cuda \
  --seed 20260723 \
  --output "$INDEPENDENT"

PYTHONDONTWRITEBYTECODE=1 "$PY" \
  scripts/replay_sammed3d_t1ce_wt_val_trainer_rng.py \
  --checkpoint "$CHECKPOINT" \
  --expected-checkpoint-sha256 "$CHECKPOINT_SHA" \
  --device cuda \
  --seed 20260723 \
  --output "$REPLAY"

echo "VALIDATION_RELOAD_DONE independent=${INDEPENDENT} replay=${REPLAY}"
