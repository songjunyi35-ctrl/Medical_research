#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aflow_env.sh"
ROOT="$AFLOW_REPO_ROOT"
PY="$AFLOW_PYTHON_RESOLVED"
OUT_REL="workspace/BraTS/sammed3d_t1ce_wt/smoke_fv_partial_last4_disclr10_2train_2val_e1_s20260725"

cd "$ROOT"

if [[ -e "$OUT_REL" ]]; then
  echo "Refusing to reuse smoke output path: $OUT_REL" >&2
  exit 2
fi

exec env PYTHONDONTWRITEBYTECODE=1 "$PY" -u \
  scripts/train_sammed3d_t1ce_wt_partial_unfreeze.py \
  --mode smoke \
  --device cuda \
  --output-dir "$OUT_REL"
