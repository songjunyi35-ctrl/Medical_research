#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aflow_env.sh"
ROOT="$AFLOW_REPO_ROOT"
PY="$AFLOW_PYTHON_RESOLVED"
TRAINER="scripts/train_sammed3d_t1ce_wt_partial_unfreeze.py"

cd "$ROOT"

case "${1:-}" in
  preflight)
    PYTHONDONTWRITEBYTECODE=1 "$PY" -u "$TRAINER" \
      --mode formal \
      --device cuda \
      --preflight-only
    ;;
  run)
    # Re-run the complete fail-closed preflight in this foreground worker
    # immediately before replacing the shell with the scientific process.
    PYTHONDONTWRITEBYTECODE=1 "$PY" -u "$TRAINER" \
      --mode formal \
      --device cuda \
      --preflight-only
    exec env PYTHONDONTWRITEBYTECODE=1 "$PY" -u "$TRAINER" \
      --mode formal \
      --device cuda
    ;;
  *)
    echo "Usage: $0 {preflight|run}" >&2
    exit 2
    ;;
esac
