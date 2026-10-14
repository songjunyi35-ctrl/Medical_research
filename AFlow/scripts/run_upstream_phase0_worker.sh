#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aflow_env.sh"
PY="${AFLOW_B200_PYTHON:-$AFLOW_PYTHON_RESOLVED}"
ROOT="$AFLOW_SAMMED3D_ROOT"

cd "$ROOT"
exec "$PY" phase0_reproduce.py --ckpt "$AFLOW_SAMMED3D_CHECKPOINT" "$@"
