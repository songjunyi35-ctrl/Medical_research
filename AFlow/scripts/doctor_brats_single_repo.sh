#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aflow_env.sh"
cd "$AFLOW_REPO_ROOT"
export PYTHONDONTWRITEBYTECODE=1
exec "$AFLOW_PYTHON_RESOLVED" scripts/doctor_brats_single_repo.py "$@"
