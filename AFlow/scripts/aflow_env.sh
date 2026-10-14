#!/usr/bin/env bash
# Shared, relocatable environment resolution for AFlow BraTS entrypoints.
# Source this file; do not execute it directly.

if [[ -z "${AFLOW_REPO_ROOT:-}" ]]; then
  AFLOW_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

export AFLOW_REPO_ROOT
export AFLOW_SAMMED3D_ROOT="${AFLOW_SAMMED3D_ROOT:-$AFLOW_REPO_ROOT/third_party/SAM-Med3D}"
export AFLOW_SAMMED3D_CHECKPOINT="${AFLOW_SAMMED3D_CHECKPOINT:-$AFLOW_REPO_ROOT/artifacts/checkpoints/sam_med3d_turbo.pth}"

# Some managed GPU workers expose the driver libraries through this audited
# compatibility directory instead of the base image linker path.  Keeping the
# discovery here makes doctor/train/eval behavior consistent.  A caller can
# override it with AFLOW_NVIDIA_DRIVER_LIB; ordinary hosts simply skip it.
AFLOW_NVIDIA_DRIVER_LIB="${AFLOW_NVIDIA_DRIVER_LIB:-/tmp/aflow_nvidia_535_161_08}"
if [[ -d "$AFLOW_NVIDIA_DRIVER_LIB" ]]; then
  export LD_LIBRARY_PATH="$AFLOW_NVIDIA_DRIVER_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Avoid worker-specific read-only home configuration failures during imports.
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/aflow_matplotlib_cache}"

aflow_resolve_python() {
  local candidate
  if [[ -n "${AFLOW_PYTHON:-}" ]]; then
    candidate="$AFLOW_PYTHON"
  elif [[ -x "$AFLOW_REPO_ROOT/.venv/bin/python" ]]; then
    candidate="$AFLOW_REPO_ROOT/.venv/bin/python"
  elif [[ -x "$AFLOW_REPO_ROOT/../sammed3d-venv/bin/python" ]]; then
    # Temporary compatibility fallback. New environments should use .venv.
    candidate="$AFLOW_REPO_ROOT/../sammed3d-venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    candidate="$(command -v python)"
  else
    echo "No usable Python found. Set AFLOW_PYTHON or create AFlow/.venv." >&2
    return 1
  fi
  if [[ ! -x "$candidate" ]]; then
    echo "Resolved Python is not executable: $candidate" >&2
    return 1
  fi
  printf '%s\n' "$candidate"
}

AFLOW_PYTHON_RESOLVED="$(aflow_resolve_python)"
export AFLOW_PYTHON_RESOLVED
