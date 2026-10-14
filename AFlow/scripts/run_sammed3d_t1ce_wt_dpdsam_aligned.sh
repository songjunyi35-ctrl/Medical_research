#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aflow_env.sh"
ROOT="$AFLOW_REPO_ROOT"
PY="$AFLOW_PYTHON_RESOLVED"
TRAINER="scripts/train_sammed3d_t1ce_wt_dpdsam_aligned.py"
NPROC="${AFLOW_NPROC:-2}"

cd "$ROOT"

# torch >= 2.6 defaults torch.load(weights_only=True); our checkpoints carry a
# pathlib.PosixPath in args. The shim allowlists exactly that one class without
# touching any byte-pinned source file.
export PYTHONPATH="workspace/BraTS/env_shims${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

launch() {
  exec "$PY" -m torch.distributed.run \
    --nproc_per_node="$NPROC" --nnodes=1 \
    --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29517 \
    "$TRAINER" "$@"
}

case "${1:-}" in
  smoke)
    shift
    launch --mode smoke \
      --output-dir "workspace/BraTS/sammed3d_t1ce_wt/smoke_sammed3d_official_fullunfreeze_ddp2_s20260805" "$@"
    ;;
  preflight)
    shift
    launch --mode formal --preflight-only "$@"
    ;;
  run)
    shift
    launch --mode formal "$@"
    ;;
  *)
    echo "Usage: $0 {smoke|preflight|run} [extra trainer args]" >&2
    exit 2
    ;;
esac
