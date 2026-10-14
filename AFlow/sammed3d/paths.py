"""Repository-local paths for the BraTS/SAM-Med3D research checkout.

All defaults stay inside AFlow. Environment variables are explicit escape
hatches for workers or future relocations; callers should not hard-code the
old sibling SAM-Med3D checkout or Python environment.
"""
from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser().resolve()
    return default.resolve()


SAMMED3D_ROOT = _path_from_env(
    "AFLOW_SAMMED3D_ROOT", REPO_ROOT / "third_party" / "SAM-Med3D"
)
SAMMED3D_CHECKPOINT = _path_from_env(
    "AFLOW_SAMMED3D_CHECKPOINT",
    REPO_ROOT / "artifacts" / "checkpoints" / "sam_med3d_turbo.pth",
)
LOCAL_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
LEGACY_SIBLING_PYTHON = REPO_ROOT.parent / "sammed3d-venv" / "bin" / "python"


def require_path(path: Path, description: str) -> Path:
    """Return a resolved path or raise an actionable missing-asset error."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"missing {description}: {resolved}. "
            "Run `bash scripts/doctor_brats_single_repo.sh` for diagnostics."
        )
    return resolved
