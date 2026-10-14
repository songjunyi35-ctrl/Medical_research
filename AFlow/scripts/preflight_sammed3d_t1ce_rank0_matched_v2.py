#!/usr/bin/env python3
"""Hash-bound wrapper for the matched T1ce/WT whole-volume rank-0 preflight."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from preflight_sammed3d_t1ce_rank0_matched import (
    ROOT,
    require_equal,
    resolve_bound_file,
    run_preflight,
    sha256_file,
)


DEFAULT_PREREG = (
    ROOT
    / "workspace/BraTS/sammed3d_t1ce_wt/"
    "FV_RANK0_MATCHED_AUG_PREREG_20260724_V2.json"
)
EXPECTED_LAUNCHER = (
    ROOT / "scripts/launch_sammed3d_t1ce_rank0_matched_a10_v2.sh"
).resolve()
SCHEMA = "aflow.sammed3d.t1ce_wt.fv_rank0_matched_preflight.v2"


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return payload


def validate_recorded_hash(
    root: Path,
    record: dict[str, Any],
    label: str,
) -> Path:
    if not isinstance(record, dict):
        raise RuntimeError(f"{label} binding must be a JSON object")
    path = resolve_bound_file(root, record.get("path"), label)
    recorded = record.get("sha256")
    if (
        not isinstance(recorded, str)
        or len(recorded) != 64
        or any(character not in "0123456789abcdef" for character in recorded)
    ):
        raise RuntimeError(f"{label} prereg SHA256 is invalid")
    require_equal(sha256_file(path), recorded, f"{label} byte SHA256")
    return path


def run_preflight_v2(
    prereg_path: Path,
    root: Path = ROOT,
) -> dict[str, Any]:
    prereg_path = prereg_path.resolve()
    prereg = load_json(prereg_path, "rank-0 v2 preregistration")
    source = prereg.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("prereg source must be an object")
    preflight_path = validate_recorded_hash(
        root, source.get("preflight"), "preflight"
    )
    launcher_path = validate_recorded_hash(
        root, source.get("launcher"), "launcher"
    )
    require_equal(
        preflight_path,
        Path(__file__).resolve(),
        "preflight bound path",
    )
    require_equal(
        launcher_path,
        EXPECTED_LAUNCHER,
        "launcher bound path",
    )
    report = run_preflight(prereg_path, root=root)
    report["schema"] = SCHEMA
    report["source_binding"] = {
        "preflight_sha256": sha256_file(preflight_path),
        "launcher_sha256": sha256_file(launcher_path),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    args = parser.parse_args()
    print(json.dumps(run_preflight_v2(args.prereg), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
