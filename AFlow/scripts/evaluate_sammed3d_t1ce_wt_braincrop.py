#!/usr/bin/env python
"""Evaluate a braincrop-geometry checkpoint, reporting the geometry honestly.

`scripts/evaluate_sammed3d_t1ce_wt_cached.py` is byte-pinned by the completed
full-unfreeze test freeze (SHA b3acbb83...), and its `_geometry_description`
hardcodes two cases: `gt_crop` -> oracle localization, anything else -> "canonical
whole-volume resize to 128^3". Running a braincrop cache through it unmodified
would emit a report that misdescribes its own geometry.

This wrapper changes exactly one thing -- the geometry sentence -- and delegates
everything else (click protocol, metrics, subject-balanced aggregation, the
sealed-test gate, the strict checkpoint reload) to that module's own code. The
pinned file's bytes are not touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.evaluate_sammed3d_t1ce_wt_cached as cached  # noqa: E402

_ORIGINAL = cached._geometry_description


def _geometry_description(index: dict) -> str:
    spatial = index["preprocessing"]["spatial"]
    if spatial == "braincrop_resize":
        return (
            "crop to the bounding box of nonzero image voxels, then resize that "
            "box to 128^3. The box is derived from image intensity only, NOT from "
            "the label, so localization is not oracle and this remains comparable "
            "with automatic segmentation methods."
        )
    return _ORIGINAL(index)


cached._geometry_description = _geometry_description
# Report under this file's identity, not the delegate's, so the recorded
# evaluator SHA resolves to the code that actually ran.
cached.SCHEMA = "aflow.sammed3d.t1ce_wt_braincrop_evaluation"

_original_evaluate = cached.evaluate


def evaluate(args):
    report = _original_evaluate(args)
    report["evaluator"] = "scripts/evaluate_sammed3d_t1ce_wt_braincrop.py"
    report["evaluator_sha256"] = cached.sha256_file(Path(__file__).resolve())
    report["delegate_evaluator_sha256"] = cached.sha256_file(
        Path(cached.__file__).resolve()
    )
    cached.write_json_atomic(args.output.resolve(), report)
    return report


cached.evaluate = evaluate

if __name__ == "__main__":
    cached.main()
