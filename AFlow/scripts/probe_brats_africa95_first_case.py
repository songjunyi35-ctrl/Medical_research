#!/usr/bin/env python3
"""Decode and preprocess only the frozen first Africa95 case before full cache access."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from sammed3d import t1ce_binary
import scripts.build_brats_africa95_cache as builder
from scripts import brats_africa95_protocol as p


def main() -> None:
    if p.FIRST_PROBE.exists(): raise FileExistsError(p.FIRST_PROBE)
    rows = p.cohort_rows(); download = json.loads(p.SOURCE_DOWNLOAD.read_text())
    if download.get("status") != "passed" or download.get("files_copy_exact") != p.FILES or download.get("label_values_accessed") is not False: raise ValueError("source receipt")
    source = {x["relative_path"]: x for x in download["comparisons"]}; row = rows[0]
    torch.set_num_threads(1)
    import SimpleITK as sitk
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    original = t1ce_binary._case_id
    t1ce_binary._case_id = lambda value: value if value == row["case_id"] else original(value)
    image, label, detail = builder.prepare(row, source)
    result = {"schema": "aflow.brats_africa95.first_case_probe", "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "passed", "case_id": row["case_id"], "source_download_sha256": p.sha(p.SOURCE_DOWNLOAD), "native_shape": detail["native_shape"], "spacing": detail["spacing"], "axis_codes": detail["axis_codes"], "affine_max_abs": detail["affine_max_abs"], "raw_label_values": detail["raw_label_values"], "raw_wt_voxels": detail["raw_wt_voxels"], "resized_wt_voxels": detail["resized_wt_voxels"], "output_image_sha256": detail["image_sha256"], "output_label_sha256": detail["label_sha256"], "output_shape": list(image.shape), "output_label_values": sorted(int(x) for x in __import__("numpy").unique(label)), "model_inference": False, "other_case_arrays_accessed": False, "runner_sha256": p.sha(Path(__file__))}
    p.write_new(p.FIRST_PROBE, result); print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
