#!/usr/bin/env python3
"""CUDA smoke for the UCSD source-correction evaluator using synthetic data only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.evaluate_ucsd_ptgbm_external_confirmation as historical_eval  # noqa: E402
from scripts.evaluate_ucsd_official_float_source_correction import (  # noqa: E402
    FULL972_SHA256,
    GCSAM184_SHA256,
)
from scripts.prepare_ucsd_ptgbm_external_confirmation import sha256_file, write_json_atomic  # noqa: E402


class SyntheticDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, slot: int):
        generator = torch.Generator().manual_seed(2026090600 + slot)
        image = torch.randn((1, 128, 128, 128), generator=generator)
        label = torch.zeros((1, 128, 128, 128), dtype=torch.float32)
        label[:, 48 + slot:72 + slot, 50:76, 52:74] = 1
        return {"image": image, "label": label, "case_id": f"synthetic-{slot}", "subject_id": f"synthetic-{slot}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full972-checkpoint", type=Path, required=True)
    parser.add_argument("--gcsam184-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite smoke receipt")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requires CUDA")
    torch.cuda.set_device(device)
    loader = DataLoader(SyntheticDataset(), batch_size=1, shuffle=False, num_workers=0)
    torch.cuda.reset_peak_memory_stats(device)
    full, _ = historical_eval.load_model(args.full972_checkpoint.resolve(), FULL972_SHA256, "primary", device)
    full_result = historical_eval.evaluate_model(full, loader, device)
    full_peak = torch.cuda.max_memory_allocated(device)
    del full
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    gcsam, _ = historical_eval.load_model(args.gcsam184_checkpoint.resolve(), GCSAM184_SHA256, "secondary", device)
    gcsam_result = historical_eval.evaluate_model(gcsam, loader, device)
    gcsam_peak = torch.cuda.max_memory_allocated(device)
    receipt = {
        "schema": "aflow.ucsd75_official_float_source_correction.synthetic_cuda_smoke",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "synthetic_only": True,
        "ucsd_image_label_or_metric_access": False,
        "models": {
            "full972_kd168": {"checkpoint_sha256": FULL972_SHA256, "forwards": 2, "cases_returned": full_result["cases"], "peak_gpu_memory_bytes": full_peak},
            "gcsam184": {"checkpoint_sha256": GCSAM184_SHA256, "forwards": 2, "cases_returned": gcsam_result["cases"], "peak_gpu_memory_bytes": gcsam_peak},
        },
        "smoke_script": str(Path(__file__).resolve().relative_to(ROOT)),
        "smoke_script_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_atomic(args.output.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
