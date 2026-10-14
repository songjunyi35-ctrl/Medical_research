"""Synthetic-only Flip8 GPU smoke for the future RHUH external checkpoint."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from scripts import rhuh_no_n4_external_protocol as p
from scripts.evaluate_rhuh_no_n4_external import load_model


@torch.inference_mode()
def main(device_name: str) -> None:
    from scripts.freeze_rhuh_no_n4_external import check_implementation

    check_implementation()
    preaccess = p.verify_preaccess()
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RHUH smoke requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(20261360)
    model, _, _ = load_model(device)
    image = torch.randn(2, 1, 128, 128, 128, device=device)
    total = None
    for axes in p.AXES:
        view = torch.flip(image, axes) if axes else image
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(view).segmentation_logits.float()
        logits = torch.flip(logits, axes) if axes else logits
        total = logits.clone() if total is None else total.add_(logits)
    output = total / 8
    gates = {
        "shape": list(output.shape) == [2, 1, 128, 128, 128],
        "finite": bool(torch.isfinite(output).all()),
        "forwards": 16 == 2 * len(p.AXES),
        "peak_under_24gib": torch.cuda.max_memory_allocated(device) <= 24 * 1024 ** 3,
    }
    report = {
        "schema": "aflow.rhuh_no_n4_external.synthetic_smoke",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(gates.values()) else "failed",
        "gates": gates,
        "checkpoint_sha256": preaccess["checkpoint"]["sha256"],
        "synthetic_only": True,
        "rhuh_label_access": False,
        "rhuh_model_metrics": False,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "runner_sha256": p.sha(Path(__file__)),
    }
    p.write_new(p.RECORD / "GPU_SMOKE_RECEIPT.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not all(gates.values()):
        raise RuntimeError("RHUH synthetic GPU smoke failed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    main(parser.parse_args().device)
