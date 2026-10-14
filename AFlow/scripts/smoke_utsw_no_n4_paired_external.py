"""Synthetic-only dual-model GPU smoke for conditional UTSW362."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from scripts import utsw_no_n4_paired_external_protocol as p
from scripts.evaluate_utsw_no_n4_paired_external import load_model


@torch.inference_mode()
def main(device_name: str) -> None:
    from scripts.freeze_utsw_no_n4_paired_external import check_implementation

    check_implementation()
    p.verify_preaccess()
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("UTSW paired smoke requires CUDA")
    torch.cuda.set_device(device)
    torch.manual_seed(20261380)
    image = torch.randn(2, 1, 128, 128, 128, device=device)
    roles, peaks = {}, {}
    for role in ("canonical", "no_n4"):
        torch.cuda.reset_peak_memory_stats(device)
        model, payload, preaccess = load_model(role, device)
        outputs = []
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs.append(model(image).segmentation_logits.float())
        total = None
        for axes in p.AXES:
            view = torch.flip(image, axes) if axes else image
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(view).segmentation_logits.float()
            logits = torch.flip(logits, axes) if axes else logits
            total = logits.clone() if total is None else total.add_(logits)
        outputs.append(total / 8)
        peaks[role] = torch.cuda.max_memory_allocated(device)
        roles[role] = {
            "checkpoint_sha256": preaccess["models"][role]["checkpoint_sha256"],
            "payload_role": payload["role"],
            "identity_shape": list(outputs[0].shape), "flip8_shape": list(outputs[1].shape),
            "finite": all(bool(torch.isfinite(value).all()) for value in outputs),
            "single_example_forwards": 18,
            "peak_under_24gib": peaks[role] <= 24 * 1024**3,
        }
        del model
        torch.cuda.empty_cache()
    gates = {
        "both_roles": set(roles) == {"canonical", "no_n4"},
        "all_shapes": all(row["identity_shape"] == [2, 1, 128, 128, 128] and row["flip8_shape"] == [2, 1, 128, 128, 128] for row in roles.values()),
        "all_finite": all(row["finite"] for row in roles.values()),
        "forward_accounting": sum(row["single_example_forwards"] for row in roles.values()) == 36,
        "peaks": all(row["peak_under_24gib"] for row in roles.values()),
    }
    report = {
        "schema": "aflow.utsw_no_n4_paired_external.synthetic_smoke",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(gates.values()) else "failed",
        "gates": gates, "roles": roles, "synthetic_only": True,
        "preaccess_sha256": p.sha(p.PREACCESS),
        "utsw_source_access": False, "utsw_metrics": False,
        "runner_sha256": p.sha(Path(__file__)),
    }
    p.write_new(p.RECORD / "GPU_SMOKE_RECEIPT.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not all(gates.values()):
        raise RuntimeError("UTSW paired synthetic smoke failed")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    main(parser.parse_args().device)
