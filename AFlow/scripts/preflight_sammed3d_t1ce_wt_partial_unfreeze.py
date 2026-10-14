#!/usr/bin/env python3
"""CPU-only preflight for the independent last-four-block training arm."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d import engine, partial_unfreeze, peft  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly load a base SAM-Med3D checkpoint on CPU and prove the "
            "fixed blocks-8--11 discriminative-LR parameter contract."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(engine.DEFAULT_CKPT),
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()

    requested = peft.make_adapter_config(
        target_token=False,
        lora_rank=0,
        model_type=engine.MODEL_TYPE,
    )
    model, adapter_config = peft.load_model_with_adapters(
        checkpoint,
        lambda model_type: engine.sam_model_registry3D[model_type](
            checkpoint=None
        ),
        device=torch.device("cpu"),
        requested_config=requested,
        default_model_type=engine.MODEL_TYPE,
    )
    if peft.adapters_enabled(adapter_config):
        raise RuntimeError("partial-unfreeze preflight requires adapters disabled")

    selection = partial_unfreeze.configure_vit_b_ori_last_four(model)
    optimizer = partial_unfreeze.build_adamw(selection)
    report = {
        "schema": "aflow.sammed3d.t1ce_wt_partial_unfreeze_preflight",
        "schema_version": 1,
        "training_started": False,
        "device": "cpu",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "adapter_config": adapter_config,
        "partial_unfreeze_policy": selection.record,
        "optimizer_group_count": len(optimizer.param_groups),
        "preflight_sha256": sha256_file(Path(__file__).resolve()),
        "policy_helper_sha256": sha256_file(
            Path(partial_unfreeze.__file__).resolve()
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
