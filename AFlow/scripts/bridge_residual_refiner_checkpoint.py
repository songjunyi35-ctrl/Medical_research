#!/usr/bin/env python3
"""Convert a provenance-rich refiner checkpoint into a continuation source.

Only the checkpoint envelope changes; model tensors are copied exactly and
verified by a deterministic state-dict digest before and after conversion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_sammed3d_t1ce_wt_residual_refiner import sha256_file  # noqa: E402


TARGET_SCHEMA = "aflow.sammed3d.t1ce_wt_residual_refiner_checkpoint"


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def run(args: argparse.Namespace) -> None:
    source_sha256 = sha256_file(args.source)
    if args.expected_source_sha256 and source_sha256 != args.expected_source_sha256:
        raise RuntimeError("source checkpoint hash mismatch")
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    state_digest = state_dict_sha256(source["model_state_dict"])
    source_epoch = int(source.get("source_epoch", 0))
    continuation_epoch = int(source.get("continuation_epoch", source.get("epoch", 0)))
    output = {
        "schema": TARGET_SCHEMA,
        "schema_version": 1,
        "epoch": source_epoch + continuation_epoch,
        "validation": source["validation"],
        "model_state_dict": source["model_state_dict"],
        "optimizer_state_dict": source.get("optimizer_state_dict", {}),
        "architecture": source["architecture"],
        "bridge_provenance": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_path": str(args.source.resolve()),
            "source_sha256": source_sha256,
            "source_schema": source.get("schema"),
            "model_state_dict_sha256": state_digest,
            "optimizer_will_be_reset_by_continuation": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.save(output, args.output)
    reloaded = torch.load(args.output, map_location="cpu", weights_only=False)
    if state_dict_sha256(reloaded["model_state_dict"]) != state_digest:
        raise RuntimeError("bridged model state changed")
    print(
        {
            "output": str(args.output),
            "sha256": sha256_file(args.output),
            "epoch": output["epoch"],
            "validation_dsc": output["validation"]["refined_dsc_mean"],
            "model_state_dict_sha256": state_digest,
        },
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", default="")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
