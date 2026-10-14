#!/usr/bin/env python3
"""Corrected EMA validation entry: fixed validation108 contains 50 subjects."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import scripts.evaluate_t1ce_wt_plain_ema_validation as base


SCHEMA = base.SCHEMA + ".v2"


def run(args: argparse.Namespace):
    if base.sha256_file(args.validation_manifest) != base.VALIDATION_MANIFEST_SHA256:
        raise ValueError("historical validation manifest SHA-256 mismatch")
    started = args.output.with_suffix(".started.json")
    if args.output.exists() or started.exists():
        raise FileExistsError("refusing repeated EMA validation")
    case_ids = base.read_case_ids(args.validation_manifest)
    assert case_ids is not None
    data = base.BrainCropDataset(
        args.cache, expected_split="validation", case_ids=case_ids
    )
    subjects = {item.rsplit("-", 1)[0] for item in data.case_ids}
    if len(data) != 108 or len(subjects) != 50:
        raise ValueError("unexpected fixed validation case/subject count")
    if data.index.get("manifest_sha256") != base.VALIDATION_MANIFEST_SHA256:
        raise ValueError("validation cache manifest binding mismatch")
    for filename, field in (("images.f32", "images_sha256"), ("labels.u8", "labels_sha256")):
        if base.sha256_file(args.cache / filename) != data.index[field]:
            raise ValueError(f"validation cache {filename} SHA-256 mismatch")
    base.write_json_atomic(started, {
        "schema": SCHEMA + ".started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_manifest_sha256": base.VALIDATION_MANIFEST_SHA256,
        "cases": 108,
        "subjects": 50,
        "checkpoint_selection_uses_validation": False,
        "threshold": 0.0,
        "tta": False,
        "postprocessing": False,
        "test_access": False
    })
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("EMA validation requires CUDA")
    torch.cuda.set_device(device)
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0
    )
    specs = {
        "raw_168": (args.raw168, "raw", 168),
        "ema_168": (args.ema168, "ema", 168),
        "raw_200": (args.raw200, "raw", 200),
        "ema_200": (args.ema200, "ema", 200)
    }
    summaries = {}
    checkpoint_metadata = {}
    for name, (path, kind, epoch) in specs.items():
        model, checkpoint = base.load_model(path, kind, epoch, device)
        summaries[name] = base.validate(model, loader, device)
        checkpoint_metadata[name] = {
            "path": str(path.resolve()),
            "sha256": base.sha256_file(path),
            "kind": kind,
            "epoch": epoch,
            "ema": checkpoint["ema"]
        }
        del model
        torch.cuda.empty_cache()
    primary = base.compare(summaries["raw_200"], summaries["ema_200"])
    secondary = base.compare(summaries["raw_168"], summaries["ema_168"])
    report = {
        "schema": SCHEMA,
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "historical fixed validation108/50 subjects; development only",
        "not_external_confirmation": True,
        "implementation_amendment": (
            "v1 failed before inference because it incorrectly asserted 45 subjects; "
            "the frozen manifest contains 50 subjects"
        ),
        "protocol": {
            "resolution": 128,
            "threshold": "raw logit > 0",
            "tta": False,
            "postprocessing": False,
            "ema_decay": 0.999,
            "ema_start_after_epoch": 100,
            "primary": "EMA epoch200 minus raw epoch200",
            "secondary": "EMA epoch168 minus raw epoch168"
        },
        "data": {
            "manifest": str(args.validation_manifest.resolve()),
            "manifest_sha256": base.VALIDATION_MANIFEST_SHA256,
            "cache": str(args.cache.resolve()),
            "cache_index_sha256": base.sha256_file(args.cache / "index.json"),
            "cases": 108,
            "subjects": 50
        },
        "checkpoints": checkpoint_metadata,
        "summaries": summaries,
        "primary_epoch200": primary,
        "secondary_epoch168": secondary,
        "ema_candidate_authorized_for_test": primary["all_gates_passed"],
        "selected_candidate": "ema_200" if primary["all_gates_passed"] else None,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False
    }
    base.write_json_atomic(args.output, report)
    print(json.dumps({
        "primary": primary,
        "secondary": secondary,
        "ema_candidate_authorized_for_test": report["ema_candidate_authorized_for_test"]
    }, indent=2, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--raw168", type=Path, required=True)
    parser.add_argument("--ema168", type=Path, required=True)
    parser.add_argument("--raw200", type=Path, required=True)
    parser.add_argument("--ema200", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
