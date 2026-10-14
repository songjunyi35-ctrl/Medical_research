#!/usr/bin/env python3
"""One-time identity-versus-Fixed-Flip8 evaluation on reused internal test270."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator, LocatorOutput, automatic_prompt_points  # noqa: E402
import scripts.evaluate_t1ce_wt_flip8_tta_replication_inner as inner  # noqa: E402
from scripts.evaluate_t1ce_wt_plain_fixed_test270 import intervals  # noqa: E402
from scripts.evaluate_t1ce_wt_plain_locator_test270 import (  # noqa: E402
    TEST_CASES,
    TEST_MANIFEST_SHA256,
    TEST_SUBJECTS,
    TestBrainCropDataset,
)
import scripts.freeze_t1ce_wt_flip8_tta_test270_preaccess as freeze  # noqa: E402
from scripts.train_t1ce_wt_capg3d import dice_per_case, sha256_file, summarize_cases, write_json_atomic  # noqa: E402
import scripts.train_t1ce_wt_voxel_kd_fulltrajectory_final as full972  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_flip8_tta_test270"


def assert_identity_parity(current: dict[str, Any], archived: dict[str, Any]) -> None:
    if current != archived:
        current_rows = current.get("per_case", [])
        archived_rows = archived.get("per_case", [])
        first_difference = next(
            (
                index
                for index, (left, right) in enumerate(zip(current_rows, archived_rows))
                if left != right
            ),
            None,
        )
        raise RuntimeError(f"identity parity failed; first_per_case_difference={first_difference}")


def success_gates(identity: dict[str, Any], flip8: dict[str, Any]) -> dict[str, bool]:
    identity_hits = sum(bool(row["automatic_point_hits_gt"]) for row in identity["per_case"])
    flip8_hits = sum(bool(row["automatic_point_hits_gt"]) for row in flip8["per_case"])
    return {
        "flip8_scan_strictly_above_current_leader": float(flip8["locator_dsc_mean"]) > freeze.CURRENT_SCAN_LEADER,
        "flip8_scan_strictly_above_identity": float(flip8["locator_dsc_mean"]) > float(identity["locator_dsc_mean"]),
        "flip8_subject_at_least_identity": float(flip8["locator_subject_dsc_mean"]) >= float(identity["locator_subject_dsc_mean"]),
        "flip8_point_loses_at_most_one": flip8_hits - identity_hits >= -1,
    }


def _rows_from_logits(logits: torch.Tensor, label: torch.Tensor, batch: dict[str, Any]) -> list[dict[str, Any]]:
    dsc = dice_per_case(logits, label)
    points = automatic_prompt_points(LocatorOutput(segmentation_logits=logits, center_logits=None)).long()
    rows: list[dict[str, Any]] = []
    for index, case_id in enumerate(batch["case_id"]):
        z, y, x = (int(value) for value in points[index, 0].tolist())
        rows.append(
            {
                "case_id": str(case_id),
                "subject_id": str(batch["subject_id"][index]),
                "locator_dsc": float(dsc[index]),
                "automatic_point": [z, y, x],
                "automatic_point_hits_gt": bool(label[index, 0, z, y, x] > 0.5),
            }
        )
    return rows


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    identity_rows: list[dict[str, Any]] = []
    flip8_rows: list[dict[str, Any]] = []
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
        identity_logits, flip8_logits = inner.flip8_logits(model, image)
        identity_rows.extend(_rows_from_logits(identity_logits, label, batch))
        flip8_rows.extend(_rows_from_logits(flip8_logits, label, batch))
    return {"identity": summarize_cases(identity_rows), "flip8": summarize_cases(flip8_rows)}


def _require_path_and_hash(path: Path, frozen: dict[str, Any], label: str) -> None:
    if str(path.resolve()) != str(frozen["path"]):
        raise ValueError(f"{label} path differs from pretest receipt")
    if sha256_file(path.resolve()) != str(frozen["sha256"]):
        raise ValueError(f"{label} SHA-256 differs from pretest receipt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretest", type=Path, required=True)
    parser.add_argument("--expected-pretest-sha256", required=True)
    parser.add_argument("--stage-a-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    output = args.output.resolve()
    started = output.with_suffix(output.suffix + ".started.json")
    if output.exists() or started.exists():
        raise FileExistsError("refusing repeated test270 evaluation")
    pretest_path = args.pretest.resolve()
    if sha256_file(pretest_path) != args.expected_pretest_sha256:
        raise ValueError("pretest receipt SHA-256 mismatch")
    receipt = json.loads(pretest_path.read_text())
    if (
        receipt.get("schema") != freeze.SCHEMA
        or receipt.get("status") != "frozen_after_stage_a_pass_before_test270_access"
        or receipt.get("one_run_limit") is not True
        or receipt.get("access", {}).get("test270") is not False
    ):
        raise ValueError("invalid pretest receipt")
    if receipt.get("evaluator", {}).get("sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("evaluator changed after pretest freeze")
    if str(output) != str(receipt.get("test_output")):
        raise ValueError("test output differs from pretest receipt")
    if receipt.get("tta") != {
        "axes": [list(axes) for axes in inner.REFLECTION_AXES],
        "aggregation": "ordered float32 mean of inverse-reflected raw logits",
        "forwards_per_case": 8,
        "identity_is_first_atom": True,
        "batch_size": 8,
        "threshold": 0.0,
        "point": "argmax mean raw logit",
        "sam": False,
        "postprocessing": False,
    }:
        raise ValueError("TTA contract changed after pretest freeze")

    stage_a_path = args.stage_a_report.resolve()
    if str(stage_a_path) != receipt["stage_a"]["report_path"] or sha256_file(stage_a_path) != receipt["stage_a"]["report_sha256"]:
        raise ValueError("Stage A report differs from pretest receipt")
    stage_a = json.loads(stage_a_path.read_text())
    freeze.validate_stage_a(stage_a)
    _require_path_and_hash(args.checkpoint, receipt["checkpoint"], "checkpoint")
    _require_path_and_hash(args.identity_report, receipt["archived_identity_report"], "identity report")
    if str(args.test_manifest.resolve()) != receipt["test270"]["manifest_path"] or sha256_file(args.test_manifest.resolve()) != TEST_MANIFEST_SHA256:
        raise ValueError("test manifest differs from pretest receipt")
    cache = args.cache.resolve()
    if str(cache) != receipt["test270"]["cache_path"] or sha256_file(cache / "index.json") != receipt["test270"]["cache_index_sha256"]:
        raise ValueError("test cache index differs from pretest receipt")

    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != full972.CHECKPOINT_SCHEMA or checkpoint.get("arm") != "full972_candidate" or checkpoint.get("epoch") != 168:
        raise ValueError("Full972 checkpoint contract mismatch")
    if checkpoint.get("model_state_sha256") != full972.state_dict_sha256(checkpoint["model_state_dict"]):
        raise ValueError("Full972 model-state SHA mismatch")
    for key in ("validation_access", "outer_selector_access", "confirmation_access", "test_access"):
        if checkpoint.get(key) is not False:
            raise ValueError("checkpoint crossed an evaluation boundary before freezing")

    write_json_atomic(
        started,
        {
            "schema": SCHEMA + ".started",
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "pretest_receipt_sha256": args.expected_pretest_sha256,
            "evidence_label": receipt["evidence_label"],
            "identity_parity_required_before_candidate_publication": True,
            "test270_access": True,
        },
    )

    rows = [json.loads(line) for line in args.test_manifest.read_text().splitlines() if line.strip()]
    case_ids = [str(row["case_id"]) for row in rows]
    if len(rows) != TEST_CASES or len(set(case_ids)) != TEST_CASES or len({str(row["subject_id"]) for row in rows}) != TEST_SUBJECTS:
        raise ValueError("test270 cohort mismatch")
    data = TestBrainCropDataset(cache, case_ids)
    if data.index.get("manifest_sha256") != TEST_MANIFEST_SHA256:
        raise ValueError("test cache manifest binding mismatch")
    for filename, field in (("images.f32", "images_sha256"), ("labels.u8", "labels_sha256")):
        if sha256_file(cache / filename) != str(data.index[field]):
            raise ValueError(f"test cache tensor SHA mismatch: {filename}")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("test270 evaluation requires CUDA")
    torch.cuda.set_device(device)
    architecture = checkpoint["architecture"]
    model = AutomaticPromptLocator(
        center_aware=False,
        init_filters=int(architecture["init_filters"]),
        dropout=float(architecture["dropout"]),
    ).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    loader = DataLoader(
        data,
        batch_size=8,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    summaries = evaluate(model, loader, device)
    archived = json.loads(args.identity_report.read_text())["candidate"]
    assert_identity_parity(summaries["identity"], archived)
    comparison = inner.compare(summaries["identity"], summaries["flip8"], freeze.PAIRED_BOOTSTRAP_SEED)
    gates = success_gates(summaries["identity"], summaries["flip8"])
    gates["all_passed"] = all(gates.values())
    identity_hits = sum(bool(row["automatic_point_hits_gt"]) for row in summaries["identity"]["per_case"])
    flip8_hits = sum(bool(row["automatic_point_hits_gt"]) for row in summaries["flip8"]["per_case"])
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "evidence_scope": "post-test development; reused internal test; not pristine blind; not external",
        "not_pristine_blind_test": True,
        "not_external_confirmation": True,
        "test_tuned": False,
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": freeze.CHECKPOINT_SHA256, "epoch": 168},
        "data": {
            "manifest": str(args.test_manifest.resolve()),
            "manifest_sha256": TEST_MANIFEST_SHA256,
            "cache": str(cache),
            "cache_index_sha256": sha256_file(cache / "index.json"),
            "cache_images_sha256": data.index["images_sha256"],
            "cache_labels_sha256": data.index["labels_sha256"],
            "cases": TEST_CASES,
            "subjects": TEST_SUBJECTS,
        },
        "tta": receipt["tta"],
        "identity": summaries["identity"],
        "flip8": summaries["flip8"],
        "identity_bootstrap": intervals(summaries["identity"]),
        "flip8_bootstrap": intervals(summaries["flip8"]),
        "flip8_minus_identity": comparison,
        "identity_parity": {
            "passed": True,
            "archived_report_path": str(args.identity_report.resolve()),
            "archived_report_sha256": freeze.IDENTITY_REPORT_SHA256,
            "scope": "exact full summary and per-case equality",
        },
        "point_hits": {"identity": identity_hits, "flip8": flip8_hits, "delta": flip8_hits - identity_hits},
        "gates": gates,
        "leader": {"identifier": "gcsam_inner_fit864_epoch184", "scan_dice": freeze.CURRENT_SCAN_LEADER},
        "leaderboard_refreshed": gates["all_passed"],
        "pretest_receipt": {"path": str(pretest_path), "sha256": args.expected_pretest_sha256},
        "stage_a_report": {"path": str(stage_a_path), "sha256": receipt["stage_a"]["report_sha256"]},
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "result_must_not_trigger_view_weight_threshold_checkpoint_or_postprocessing_tuning": True,
        "access": {
            "fit": True,
            "inner_metrics": True,
            "fixed_validation108": False,
            "outer": False,
            "confirmation": False,
            "test270": True,
            "ucsd_external75": False,
        },
    }
    write_json_atomic(output, report)
    print(
        json.dumps(
            {
                "identity": {
                    "scan": summaries["identity"]["locator_dsc_mean"],
                    "subject": summaries["identity"]["locator_subject_dsc_mean"],
                    "point_hits": identity_hits,
                },
                "flip8": {
                    "scan": summaries["flip8"]["locator_dsc_mean"],
                    "subject": summaries["flip8"]["locator_subject_dsc_mean"],
                    "point_hits": flip8_hits,
                },
                "comparison": comparison,
                "gates": gates,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
