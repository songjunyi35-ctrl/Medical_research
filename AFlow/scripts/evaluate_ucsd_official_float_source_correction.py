#!/usr/bin/env python3
"""One-time source-correction evaluation on direct-TCIA UCSD75 float images."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_cached import CACHE_SCHEMA, MODEL_SHAPE  # noqa: E402
import scripts.evaluate_ucsd_ptgbm_external_confirmation as historical_eval  # noqa: E402
from scripts.prepare_ucsd_ptgbm_external_confirmation import sha256_file, write_json_atomic  # noqa: E402
from scripts.build_ucsd_official_float_source_correction_cache import SCHEMA as CACHE_EXTERNAL_SCHEMA  # noqa: E402


SCHEMA = "aflow.ucsd75_official_float_source_correction.evaluation"
OLD_REPORT_SHA256 = "91910760fbab804ee17b3ad8027a035a6676b02307a6f34d8408ba858c42f593"
FULL972_SHA256 = "f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b"
GCSAM184_SHA256 = "acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d"
EXPECTED_CASES = 75
EXPECTED_SUBJECTS = 53
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 2026090201


class SourceCorrectionEvaluationError(RuntimeError):
    pass


class OfficialFloatCache(Dataset):
    def __init__(self, cache: Path, expected_index_sha256: str) -> None:
        self.cache = cache.resolve()
        index_path = self.cache / "index.json"
        if sha256_file(index_path) != expected_index_sha256:
            raise SourceCorrectionEvaluationError("official-float cache index SHA-256 mismatch")
        self.index = json.loads(index_path.read_text())
        expected = {
            "schema": CACHE_SCHEMA,
            "external_schema": CACHE_EXTERNAL_SCHEMA,
            "source_version": "direct_TCIA_package_1019_float32",
            "source_correction_reuses_old_ucsd75": True,
            "independent_new_cohort": False,
            "count": EXPECTED_CASES,
            "subjects": EXPECTED_SUBJECTS,
            "split": "external_reused_source_correction",
            "external_confirmation": False,
            "not_pristine_blind": True,
        }
        for key, value in expected.items():
            if self.index.get(key) != value:
                raise SourceCorrectionEvaluationError(f"cache contract mismatch: {key}")
        required_preprocessing = {
            "spatial": "braincrop_resize",
            "output_resolution": 128,
            "uses_gt_for_localization": False,
            "n4_bias_field_correction": True,
            "n4_shrink_factor": 2,
        }
        for key, value in required_preprocessing.items():
            if self.index.get("preprocessing", {}).get(key) != value:
                raise SourceCorrectionEvaluationError(f"preprocessing mismatch: {key}")
        if self.index.get("target") != {"name": "WT", "foreground_raw_labels": [1, 2, 3], "excluded_raw_labels": [4]}:
            raise SourceCorrectionEvaluationError("target contract mismatch")
        if not self.index.get("qc", {}).get("first_case_reproduction", {}).get("image_byte_exact"):
            raise SourceCorrectionEvaluationError("first-case image reproduction gate failed")
        if not self.index.get("qc", {}).get("first_case_reproduction", {}).get("label_byte_exact"):
            raise SourceCorrectionEvaluationError("first-case label reproduction gate failed")
        if not self.index.get("qc", {}).get("official_source_labels_exact_to_historical_mirror"):
            raise SourceCorrectionEvaluationError("source label exactness gate failed")
        if sha256_file(self.cache / "images.f32") != self.index.get("images_sha256"):
            raise SourceCorrectionEvaluationError("image tensor SHA-256 mismatch")
        if sha256_file(self.cache / "labels.u8") != self.index.get("labels_sha256"):
            raise SourceCorrectionEvaluationError("label tensor SHA-256 mismatch")
        self.case_ids = [str(x) for x in self.index["case_ids"]]
        self.subject_ids = [str(x) for x in self.index["subject_ids"]]
        if len(set(self.case_ids)) != EXPECTED_CASES or len(set(self.subject_ids)) != EXPECTED_SUBJECTS:
            raise SourceCorrectionEvaluationError("cache identifiers are not unique/expected")
        self._images: np.memmap | None = None
        self._labels: np.memmap | None = None

    def _open(self) -> None:
        if self._images is None:
            self._images = np.memmap(self.cache / "images.f32", dtype=np.float32, mode="r", shape=(EXPECTED_CASES, *MODEL_SHAPE))
            self._labels = np.memmap(self.cache / "labels.u8", dtype=np.uint8, mode="r", shape=(EXPECTED_CASES, *MODEL_SHAPE))

    def __len__(self) -> int:
        return EXPECTED_CASES

    def __getitem__(self, slot: int) -> dict[str, Any]:
        self._open()
        assert self._images is not None and self._labels is not None
        return {
            "image": torch.from_numpy(np.array(self._images[slot], dtype=np.float32))[None],
            "label": torch.from_numpy(np.array(self._labels[slot], dtype=np.float32))[None],
            "case_id": self.case_ids[slot],
            "subject_id": self.subject_ids[slot],
        }


def _bootstrap(values: list[float], seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
    return [float(x) for x in np.quantile(array[draws].mean(axis=1), (0.025, 0.975))]


def compare_source_version(official: dict[str, Any], mirror: dict[str, Any], seed: int) -> dict[str, Any]:
    """Compare one frozen model on official-float versus old mirror-uint8."""
    official_cases = official["per_case"]
    mirror_cases = mirror["per_case"]
    if [x["case_id"] for x in official_cases] != [x["case_id"] for x in mirror_cases]:
        raise SourceCorrectionEvaluationError("official/mirror case order mismatch")
    scan_delta: list[float] = []
    subject_delta: dict[str, list[float]] = defaultdict(list)
    per_case: list[dict[str, Any]] = []
    for new, old in zip(official_cases, mirror_cases, strict=True):
        if new["subject_id"] != old["subject_id"]:
            raise SourceCorrectionEvaluationError("official/mirror subject mapping mismatch")
        delta = float(new["locator_dsc"]) - float(old["locator_dsc"])
        scan_delta.append(delta)
        subject_delta[str(new["subject_id"])].append(delta)
        per_case.append({
            "case_id": new["case_id"],
            "subject_id": new["subject_id"],
            "official_float_dsc": float(new["locator_dsc"]),
            "mirror_uint8_dsc": float(old["locator_dsc"]),
            "official_minus_mirror_dsc": delta,
            "official_point_hit": new["automatic_point_hits_gt"],
            "mirror_point_hit": old["automatic_point_hits_gt"],
        })
    subject_values = [statistics.fmean(values) for values in subject_delta.values()]
    point_delta = int(official["automatic_point_hits"]) - int(mirror["automatic_point_hits"])
    return {
        "official_float_minus_mirror_uint8": {
            "scan_dsc_mean": statistics.fmean(scan_delta),
            "subject_dsc_mean": statistics.fmean(subject_values),
            "positive_point_hits": point_delta,
        },
        "paired_scan_dsc_mean_95ci": _bootstrap(scan_delta, seed),
        "paired_subject_dsc_mean_95ci": _bootstrap(subject_values, seed + 1),
        "scan_cases_improved": sum(x > 0 for x in scan_delta),
        "scan_cases_worse": sum(x < 0 for x in scan_delta),
        "scan_cases_tied": sum(x == 0 for x in scan_delta),
        "subjects_improved": sum(x > 0 for x in subject_values),
        "subjects_worse": sum(x < 0 for x in subject_values),
        "subjects_tied": sum(x == 0 for x in subject_values),
        "per_case": per_case,
    }


def _validate_old_report(path: Path) -> dict[str, Any]:
    if sha256_file(path.resolve()) != OLD_REPORT_SHA256:
        raise SourceCorrectionEvaluationError("historical mirror report SHA-256 mismatch")
    report = json.loads(path.read_text())
    expected = {
        "primary": ("full972_random_init_voxel_kd_epoch168", 0.7455541275938352, 0.739054558458943, 70),
        "secondary": ("gcsam_fit864_inner_selected_epoch184", 0.7556473382314046, 0.7551516704964187, 71),
    }
    for role, (identifier, scan, subject, point) in expected.items():
        if report[role]["identifier"] != identifier:
            raise SourceCorrectionEvaluationError(f"historical {role} identifier mismatch")
        summary = report[role]["summary"]
        values = (float(summary["locator_dsc_mean"]), float(summary["locator_subject_dsc_mean"]), int(summary["automatic_point_hits"]))
        if values != (scan, subject, point):
            raise SourceCorrectionEvaluationError(f"historical {role} metrics mismatch")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--expected-cache-index-sha256", required=True)
    parser.add_argument("--old-report", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--preaccess-receipt", type=Path, required=True)
    parser.add_argument("--expected-preaccess-receipt-sha256", required=True)
    parser.add_argument("--full972-checkpoint", type=Path, required=True)
    parser.add_argument("--gcsam184-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    output = args.output.resolve()
    started_path = output.with_suffix(output.suffix + ".started.json")
    if output.exists() or started_path.exists():
        raise FileExistsError("refusing repeated UCSD75 source-correction evaluation")
    if sha256_file(args.preregistration.resolve()) != args.expected_preregistration_sha256:
        raise SourceCorrectionEvaluationError("preregistration SHA-256 mismatch")
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("status") != "frozen_before_official_float_cache_and_model_inference":
        raise SourceCorrectionEvaluationError("preregistration is not frozen")
    if sha256_file(args.preaccess_receipt.resolve()) != args.expected_preaccess_receipt_sha256:
        raise SourceCorrectionEvaluationError("preaccess receipt SHA-256 mismatch")
    preaccess = json.loads(args.preaccess_receipt.read_text())
    if preaccess.get("status") != "frozen_after_cache_before_model_inference" or preaccess.get("access", {}).get("ucsd75_official_float_model_inference") is not False:
        raise SourceCorrectionEvaluationError("preaccess receipt contract mismatch")
    old_report = _validate_old_report(args.old_report.resolve())
    data = OfficialFloatCache(args.cache, args.expected_cache_index_sha256)
    if data.case_ids != [x["case_id"] for x in old_report["primary"]["summary"]["per_case"]]:
        raise SourceCorrectionEvaluationError("cache case order differs from historical report")
    if data.subject_ids != [x["subject_id"] for x in old_report["primary"]["summary"]["per_case"]]:
        raise SourceCorrectionEvaluationError("cache subject order differs from historical report")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SourceCorrectionEvaluationError("formal source correction requires CUDA")
    torch.cuda.set_device(device)
    loader = DataLoader(data, batch_size=min(args.batch_size, len(data)), shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    full_model, full_payload = historical_eval.load_model(args.full972_checkpoint.resolve(), FULL972_SHA256, "primary", device)
    gcsam_model, gcsam_payload = historical_eval.load_model(args.gcsam184_checkpoint.resolve(), GCSAM184_SHA256, "secondary", device)
    write_json_atomic(started_path, {
        "schema": SCHEMA + ".started",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_index_sha256": args.expected_cache_index_sha256,
        "preregistration_sha256": args.expected_preregistration_sha256,
        "preaccess_receipt_sha256": args.expected_preaccess_receipt_sha256,
        "old_mirror_report_sha256": OLD_REPORT_SHA256,
        "full972_checkpoint_sha256": FULL972_SHA256,
        "gcsam184_checkpoint_sha256": GCSAM184_SHA256,
        "threshold": 0.0,
        "views": ["identity"],
        "reused_ucsd75_source_correction_access": True,
    })

    full = historical_eval.evaluate_model(full_model, loader, device)
    del full_model
    torch.cuda.empty_cache()
    gcsam = historical_eval.evaluate_model(gcsam_model, loader, device)
    del gcsam_model
    model_comparison = historical_eval.compare(full, gcsam)
    full_source_delta = compare_source_version(full, old_report["primary"]["summary"], BOOTSTRAP_SEED + 100)
    gcsam_source_delta = compare_source_version(gcsam, old_report["secondary"]["summary"], BOOTSTRAP_SEED + 200)
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "reused UCSD75 source correction: same 75 scans/53 subjects and labels; direct TCIA package 1019 float32 images replace Pittsburgh Fiber Data Hub uint8 mirror; not a new independent cohort",
        "preregistered": True,
        "not_test_tuned": True,
        "not_pristine_blind": True,
        "external_confirmation": False,
        "reused_external_cohort": True,
        "independent_new_cohort": False,
        "protocol": {
            "single_factor": "image source bytes only: official float32 versus historical mirror uint8",
            "input": "T1 post-contrast only",
            "target": "WT={1,2,3}; raw label4 excluded",
            "threshold": 0.0,
            "forwards_per_model": 1,
            "views": ["identity"],
            "tta": False,
            "sam": False,
            "postprocessing": False,
            "ensemble": False,
            "bootstrap": {"method": "percentile", "samples": BOOTSTRAP_SAMPLES, "base_seed": BOOTSTRAP_SEED},
        },
        "data": {
            "cache": str(args.cache.resolve()),
            "cache_index_sha256": args.expected_cache_index_sha256,
            "images_sha256": data.index["images_sha256"],
            "labels_sha256": data.index["labels_sha256"],
            "cases": len(data),
            "subjects": len(set(data.subject_ids)),
            "official_source": "TCIA package 1019",
            "historical_mirror_report": str(args.old_report.resolve()),
            "historical_mirror_report_sha256": OLD_REPORT_SHA256,
        },
        "full972_kd168": {
            "identifier": "full972_random_init_voxel_kd_epoch168",
            "checkpoint": str(args.full972_checkpoint.resolve()),
            "checkpoint_sha256": FULL972_SHA256,
            "checkpoint_epoch": int(full_payload["epoch"]),
            "summary": full,
            "bootstrap": historical_eval.bootstrap_absolute(full, BOOTSTRAP_SEED),
            "source_delta": full_source_delta,
        },
        "gcsam184": {
            "identifier": "gcsam_fit864_inner_selected_epoch184",
            "checkpoint": str(args.gcsam184_checkpoint.resolve()),
            "checkpoint_sha256": GCSAM184_SHA256,
            "checkpoint_epoch": int(gcsam_payload["epoch"]),
            "summary": gcsam,
            "bootstrap": historical_eval.bootstrap_absolute(gcsam, BOOTSTRAP_SEED + 1),
            "source_delta": gcsam_source_delta,
        },
        "official_float_model_comparison_full972_minus_gcsam": model_comparison,
        "data_engineering_gates": {"all_passed_before_inference": True},
        "performance_promotion_gate": None,
        "result_must_not_trigger_tuning_promotion_or_repeat": True,
        "access": {"train": False, "fixed_validation108": False, "outer": False, "confirmation": False, "internal_test270": False, "ucsd75_reused_source_correction": True},
        "evaluator": str(Path(__file__).resolve().relative_to(ROOT)),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_atomic(output, report)
    print(json.dumps({
        "full972_kd168": {"scan": full["locator_dsc_mean"], "subject": full["locator_subject_dsc_mean"], "point": full["automatic_point_hits"], "source_delta": full_source_delta["official_float_minus_mirror_uint8"]},
        "gcsam184": {"scan": gcsam["locator_dsc_mean"], "subject": gcsam["locator_subject_dsc_mean"], "point": gcsam["automatic_point_hits"], "source_delta": gcsam_source_delta["official_float_minus_mirror_uint8"]},
        "full972_minus_gcsam": model_comparison["primary_minus_secondary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
