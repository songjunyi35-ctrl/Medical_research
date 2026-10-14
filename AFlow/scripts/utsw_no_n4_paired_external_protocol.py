"""Fail-closed contract for conditional UTSW362 paired external evaluation."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts import n4_family_full972_protocol as full


ROOT = full.ROOT
RECORD = ROOT / "archive/research_records/2026-09-05_utsw_no_n4_paired_external"
SOURCE_REVIEW = ROOT / "archive/reviews/2026-09-05_utsw_glioma_source_audit"
METADATA = SOURCE_REVIEW / "source_metadata/UTSW_Glioma_Metadata-2-1.tsv"
INVENTORY = SOURCE_REVIEW / "OFFICIAL_INVENTORY_AMENDMENT_01.json"
NO_N4_BRIDGE = ROOT / "archive/research_records/2026-09-05_utsw_primary_fets_no_n4_input_bridge/INPUT_QC.json"
CANONICAL_BRIDGE = ROOT / "archive/research_records/2026-09-05_utsw_primary_fets_canonical_n4_input_bridge/INPUT_QC.json"
CANONICAL_BRIDGE_COPY_A = ROOT / "archive/research_records/2026-09-05_utsw_primary_fets_canonical_n4_input_bridge/COPY_A_QC.json"
SOURCE = Path("/tmp/aflow-utsw-paired-source-20260905")
CACHE_ROOT = Path("/tmp/aflow-utsw-paired-cache-20260905")
RESULT_WORK = ROOT / "workspace/BraTS/utsw_no_n4_paired_external_20260905"
PREACCESS = RECORD / "EXTERNAL_PREACCESS.json"
PRECEDENCE = RECORD / "RHUH_PRECEDENCE.json"
DEADLINE = full.DEADLINE
BOOTSTRAP_SEED = 20261381
BOOTSTRAP_SAMPLES = 10_000
AXES = ((), (-1,), (-2,), (-2, -1), (-3,), (-3, -1), (-3, -2), (-3, -2, -1))
IDEA_SHA = "32f9fc3e6f6aa26ad792059de6cf35b282ae4a5a29705cfa1eeee4cea22db818"
PREREG_SHA = "11e51dca5b5438a14f046d7d8c1532b2d65d36284607090df8fa59aaf50b2ea8"
METADATA_SHA = "f7ff889fa7d197b2f589695d3e7ea8a2d1d198c4f9c0044fc71898a08aa24ed0"
INVENTORY_SHA = "5f210c0a1e0c33f7b71a56d9bbce8b52a62f24e2a0b53655aa3a098dbde3f871"
NO_N4_BRIDGE_SHA = "7573bed6d03f194a429d7eacb4ffb3b711feeda15e91a47539aaf025020deb59"
CANONICAL_BRIDGE_SHA = "1cec4ba98fae05e54caca2a1eb0a14681758e79ed6ac35ec47b0a28a58353be0"
CANONICAL_BRIDGE_COPY_A_SHA = "a1776d09d7e84a786e3fc27f114ec4119a6f02bf1cdb35102906a3e8876a3665"
MANUAL_IDS_SHA = "e4809dd7f43439309bef71784b3accff95be1a6915e044f46de352e780ec525c"
SOURCE_FILES = 724
SOURCE_BYTES = 1_813_086_599
CASES = 362
MODEL_SHAPE = (128, 128, 128)

sha = full.sha
checked = full.checked
write_new = full.write_new


def docs() -> None:
    checked(RECORD / "IDEA.md", IDEA_SHA)
    checked(RECORD / "PREREG.md", PREREG_SHA)
    checked(METADATA, METADATA_SHA)
    checked(INVENTORY, INVENTORY_SHA)
    checked(NO_N4_BRIDGE, NO_N4_BRIDGE_SHA)
    checked(CANONICAL_BRIDGE, CANONICAL_BRIDGE_SHA)
    checked(CANONICAL_BRIDGE_COPY_A, CANONICAL_BRIDGE_COPY_A_SHA)


def cohort() -> list[dict]:
    docs()
    with METADATA.open(newline="") as handle:
        metadata = list(csv.DictReader(handle, delimiter="\t"))
    manual_ids = sorted(row["Subject ID"].strip() for row in metadata if row["Manually Refined Segmentation"].strip() == "Yes")
    if len(metadata) != 625 or len(manual_ids) != CASES:
        raise ValueError("UTSW metadata/manual cohort mismatch")
    import hashlib
    digest = hashlib.sha256(("\n".join(manual_ids) + "\n").encode()).hexdigest()
    if digest != MANUAL_IDS_SHA:
        raise ValueError("UTSW manual ID list changed")
    inventory = json.loads(INVENTORY.read_text())["files"]
    by_path = {item["path"]: item for item in inventory}
    rows = []
    for subject in manual_ids:
        image_path = f"/UTSW-Glioma/{subject}/brain_t1ce.nii.gz"
        label_path = f"/UTSW-Glioma/{subject}/rtumorseg_manual_correction.nii.gz"
        if image_path not in by_path or label_path not in by_path:
            raise ValueError(f"missing frozen UTSW pair: {subject}")
        rows.append({
            "subject_id": subject,
            "case_id": subject,
            "image_official_path": image_path,
            "image_bytes": int(by_path[image_path]["size"]),
            "label_official_path": label_path,
            "label_bytes": int(by_path[label_path]["size"]),
        })
    if len(rows) * 2 != SOURCE_FILES or sum(row["image_bytes"] + row["label_bytes"] for row in rows) != SOURCE_BYTES:
        raise ValueError("UTSW frozen source size/count mismatch")
    return rows


def four_hour_gate(now: datetime | None = None) -> bool:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        raise ValueError("timezone-aware time required")
    return (DEADLINE - value).total_seconds() >= 4 * 3600


def verify_precedence() -> dict:
    value = json.loads(PRECEDENCE.read_text())
    status = value.get("status")
    if status == "rhuh_metric_archived":
        path = Path(value["rhuh_report_path"])
        checked(path, value["rhuh_report_sha256"])
    elif status == "rhuh_closed_before_metric":
        if value.get("external_metrics_accessed") is not False or not value.get("failure_boundary"):
            raise ValueError("invalid RHUH pre-metric close evidence")
    else:
        raise ValueError("RHUH priority chain has no accepted terminal evidence")
    return value


def eligible_models() -> tuple[dict[str, dict], dict]:
    docs()
    decisions = full.recompute_decisions()
    if decisions["no_n4"]["performance_passed"] is not True:
        raise RuntimeError("no-N4 did not pass Stage A")
    done_path = full.RECORD / "FULL972_FORMAL_RUNNER_DONE.json"
    pairing_path = full.RECORD / "FULL972_FORMAL_PAIRING.json"
    if not done_path.is_file() or not pairing_path.is_file():
        raise RuntimeError("Full972 formal paired models are incomplete")
    done = json.loads(done_path.read_text())
    pairing = json.loads(pairing_path.read_text())
    if done.get("status") != "complete" or "no_n4" not in done.get("eligible_candidates", []):
        raise ValueError("Full972 no-N4 completion mismatch")
    if pairing.get("comparisons", {}).get("no_n4", {}).get("all_gates_passed") is not True:
        raise ValueError("Full972 canonical/no-N4 pairing failed")
    reports = {}
    for role in ("canonical", "no_n4"):
        report_path = full.WORK / f"formal/{role}/TRAIN_DONE.json"
        report = json.loads(report_path.read_text())
        if report.get("all_gates_passed") is not True or report.get("role") != role:
            raise ValueError(f"wrong Full972 report: {role}")
        history = report.get("history", [])
        if (
            report.get("seed") != 20261343
            or len(history) != 168
            or [item.get("epoch") for item in history] != list(range(1, 169))
        ):
            raise ValueError(f"wrong Full972 identity: {role}")
        checkpoint = Path(report["checkpoint"]["path"])
        checked(checkpoint, report["checkpoint"]["sha256"])
        reports[role] = {
            "report": report,
            "report_path": str(report_path),
            "report_sha256": sha(report_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": report["checkpoint"]["sha256"],
        }
    evidence = {
        "stage_a_no_n4_raw_sha256": decisions["no_n4"]["raw_sha256"],
        "full972_done_sha256": sha(done_path),
        "full972_pairing_sha256": sha(pairing_path),
    }
    return reports, evidence


def verify_preaccess() -> dict:
    value = json.loads(PREACCESS.read_text())
    reports, evidence = eligible_models()
    precedence = verify_precedence()
    if value.get("status") != "frozen_before_utsw_model_and_source_access":
        raise ValueError("wrong UTSW preaccess status")
    if value.get("models") != {
        role: {"checkpoint": item["checkpoint"], "checkpoint_sha256": item["checkpoint_sha256"], "report_sha256": item["report_sha256"]}
        for role, item in reports.items()
    }:
        raise ValueError("UTSW preaccess model binding mismatch")
    if value.get("eligibility_evidence") != evidence or value.get("rhuh_precedence_sha256") != sha(PRECEDENCE):
        raise ValueError("UTSW preaccess evidence mismatch")
    if precedence.get("status") not in ("rhuh_metric_archived", "rhuh_closed_before_metric"):
        raise ValueError("UTSW precedence status changed")
    if any(value.get("access", {}).get(key) is not False for key in (
        "new_utsw_subjects", "model_checkpoint_loaded", "model_forwards", "external_metrics"
    )):
        raise ValueError("UTSW preaccess access flags are not sealed")
    return value
