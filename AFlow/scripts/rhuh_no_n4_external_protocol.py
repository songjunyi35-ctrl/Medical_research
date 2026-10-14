"""Fail-closed contract for conditional one-time RHUH no-N4 confirmation."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from scripts import n4_family_full972_protocol as full

ROOT = full.ROOT
RECORD = ROOT / "archive/research_records/2026-09-05_rhuh_no_n4_input_bridge"
REVIEW = ROOT / "archive/reviews/2026-09-05_30h_chain_audit"
SOURCE_WORK = ROOT / "workspace/BraTS/rhuh_gbm_source_20260905"
RESULT_WORK = ROOT / "workspace/BraTS/rhuh_no_n4_external_20260905"
CACHE = Path("/tmp/aflow-rhuh-no-n4-external-20260905")
PREACCESS = RECORD / "EXTERNAL_PREACCESS.json"
DEADLINE = full.DEADLINE
IDEA_SHA = "5f5100b218ed8770106c712f93a3fabae994f31ffb7956e4acf2cffcbf2bf47c"
PREREG_SHA = "a94a46f278326655f5db18552e90ec533053cd3b6470d9ead58ec264faa8a36c"
AMENDMENT_SHA = "d5b7b3d86a223d0715a00bc0b5d11e2610e2b20bf9d61b42e4c1911407926908"
INPUT_QC_SHA = "0df16cddc00d301774e9e986380bf60dcf584311e00877d32365f4c5c2274245"
LABEL_INVENTORY_SHA = "92da8952b2bb76b90c22f41c1a0e6295fcb60115ba724badd4f6c8b52ab64e98"
SUMS_SHA = "7c6ba3fa767bf169679db30caa854e671796416ca365fac6e57662796177c130"
IMAGE_DOWNLOAD_SHA = "c6f17decbc81e8a2bc8ec385438b7e83547644aed26009fb376bd27a218d725f"
BOOTSTRAP_SEED = 20261361
BOOTSTRAP_SAMPLES = 10_000
AXES = ((), (-1,), (-2,), (-2, -1), (-3,), (-3, -1), (-3, -2), (-3, -2, -1))

sha = full.sha
checked = full.checked
write_new = full.write_new


def docs() -> None:
    checked(RECORD / "IDEA.md", IDEA_SHA)
    checked(RECORD / "PREREG.md", PREREG_SHA)
    checked(RECORD / "AMENDMENT_01.md", AMENDMENT_SHA)
    checked(RECORD / "INPUT_QC.json", INPUT_QC_SHA)
    checked(REVIEW / "RHUH_LABEL_INVENTORY_READINESS.json", LABEL_INVENTORY_SHA)
    checked(REVIEW / "source_metadata/RHUH-GBM-nii-v1.sums", SUMS_SHA)
    checked(REVIEW / "RHUH_IMAGE_DOWNLOAD.json", IMAGE_DOWNLOAD_SHA)


def eligible_checkpoint() -> tuple[Path, dict, dict]:
    """Recompute no-N4 Stage A eligibility and verify completed Full972 training."""
    docs()
    decisions = full.recompute_decisions()
    if decisions["no_n4"]["performance_passed"] is not True:
        raise RuntimeError("no-N4 did not pass every Stage A gate")
    done_path = full.RECORD / "FULL972_FORMAL_RUNNER_DONE.json"
    pairing_path = full.RECORD / "FULL972_FORMAL_PAIRING.json"
    if not done_path.is_file() or not pairing_path.is_file():
        raise RuntimeError("conditional Full972 no-N4 training is incomplete")
    done = json.loads(done_path.read_text())
    pairing = json.loads(pairing_path.read_text())
    if done.get("status") != "complete" or "no_n4" not in done.get("eligible_candidates", []):
        raise ValueError("Full972 runner did not complete eligible no-N4")
    comparison = pairing.get("comparisons", {}).get("no_n4", {})
    if comparison.get("all_gates_passed") is not True:
        raise ValueError("Full972 no-N4 paired-control gates failed")
    report_path = full.WORK / "formal/no_n4/TRAIN_DONE.json"
    report = json.loads(report_path.read_text())
    if report.get("all_gates_passed") is not True or report.get("role") != "no_n4":
        raise ValueError("wrong Full972 no-N4 training report")
    checkpoint = Path(report["checkpoint"]["path"])
    checked(checkpoint, report["checkpoint"]["sha256"])
    return checkpoint, report, {
        "stage_a_raw_sha256": decisions["no_n4"]["raw_sha256"],
        "full972_done_sha256": sha(done_path),
        "full972_pairing_sha256": sha(pairing_path),
        "full972_train_report_sha256": sha(report_path),
    }


def verify_preaccess() -> dict:
    docs()
    value = json.loads(PREACCESS.read_text())
    checkpoint, report, evidence = eligible_checkpoint()
    if value.get("status") != "frozen_before_rhuh_label_download":
        raise ValueError("wrong RHUH external preaccess status")
    if value.get("checkpoint", {}).get("path") != str(checkpoint):
        raise ValueError("RHUH preaccess checkpoint path mismatch")
    if value.get("checkpoint", {}).get("sha256") != report["checkpoint"]["sha256"]:
        raise ValueError("RHUH preaccess checkpoint hash mismatch")
    if value.get("eligibility_evidence") != evidence:
        raise ValueError("RHUH preaccess eligibility evidence mismatch")
    if any(value.get("access", {}).get(key) is not False for key in (
        "rhuh_labels_downloaded", "rhuh_label_voxels", "model_checkpoint_loaded",
        "model_forwards", "new_metrics"
    )):
        raise ValueError("RHUH preaccess access flags are not sealed")
    return value


def two_hour_gate(now: datetime | None = None) -> bool:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("timezone-aware time required")
    return (DEADLINE - current).total_seconds() >= 2 * 3600
