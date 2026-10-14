"""Freeze/verify the conditional UTSW362 paired external implementation."""
from __future__ import annotations

from datetime import datetime, timezone
import json

from scripts import utsw_no_n4_paired_external_protocol as p


FILES = (
    "scripts/utsw_no_n4_paired_external_protocol.py",
    "scripts/prepare_utsw_no_n4_paired_external.py",
    "scripts/smoke_utsw_no_n4_paired_external.py",
    "scripts/download_utsw_no_n4_paired_external.py",
    "scripts/build_utsw_no_n4_paired_external_caches.py",
    "scripts/evaluate_utsw_no_n4_paired_external.py",
    "scripts/queue_utsw_no_n4_paired_external.py",
    "scripts/freeze_utsw_no_n4_paired_external.py",
    "scripts/build_no_n4_clean_cache.py",
    "scripts/build_t1ce_wt_cache.py",
    "scripts/audit_utsw_primary_fets_no_n4_input_bridge.py",
    "scripts/train_t1ce_wt_gcsam_inner.py",
    "scripts/n4_family_full972_protocol.py",
    "sammed3d/t1ce_binary.py",
    "tests/test_utsw_no_n4_paired_external_protocol.py",
    "tests/test_utsw_no_n4_paired_external_download.py",
    "tests/test_utsw_no_n4_paired_external_cache.py",
    "tests/test_utsw_no_n4_paired_external_evaluation.py",
    "tests/test_utsw_no_n4_paired_external_queue.py",
)
RECEIPT = p.RECORD / "EXTERNAL_IMPLEMENTATION_AMENDMENT_04.json"


def current() -> dict[str, str]:
    return {name: p.sha(p.ROOT / name) for name in FILES}


def check_implementation() -> dict:
    p.docs()
    value = json.loads(RECEIPT.read_text())
    if value.get("files") != current():
        raise ValueError("UTSW external implementation hash mismatch")
    return value


def main() -> None:
    p.docs()
    value = {
        "schema": "aflow.utsw_no_n4_paired_external.implementation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_checkpoint_eligibility_and_new_utsw_access",
        "files": current(),
        "cohort": {"cases": p.CASES, "subjects": p.CASES, "source_files": p.SOURCE_FILES, "source_bytes_per_copy": p.SOURCE_BYTES},
        "arms": ["canonical", "no_n4"], "views": ["identity", "Flip8"],
        "threshold": 0.0, "forwards": 6516,
        "bootstrap": {"samples": p.BOOTSTRAP_SAMPLES, "seed": p.BOOTSTRAP_SEED},
        "automatic_old_cohort_access": False,
    }
    p.write_new(RECEIPT, value)
    check_implementation()
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
