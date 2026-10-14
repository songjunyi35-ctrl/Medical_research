"""Fail-closed shared contract for the conditional N4-family Full972 stage."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from scripts import no_n4_clean_protocol as parent
from scripts import n4_otsu_protocol as otsu
from scripts.prepare_n4_family_full972_admission import DEADLINE, performance_passes, time_gate

ROOT = parent.ROOT
RECORD = parent.RECORD
OTSU_RECORD = otsu.RECORD
WORK = parent.WORK / "full972"
SOURCE = parent.SOURCE
NATIVE = parent.NATIVE
CONTROL_CACHE = parent.CONTROL_CACHE
SOURCE_SHA = parent.SOURCE_SHA
NATIVE_SHA = parent.NATIVE_SHA
CONTROL_INDEX_SHA = parent.INDEX_SHA
LABELS_SHA = parent.LABELS_SHA
PARAMETERS = parent.PARAMETERS
EPOCHS = parent.EPOCHS
STEPS = parent.STEPS
DRAWS = parent.DRAWS
ROLES = ("history", "ema", "canonical", "no_n4", "otsu")
CANDIDATES = ("no_n4", "otsu")
SEEDS = {"history": 20261341, "ema": 20261342,
         "canonical": 20261343, "no_n4": 20261343, "otsu": 20261343}

sha = parent.sha
checked = parent.checked
write_new = parent.write_new
read_rows = parent.read_rows


def docs() -> None:
    """Verify both already-preregistered idea families and their amendments."""
    parent.docs()
    otsu.docs()


def cache_for(role: str) -> tuple[Path, str]:
    if role in ("history", "ema", "canonical"):
        return CONTROL_CACHE, CONTROL_INDEX_SHA
    if role == "no_n4":
        receipt = json.loads((RECORD / "CACHE_RECEIPT.json").read_text())
    elif role == "otsu":
        receipt = json.loads((OTSU_RECORD / "CACHE_RECEIPT.json").read_text())
    else:
        raise ValueError("unknown Full972 role")
    if receipt.get("status") != "passed" or receipt.get("all972_labels_exact") is not True:
        raise ValueError(f"{role} cache receipt did not pass")
    return Path(receipt["candidate_cache"]), receipt["candidate_index_sha256"]


def verify_cache(role: str) -> tuple[Path, str]:
    cache, expected_sha = cache_for(role)
    checked(cache / "index.json", expected_sha)
    value = json.loads((cache / "index.json").read_text())
    if value.get("manifest_sha256") != SOURCE_SHA or value.get("count") != 972:
        raise ValueError("Full972 cache source/count mismatch")
    if value.get("split") != "train" or value.get("test_access") is not False:
        raise ValueError("Full972 cache role mismatch")
    if value.get("labels_sha256") != LABELS_SHA:
        raise ValueError("Full972 cache labels mismatch")
    prep = value.get("preprocessing", {})
    if role == "no_n4" and prep.get("n4_bias_field_correction") is not False:
        raise ValueError("no-N4 cache mismatch")
    if role == "otsu" and (
        prep.get("n4_bias_field_correction") is not True
        or prep.get("n4_fit_mask") != "OtsuThreshold(full_image,0,1,200)"
    ):
        raise ValueError("Otsu cache mismatch")
    if role in ("history", "ema", "canonical") and prep.get("n4_bias_field_correction") is not True:
        raise ValueError("canonical N4 cache mismatch")
    return cache, expected_sha


def recompute_decisions() -> dict[str, dict[str, object]]:
    """Recompute eligibility from archived raw rows, never trust a stored boolean alone."""
    family_path = RECORD / "FAMILY_READOUT_DONE.json"
    if not family_path.is_file():
        raise RuntimeError("family readout and archival are incomplete")
    family = json.loads(family_path.read_text())
    decisions: dict[str, dict[str, object]] = {}
    for method, directory in (("no_n4", RECORD), ("otsu", OTSU_RECORD)):
        raw_path = directory / "RAW_INNER.json"
        checked(raw_path, family[method + "_raw_sha256"])
        raw = json.loads(raw_path.read_text())
        decisions[method] = {
            "performance_passed": performance_passes(raw),
            "raw_sha256": sha(raw_path),
        }
    return decisions


def verify_admission(*, require_time: bool) -> tuple[list[str], dict]:
    """Verify the generated receipt and return eligible candidates in frozen order."""
    path = RECORD / "FULL972_ADMISSION_CHECK.json"
    if not path.is_file():
        raise RuntimeError("Full972 admission check has not been generated")
    value = json.loads(path.read_text())
    decisions = recompute_decisions()
    if value.get("decisions") != decisions:
        raise ValueError("admission decisions contradict raw reports")
    eligible = [name for name in CANDIDATES if decisions[name]["performance_passed"]]
    if value.get("eligible_methods_in_fixed_priority") != eligible:
        raise ValueError("admission priority/eligibility mismatch")
    if not eligible:
        raise RuntimeError("no N4-family candidate passed every Stage A gate")
    if value.get("old_cohort_evaluation_authorized") is not False:
        raise ValueError("admission must not authorize old-cohort evaluation")
    if value.get("automatic_training_launched") is not False:
        raise ValueError("admission receipt unexpectedly claims a launch")
    if require_time:
        now = datetime.now(timezone.utc)
        if value.get("eight_hour_time_gate") is not True or not time_gate(now):
            raise RuntimeError(f"Full972 eight-hour gate failed; deadline={DEADLINE.isoformat()}")
    return eligible, value
