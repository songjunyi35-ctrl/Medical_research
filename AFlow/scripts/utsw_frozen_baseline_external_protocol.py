"""Contract for one-time UTSW362 frozen-baseline cross-stage evaluation."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from scripts import no_n4_clean_protocol as common
from scripts import utsw_no_n4_paired_external_protocol as source_contract


ROOT = common.ROOT
RECORD = ROOT / "archive/research_records/2026-09-05_utsw_frozen_baseline_external"
SOURCE = Path("/tmp/aflow-utsw-frozen-baseline-source-v2-20260905")
CACHE = Path("/tmp/aflow-utsw-frozen-baseline-cache-20260905")
RESULT_WORK = ROOT / "workspace/BraTS/utsw_frozen_baseline_external_20260905"
PREACCESS = RECORD / "EXTERNAL_PREACCESS.json"
SOURCE_DOWNLOAD = RECORD / "SOURCE_DOWNLOAD_AMENDMENT_01.json"
DEADLINE = datetime.fromisoformat("2026-09-06T14:02:39+00:00")
BOOTSTRAP_SEED = 20261391
BOOTSTRAP_SAMPLES = 10_000
AXES = source_contract.AXES
CASES = source_contract.CASES
SOURCE_FILES = source_contract.SOURCE_FILES
SOURCE_BYTES = source_contract.SOURCE_BYTES
MODEL_SHAPE = source_contract.MODEL_SHAPE
IDEA_SHA = "841b4e51c5bb779210ec7ee25d897396fc03d2aaf40f5c5cfd5655abb5123441"
PREREG_SHA = "86036d81ec913757074911dad7fa1007b8fdee41ccf8915cd9465cc0de81600b"
MODELS = {
    "full972_kd168": {
        "path": ROOT / "workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824/final_formal_full972_candidate/plain_fulltrajectory_full972_candidate_epoch168.pth",
        "file_sha256": "f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b",
        "state_sha256": "f24201eecc7bf4aaff92b8e0a50c85a31e6913f5c9701cca6ad7df3439618ea7",
        "schema": "aflow.sammed3d.t1ce_wt_voxel_kd_fulltrajectory_final_checkpoint",
        "epoch": 168,
        "primary_view": "flip8",
    },
    "gcsam184": {
        "path": ROOT / "workspace/BraTS/sammed3d_t1ce_wt/gcsam_full972_stability_20260831/inner_formal_gcsam_candidate/post100_best.pth",
        "file_sha256": "acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d",
        "state_sha256": "eea33e1d3a1ab9505ba0b85aa915a9df59260b50a0991b49e1e9c77417690f02",
        "schema": "aflow.sammed3d.t1ce_wt_gcsam_inner.checkpoint",
        "epoch": 184,
        "primary_view": "identity",
    },
}

sha = common.sha
checked = common.checked
write_new = common.write_new


def docs() -> None:
    checked(RECORD / "IDEA.md", IDEA_SHA)
    checked(RECORD / "PREREG.md", PREREG_SHA)
    source_contract.cohort()
    checked(source_contract.CANONICAL_BRIDGE, source_contract.CANONICAL_BRIDGE_SHA)
    checked(source_contract.CANONICAL_BRIDGE_COPY_A, source_contract.CANONICAL_BRIDGE_COPY_A_SHA)
    for model in MODELS.values():
        checked(model["path"], model["file_sha256"])


def cohort() -> list[dict]:
    return source_contract.cohort()


def three_hour_gate(now: datetime | None = None) -> bool:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        raise ValueError("timezone-aware time required")
    return (DEADLINE - value).total_seconds() >= 3 * 3600


def verify_preaccess() -> dict:
    docs()
    value = json.loads(PREACCESS.read_text())
    expected = {
        name: {
            "path": str(row["path"]), "file_sha256": row["file_sha256"],
            "state_sha256": row["state_sha256"], "schema": row["schema"],
            "epoch": row["epoch"], "primary_view": row["primary_view"],
        }
        for name, row in MODELS.items()
    }
    if value.get("status") != "frozen_before_utsw_source_and_gpu_model_access" or value.get("models") != expected:
        raise ValueError("UTSW baseline preaccess model binding mismatch")
    if any(value.get("access", {}).get(key) is not False for key in (
        "new_utsw_subjects", "model_checkpoint_loaded_on_gpu", "model_forwards", "external_metrics"
    )):
        raise ValueError("UTSW baseline preaccess access flags not sealed")
    return value
