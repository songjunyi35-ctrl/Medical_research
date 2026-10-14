import ast
import hashlib
import json
from pathlib import Path

from scripts import waw_gliomai152_protocol as p
from scripts.evaluate_waw_gliomai152_panel import interpretation


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_source_and_cohort_contract():
    p.docs()
    cohort = json.loads(p.COHORT.read_text())
    receipt = json.loads(p.SOURCE_RECEIPT.read_text())
    assert cohort["case_count"] == cohort["subject_count"] == p.CASES == 152
    assert len(cohort["cases"]) == 152
    assert len({row["case_id"] for row in cohort["cases"]}) == 152
    assert len({row["subject_id"] for row in cohort["cases"]}) == 152
    assert receipt["status"] == "passed"
    assert receipt["selected"]["global_raw_label_union"] == [0, 1, 2, 3]
    assert receipt["checkpoint_loaded"] is False and receipt["model_forwards"] == 0


def test_view_and_statistical_contract():
    assert len(p.AXES) == len(set(p.AXES)) == 8
    assert p.AXES[0] == ()
    assert p.BOOTSTRAP_SAMPLES == 10000
    assert p.BOOTSTRAP_SEED == 20261452
    assert p.FULL_SHA == "f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b"
    assert p.GCSAM_SHA == "acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d"


def test_entrypoints_are_isolated_and_do_not_import_old_cohort_data():
    forbidden = ("RAW_EXTERNAL95.json", "RAW_CPTAC33.json", "UCSD_EXTERNAL_CONFIRMATION.json", "RAW_TEST270.json")
    for name in ("build_waw_gliomai152_cache.py", "smoke_waw_gliomai152_panel.py", "evaluate_waw_gliomai152_panel.py"):
        path = ROOT / "scripts" / name
        ast.parse(path.read_text())
        text = path.read_text()
        assert not any(token in text for token in forbidden)


def test_cache_contract_when_present():
    if not p.CACHE_RECEIPT.exists():
        return
    receipt = json.loads(p.CACHE_RECEIPT.read_text())
    if not p.CACHE.exists():
        verification = json.loads((p.RECORD / "VERIFICATION_RECEIPT.json").read_text())
        archived_index = p.RECORD / "CACHE_INDEX_FINAL.json"
        assert verification["status"] == "passed"
        assert verification["cache_receipt_sha256"] == sha256(p.CACHE_RECEIPT)
        assert verification["cache_index_archive_sha256"] == sha256(archived_index)
        assert verification["cache_index_sha256"] == receipt["index_sha256"]
        assert "删除" in (p.RECORD / "RESULT.md").read_text()
        return
    index = json.loads((p.CACHE / "index.json").read_text())
    assert receipt["all_gates_passed"] is True
    assert index["cases"] == index["subjects"] == 152
    assert index["raw_union"] == [0, 1, 2, 3]
    assert sha256(p.CACHE / "index.json") == receipt["index_sha256"]
    assert sha256(p.CACHE / "images.f32") == receipt["images_sha256"]
    assert sha256(p.CACHE / "labels.u8") == receipt["labels_sha256"]


def test_raw_contract_when_present():
    raw_path = p.RECORD / "RAW_WAW152.json"
    if not raw_path.exists():
        return
    raw = json.loads(raw_path.read_text())
    assert raw["forwards"] == 152 * 9
    assert raw["result_must_not_trigger_tuning_or_repeat"] is True
    assert raw["access"]["waw_gliomai152"] is True
    assert all(value is False for key, value in raw["access"].items() if key != "waw_gliomai152")


def test_frozen_three_level_interpretation_gate():
    def comparison(delta, lower, upper):
        return {"candidate": {"scan_dice": delta}, "paired_scan_dice_95ci": [lower, upper]}

    assert interpretation(comparison(0.01, 0.001, 0.02), "candidate") == "supported_on_WAW"
    assert interpretation(comparison(0.01, -0.001, 0.02), "candidate") == "positive_but_uncertain"
    assert interpretation(comparison(0.0, -0.01, 0.01), "candidate") == "not_supported_or_reversed"
    assert interpretation(comparison(-0.01, -0.02, -0.001), "candidate") == "not_supported_or_reversed"
