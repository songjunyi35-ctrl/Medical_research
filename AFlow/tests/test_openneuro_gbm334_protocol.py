import json
import ast
from pathlib import Path

from scripts import openneuro_gbm334_protocol as p
from scripts.evaluate_openneuro_gbm334_panel import interpretation


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_inventory_contract():
    rows = p.cases()
    assert len(rows) == p.CASES == 334
    assert len({row["subject_id"] for row in rows}) == 334
    assert all("versionId=" in row[k]["url"] for row in rows for k in ("image", "label"))
    assert all(len(row[k]["sha256"]) == 64 for row in rows for k in ("image", "label"))


def test_pre_payload_exclusions_and_site_counts_are_frozen():
    inventory = json.loads(p.INVENTORY.read_text())
    assert {row["subject_id"] for row in inventory["pre_payload_exclusions"]} == {
        "sub-BO199",
        "sub-BO97",
        "sub-NSK07",
    }
    assert {key: value["subjects"] for key, value in inventory["site_counts"].items()} == {
        "BA": 3,
        "BO": 170,
        "FO": 9,
        "JS": 11,
        "KA": 80,
        "ME": 7,
        "NSK": 40,
        "TO": 14,
    }


def test_model_and_statistical_contract():
    assert len(p.AXES) == len(set(p.AXES)) == 8
    assert p.AXES[0] == ()
    assert p.BOOTSTRAP_SAMPLES == 10000
    assert p.BOOTSTRAP_SEED == 20261640
    assert p.sha256(p.FULL) == p.FULL_SHA
    assert p.sha256(p.GCSAM) == p.GCSAM_SHA


def test_frozen_interpretation_gate():
    def comparison(delta, lower, upper):
        return {"candidate": {"scan_dice": delta}, "paired_scan_dice_95ci": [lower, upper]}

    assert interpretation(comparison(0.01, 0.001, 0.02), "candidate") == "supported_on_OpenNeuro_GBM334"
    assert interpretation(comparison(0.01, -0.001, 0.02), "candidate") == "positive_but_uncertain"
    assert interpretation(comparison(0.0, -0.01, 0.01), "candidate") == "not_supported_or_reversed"
    assert interpretation(comparison(-0.01, -0.02, -0.001), "candidate") == "not_supported_or_reversed"


def test_entrypoints_are_syntax_valid_and_do_not_import_old_cohort_payloads():
    forbidden = (
        "RAW_EXTERNAL95.json",
        "RAW_CPTAC33.json",
        "RAW_WAW152.json",
        "RAW_WAW_POSTOP31.json",
        "RAW_TEST270.json",
    )
    for name in (
        "probe_openneuro_gbm334_first_case.py",
        "preflight_openneuro_gbm334_inputs.py",
        "smoke_openneuro_gbm334_panel.py",
        "freeze_openneuro_gbm334_model_preaccess.py",
        "evaluate_openneuro_gbm334_panel.py",
        "finalize_openneuro_gbm334_result.py",
    ):
        text = (ROOT / "scripts" / name).read_text()
        ast.parse(text)
        assert not any(token in text for token in forbidden)
