import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "archive/research_records/2026-09-06_motum_hgg29_frozen_panel"
BRIDGE = ROOT / "archive/research_records/2026-09-06_motum_hgg29_shared_orientation_bridge"
SHARED = ROOT / "archive/research_records/2026-09-06_motum_hgg29_shared_orientation_panel"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_independent_canonical_protocol_closed_before_model():
    value = load(OLD / "CACHE_PROBE_FAILURE.json")
    assert value["status"] == "closed_before_cache_checkpoint_or_performance"
    assert value["independent_closest_canonical"]["passed"] is False
    assert value["all29_read_only_diagnostic"]["cases_above_frozen_1e_5_gate"] == 2
    assert value["cache_created"] is False
    assert value["checkpoint_loaded"] is False
    assert value["model_forwards"] == 0
    assert value["performance_metrics"] is False


def test_shared_orientation_bridge_passed_without_interpolation_or_model():
    value = load(BRIDGE / "RESULT.json")
    assert value["status"] == "passed"
    assert value["cases"] == 29
    assert value["native_affine_within_5e_7"] == 29
    assert value["identical_discrete_orientation"] == 29
    assert value["shared_arrays_equal_independent_canonical"] == 29
    assert value["copy_a_b_case_payload_exact"] is True
    assert value["interpolation"] is False
    assert value["cache_created"] is False
    assert value["model_forwards"] == 0


def test_shared_panel_stopped_at_image_support_not_geometry():
    value = load(SHARED / "CACHE_PROBE_FAILURE.json")
    assert value["shared_orientation_geometry_passed"] is True
    assert value["failure"] == "frozen zero-WT-crop-loss gate"
    assert value["raw_wt_voxels"] == 21007
    assert value["wt_voxels_lost"] == 892
    assert value["all29_read_only_diagnostic"]["cases_with_wt_crop_loss"] == 1
    assert value["cache_created"] is False
    assert value["checkpoint_loaded"] is False
    assert value["model_forwards"] == 0
    assert value["performance_metrics"] is False


def test_no_motum_performance_artifact_exists():
    forbidden = (
        OLD / "RAW_MOTUM_HGG29.json",
        OLD / "RAW_MOTUM_HGG29_STARTED.json",
        SHARED / "RAW_MOTUM_HGG29.json",
        SHARED / "RAW_MOTUM_HGG29_STARTED.json",
    )
    assert not any(path.exists() for path in forbidden)
