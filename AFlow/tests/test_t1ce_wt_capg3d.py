import json

import pytest
import torch

from sammed3d.t1ce_wt_cached import CACHE_SCHEMA
from sammed3d.t1ce_wt_capg3d import (
    AutomaticPromptLocator,
    LocatorOutput,
    automatic_prompt_points,
    center_heatmap_loss,
    foreground_anchor,
    gaussian_center_target,
)
from scripts.train_t1ce_wt_capg3d import BrainCropDataset, resolve_case_slots


@pytest.mark.parametrize("center_aware,channels", [(False, 1), (True, 2)])
def test_locator_output_contract(center_aware, channels):
    torch.manual_seed(3)
    model = AutomaticPromptLocator(
        center_aware=center_aware, init_filters=8, dropout=0.0
    ).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 1, 32, 32, 32))
    assert output.segmentation_logits.shape == (1, 1, 32, 32, 32)
    assert (output.center_logits is not None) is center_aware
    assert model.network.conv_final[2].conv.out_channels == channels


def test_center_target_peak_is_valid_foreground():
    label = torch.zeros(2, 1, 15, 15, 15)
    label[0, 0, 2:5, 3:7, 4:9] = 1
    label[1, 0, 10:13, 8:12, 1:4] = 1
    target, anchors = gaussian_center_target(label, sigma=2.0)
    assert target.shape == label.shape
    for batch in range(2):
        anchor = tuple(int(value) for value in anchors[batch])
        assert label[(batch, 0, *anchor)] == 1
        assert target[(batch, 0, *anchor)] == 1
    assert torch.equal(anchors, foreground_anchor(label))
    assert torch.isfinite(center_heatmap_loss(torch.zeros_like(target), target))


def test_automatic_prompt_prefers_center_head_when_present():
    segmentation = torch.zeros(1, 1, 8, 8, 8)
    segmentation[0, 0, 1, 2, 3] = 9
    center = torch.zeros_like(segmentation)
    center[0, 0, 6, 5, 4] = 12
    plain = automatic_prompt_points(LocatorOutput(segmentation, None))
    capg = automatic_prompt_points(LocatorOutput(segmentation, center))
    assert plain.tolist() == [[[1.0, 2.0, 3.0]]]
    assert capg.tolist() == [[[6.0, 5.0, 4.0]]]


def test_paired_initialization_matches_outside_final_head():
    torch.manual_seed(20260819)
    plain = AutomaticPromptLocator(center_aware=False, init_filters=8, dropout=0.1)
    torch.manual_seed(20260819)
    capg = AutomaticPromptLocator(center_aware=True, init_filters=8, dropout=0.1)
    plain_state = plain.state_dict()
    capg_state = capg.state_dict()
    final_prefix = "network.conv_final.2.conv."
    shared_keys = [key for key in plain_state if not key.startswith(final_prefix)]
    assert shared_keys
    for key in shared_keys:
        assert torch.equal(plain_state[key], capg_state[key]), key
    # Channel zero in the enlarged final head is also initialized identically.
    assert torch.equal(
        plain_state[final_prefix + "weight"][0],
        capg_state[final_prefix + "weight"][0],
    )
    assert torch.equal(
        plain_state[final_prefix + "bias"][0],
        capg_state[final_prefix + "bias"][0],
    )
    assert float(capg_state[final_prefix + "bias"][1]) == -4.0


def test_dataset_rejects_test_access(tmp_path):
    index = {
        "schema": CACHE_SCHEMA,
        "test_access": True,
        "split": "test",
        "count": 1,
        "case_ids": ["BraTS-GLI-00001-100"],
        "preprocessing": {
            "spatial": "braincrop_resize",
            "uses_gt_for_localization": False,
        },
    }
    (tmp_path / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="test-accessible"):
        BrainCropDataset(tmp_path, expected_split="validation")


def test_dataset_rejects_gt_localized_geometry(tmp_path):
    index = {
        "schema": CACHE_SCHEMA,
        "test_access": False,
        "split": "validation",
        "count": 1,
        "case_ids": ["BraTS-GLI-00001-100"],
        "preprocessing": {
            "spatial": "braincrop_resize",
            "uses_gt_for_localization": True,
        },
    }
    (tmp_path / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="must not use GT"):
        BrainCropDataset(tmp_path, expected_split="validation")


def test_case_view_resolves_ordered_source_slots():
    available = [
        "BraTS-GLI-00001-100",
        "BraTS-GLI-00001-101",
        "BraTS-GLI-00002-100",
    ]
    requested = ["BraTS-GLI-00002-100", "BraTS-GLI-00001-100"]
    slots, case_ids = resolve_case_slots(available, requested)
    assert slots == [2, 0]
    assert case_ids == requested


def test_case_view_rejects_unknown_and_duplicate_cases():
    available = ["BraTS-GLI-00001-100", "BraTS-GLI-00002-100"]
    with pytest.raises(ValueError, match="absent from source cache"):
        resolve_case_slots(available, ["BraTS-GLI-99999-100"])
    with pytest.raises(ValueError, match="duplicate"):
        resolve_case_slots(available, [available[0], available[0]])
