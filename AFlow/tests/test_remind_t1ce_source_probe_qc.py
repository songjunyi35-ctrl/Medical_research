from pathlib import Path

import numpy as np
import SimpleITK as sitk

from scripts.audit_remind_t1ce_source_probe import binary, geometry_delta


def test_geometry_delta_exact_and_detects_origin_shift():
    first = sitk.Image([8, 9, 10], sitk.sitkFloat32)
    first.SetSpacing((1.0, 1.2, 1.4))
    first.SetOrigin((2.0, 3.0, 4.0))
    second = sitk.Image(first)
    value = geometry_delta(first, second)
    assert value["size_exact"] and value["corner_max_abs_mm"] == 0
    second.SetOrigin((2.0, 3.0, 4.01))
    assert geometry_delta(first, second)["corner_max_abs_mm"] > 1e-5


def test_binary_mask_contract():
    ok, values = binary(np.asarray([0, 1, 0], dtype=np.uint8).reshape(1, 1, 3))
    assert ok and values == [0.0, 1.0]
    assert not binary(np.zeros((2, 2, 2), dtype=np.uint8))[0]
    assert not binary(np.asarray([0, 2], dtype=np.uint8).reshape(1, 1, 2))[0]


def test_download_source_requires_two_transfers_and_no_model():
    root = Path(__file__).parents[1]
    source = (root / "scripts/download_remind_t1ce_source_probe.py").read_text()
    assert 'first = download_nrrds(paths' in source
    assert 'second = download_nrrds(paths' in source
    assert 'a["sha256"] == b["sha256"]' in source
    assert 'model_inference": False' in source
