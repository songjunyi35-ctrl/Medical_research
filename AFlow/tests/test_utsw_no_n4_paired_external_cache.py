import numpy as np
from pathlib import Path

from scripts.build_utsw_no_n4_paired_external_caches import pair_gates


def test_pair_gate_requires_same_label_geometry_and_changed_image():
    label = np.array([0, 1], dtype=np.uint8)
    assert all(pair_gates(np.array([1.0]), label, {"x": 1}, np.array([2.0]), label.copy(), {"x": 1}).values())
    assert not pair_gates(np.array([1.0]), label, {"x": 1}, np.array([1.0]), label.copy(), {"x": 1})["images_different"]
    assert not pair_gates(np.array([1.0]), label, {"x": 1}, np.array([2.0]), np.array([1, 0]), {"x": 1})["labels_exact"]
    assert not pair_gates(np.array([1.0]), label, {"x": 1}, np.array([2.0]), label.copy(), {"x": 2})["geometry_exact"]


def test_full_cache_hard_gate_matches_preregistration():
    text = Path("scripts/build_utsw_no_n4_paired_external_caches.py").read_text()
    assert 'if not gates["labels_exact"] or not gates["geometry_exact"]' in text
    assert '"images_files_different": indices["canonical"]["images_sha256"] != indices["no_n4"]["images_sha256"]' in text
    assert 'set(source["label_values"]) <= {0.0, 1.0, 2.0, 4.0}' in text
