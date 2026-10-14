from scripts.audit_utsw_primary_fets_canonical_n4_input_bridge import comparison_row, external_case_id


def test_comparison_requires_cross_process_exact_and_expected_arm_difference():
    canonical = {"output_image_sha256": "n4", "output_label_sha256": "label", "geometry": {"x": 1}}
    no_n4 = {"output_image_sha256": "raw", "output_label_sha256": "label", "geometry": {"x": 1}}
    assert all(comparison_row(canonical, dict(canonical), no_n4).values())
    same_image = dict(no_n4)
    same_image["output_label_sha256"] = "label"
    assert not comparison_row(same_image, dict(same_image), no_n4)["image_different_vs_no_n4"]


def test_comparison_rejects_label_or_geometry_drift():
    a = {"output_image_sha256": "a", "output_label_sha256": "label", "geometry": {"x": 1}}
    b = {"output_image_sha256": "a", "output_label_sha256": "wrong", "geometry": {"x": 2}}
    no_n4 = {"output_image_sha256": "b", "output_label_sha256": "label", "geometry": {"x": 1}}
    row = comparison_row(a, b, no_n4)
    assert not row["cross_process_label_exact"]
    assert not row["cross_process_geometry_exact"]


def test_external_id_scope_is_fixed():
    assert external_case_id("BT0001") == "BT0001"
    assert external_case_id("BraTS-GLI-00005-100") == "BraTS-GLI-00005-100"
