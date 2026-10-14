from scripts.audit_utsw_primary_fets_otsu_n4_input_bridge import comparison_row, external_case_id


def test_comparison_requires_reproducible_distinct_otsu_image():
    otsu = {"output_image_sha256": "otsu", "output_label_sha256": "label", "geometry": {"x": 1}, "fit_mask": {"x": 2}}
    no_n4 = {"output_image_sha256": "raw", "output_label_sha256": "label", "geometry": {"x": 1}}
    canonical = {"output_image_sha256": "canonical", "output_label_sha256": "label", "geometry": {"x": 1}}
    assert all(comparison_row(otsu, dict(otsu), no_n4, canonical).values())


def test_comparison_rejects_mask_or_arm_collision():
    a = {"output_image_sha256": "same", "output_label_sha256": "label", "geometry": {"x": 1}, "fit_mask": {"x": 2}}
    b = {**a, "fit_mask": {"x": 3}}
    other = {"output_image_sha256": "same", "output_label_sha256": "label", "geometry": {"x": 1}}
    row = comparison_row(a, b, other, {**other, "output_image_sha256": "canonical"})
    assert not row["fit_mask_exact"] and not row["image_different_vs_no_n4"]


def test_external_id_scope_is_fixed():
    assert external_case_id("BT0003") == "BT0003"
    assert external_case_id("BraTS-GLI-00005-100") == "BraTS-GLI-00005-100"
