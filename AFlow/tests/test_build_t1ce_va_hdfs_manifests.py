from scripts.build_t1ce_va_hdfs_manifests import LEGACY_ROOT, VA_ROOT, relocate_row


def test_relocates_inner_t1ce_row() -> None:
    row = {
        "case_id": "BraTS-GLI-00005-100",
        "image_path": f"{LEGACY_ROOT}/part1/case/case-t1c.nii",
        "label_path": f"{LEGACY_ROOT}/part1/case/case-seg.nii",
    }
    relocated = relocate_row(row)
    assert relocated["image_path"] == f"{VA_ROOT}/part1/case/case-t1c.nii"
    assert relocated["label_path"] == f"{VA_ROOT}/part1/case/case-seg.nii"
    assert row["image_path"].startswith(LEGACY_ROOT)


def test_outer_row_becomes_t1ce_only() -> None:
    row = {
        "modalities": {
            "t1c": f"{LEGACY_ROOT}/part2/case/case-t1c.nii",
            "t1n": f"{LEGACY_ROOT}/part2/case/case-t1n.nii",
        },
        "seg_path": f"{LEGACY_ROOT}/part2/case/case-seg.nii",
        "source_case_dir": f"{LEGACY_ROOT}/part2/case",
    }
    relocated = relocate_row(row)
    assert relocated["modalities"] == {
        "t1c": f"{VA_ROOT}/part2/case/case-t1c.nii"
    }
    assert relocated["seg_path"] == f"{VA_ROOT}/part2/case/case-seg.nii"
    assert relocated["source_case_dir"] == f"{VA_ROOT}/part2/case"
