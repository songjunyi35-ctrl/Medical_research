from scripts import prepare_remind_t1ce_source_probe as prepare


def test_t1_tumor_path_contract_excludes_other_structures_and_modalities():
    assert prepare.is_t1_tumor_path("ReMIND_NRRD_Seg_Sep_2023/ReMIND-002/ReMIND-002-preop-SEG-tumor-MR-3D_AX_T1_postcontrast.nrrd")
    assert prepare.is_t1_tumor_path("ReMIND_NRRD_Seg_Sep_2023/ReMIND-099/ReMIND-099-preop-SEG-tumor-MR-3D_AX_T1_postcontrast-r.nrrd")
    assert not prepare.is_t1_tumor_path("ReMIND_NRRD_Seg_Sep_2023/ReMIND-002/ReMIND-002-preop-SEG-tumor-MR-3D_AX_T2_SPACE.nrrd")
    assert not prepare.is_t1_tumor_path("ReMIND_NRRD_Seg_Sep_2023/ReMIND-002/ReMIND-002-preop-SEG-tumor_target-MR-3D_AX_T1_postcontrast.nrrd")
    assert not prepare.is_t1_tumor_path("ReMIND_NRRD_Seg_Sep_2023/ReMIND-002/ReMIND-002-intraop-SEG-tumor_residual-MR-3D_AX_T1_postcontrast.nrrd")


def test_actual_frozen_metadata_builds_expected_cohort():
    rows = prepare.build_candidates()
    assert len(rows) == 43
    assert rows[0]["case_id"] == "ReMIND-002"
    assert len({row["case_id"] for row in rows}) == 43
    assert all(row["histopathology"] in prepare.DIAGNOSES for row in rows)
    assert sum(row["automatic_cerebrum"] is not None for row in rows) == 34
