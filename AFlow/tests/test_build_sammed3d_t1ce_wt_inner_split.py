from scripts import build_sammed3d_t1ce_wt_inner_split as split


def _rows() -> list[dict]:
    rows = []
    for subject_index, count in enumerate((1, 2, 3, 4), start=1):
        subject_id = f"BraTS-GLI-{subject_index:05d}"
        for timepoint in range(count):
            case_id = f"{subject_id}-{100 + timepoint:03d}"
            rows.append(
                {
                    "case_id": case_id,
                    "subject_id": subject_id,
                    "timepoint": f"{100 + timepoint:03d}",
                    "split": "train",
                    "modalities": {"t1c": f"/data/{case_id}-t1c.nii"},
                    "seg_path": f"/data/{case_id}-seg.nii",
                    "source_part": "Part1",
                }
            )
    return rows


def test_exact_inner_split_is_deterministic_and_subject_clean() -> None:
    first_train, first_val = split.split_rows(
        _rows(), seed=17, validation_cases=4
    )
    second_train, second_val = split.split_rows(
        list(reversed(_rows())), seed=17, validation_cases=4
    )
    assert first_train == second_train
    assert first_val == second_val
    assert len(first_train) == 6
    assert len(first_val) == 4
    assert split.subject_ids(first_train).isdisjoint(split.subject_ids(first_val))
    assert {row["split"] for row in first_train} == {"train"}
    assert {row["split"] for row in first_val} == {"validation"}
    assert {row["target"] for row in first_train + first_val} == {"WT"}
    assert {tuple(row["foreground_labels"]) for row in first_train + first_val} == {
        (1, 2, 3)
    }
