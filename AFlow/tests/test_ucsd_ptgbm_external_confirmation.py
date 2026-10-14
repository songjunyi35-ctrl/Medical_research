from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import io
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import scripts.evaluate_ucsd_ptgbm_external_confirmation as evaluate
import scripts.prepare_ucsd_ptgbm_external_confirmation as prepare


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _synthetic_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[Path], Path, Path]:
    participants = tmp_path / "participants.tsv"
    main_rows: list[tuple[str, str]] = []
    # 83 exposed subjects contribute 109 cases; 53 eligible subjects contribute
    # 75 cases.  This exactly exercises the frozen patient-level difference.
    session_counts = [2] * 26 + [1] * 57 + [2] * 22 + [1] * 31
    assert len(session_counts) == 136 and sum(session_counts) == 184
    for subject_index, sessions in enumerate(session_counts, start=1):
        for session in range(1, sessions + 1):
            local = f"sub-{subject_index:04d}_{session:02d}"
            brats = f"BraTS-GLI-{2200 + subject_index:05d}-{99 + session:03d}"
            main_rows.append((local, brats))
    challenge_rows: list[tuple[str, str]] = []
    challenge_counts = [2] * 17 + [1] * 25
    for offset, sessions in enumerate(challenge_counts, start=137):
        for session in range(1, sessions + 1):
            challenge_rows.append((f"sub-{offset:04d}_{session:02d}", f"BraTS-GLI-{2200 + offset:05d}-{99 + session:03d}"))
    assert len(main_rows) == 184 and len(challenge_rows) == 59
    with participants.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "BraTS ID"], delimiter="\t")
        writer.writeheader()
        for local, brats in main_rows + challenge_rows:
            writer.writerow({"ID": local, "BraTS ID": brats})
    monkeypatch.setattr(prepare, "PARTICIPANTS_SHA256", _sha(participants))

    release_paths = []
    assets_by_tag = {tag: [] for tag in prepare.EXPECTED_RELEASES}
    for index, (local, _) in enumerate(main_rows):
        subject, session = local.split("_")
        tag = list(prepare.EXPECTED_RELEASES)[index % 4]
        for suffix, size in (("run-post_T1w.nii.gz", 101), ("BraTS_tumor_seg.nii.gz", 17)):
            name = f"{subject}_ses-{session}_{suffix}"
            assets_by_tag[tag].append({
                "name": name, "size": size, "digest": "sha256:" + hashlib.sha256(name.encode()).hexdigest(),
                "browser_download_url": f"https://github.com/data-nih/tcia/releases/download/{tag}/{name}",
            })
    for tag, release_id in prepare.EXPECTED_RELEASES.items():
        path = tmp_path / f"{tag}.json"
        path.write_text(json.dumps({"tag_name": tag, "id": release_id, "html_url": f"https://github.com/data-nih/tcia/releases/tag/{tag}", "assets": assets_by_tag[tag]}))
        release_paths.append(path)

    exposed = main_rows[:109]
    exposure_rows = [{"case_id": brats, "subject_id": brats.rsplit("-", 1)[0]} for _, brats in exposed]
    while len(exposure_rows) < 1350:
        index = len(exposure_rows)
        case = f"BraTS-GLI-{50000 + index:05d}-100"
        exposure_rows.append({"case_id": case, "subject_id": case.rsplit("-", 1)[0]})
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    _write_jsonl(train, exposure_rows[:1080])
    _write_jsonl(test, exposure_rows[1080:])
    monkeypatch.setattr(prepare, "TRAIN_SHA256", _sha(train))
    monkeypatch.setattr(prepare, "TEST_SHA256", _sha(test))
    return participants, release_paths, train, test


def test_inventory_builds_exact_patient_disjoint_cohort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    participants, releases, train, test = _synthetic_source(tmp_path, monkeypatch)
    output = tmp_path / "inventory.json"
    args = type("Args", (), {"participants": participants, "release_json": releases, "train_manifest": train, "test_manifest": test, "output": output})()
    inventory = prepare.build_inventory(args)
    assert inventory["main_release"] == {"cases": 184, "subjects": 136, "overlapping_cases": 109}
    assert inventory["selection"]["cases"] == 75
    assert inventory["selection"]["subjects"] == 53
    assert inventory["selection"]["case_overlap"] == 0
    assert inventory["selection"]["subject_overlap"] == 0
    assert inventory["source"]["separate_brats_challenge_test_package_used"] is False
    assert len(inventory["selection"]["selected_identity_sha256"]) == 64
    assert json.loads(output.read_text())["access"]["eligible_label_voxel_access"] is False


def test_inventory_rejects_incomplete_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    participants, releases, train, test = _synthetic_source(tmp_path, monkeypatch)
    payload = json.loads(releases[0].read_text())
    payload["assets"] = payload["assets"][1:]
    releases[0].write_text(json.dumps(payload))
    args = type("Args", (), {"participants": participants, "release_json": releases, "train_manifest": train, "test_manifest": test, "output": tmp_path / "bad.json"})()
    with pytest.raises(prepare.ExternalPreparationError, match="incomplete"):
        prepare.build_inventory(args)


def test_release_reader_rejects_unfrozen_tag(tmp_path: Path) -> None:
    path = tmp_path / "challenge.json"
    path.write_text(json.dumps({"tag_name": "BraTS-GLI-2024-Test-Data", "id": 1, "html_url": "x", "assets": []}))
    with pytest.raises(prepare.ExternalPreparationError, match="unexpected release"):
        prepare.read_release_assets([path])


def test_download_is_atomic_and_hash_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"official bytes"

    @contextmanager
    def fake_open(*_args, **_kwargs):
        yield io.BytesIO(payload)

    monkeypatch.setattr(prepare, "urlopen", fake_open)
    output = tmp_path / "case" / "image.nii.gz"
    result = prepare._download_one("https://example.invalid/image", output, hashlib.sha256(payload).hexdigest(), len(payload))
    assert result == "downloaded"
    assert output.read_bytes() == payload
    assert not output.with_suffix(output.suffix + ".part").exists()
    assert prepare._download_one("https://example.invalid/image", output, hashlib.sha256(payload).hexdigest(), len(payload)) == "already_verified"


def test_download_rejects_wrong_hash_without_leaving_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    @contextmanager
    def fake_open(*_args, **_kwargs):
        yield io.BytesIO(b"wrong")

    monkeypatch.setattr(prepare, "urlopen", fake_open)
    output = tmp_path / "label.nii.gz"
    with pytest.raises(prepare.ExternalPreparationError, match="digest/size"):
        prepare._download_one("https://example.invalid/label", output, "0" * 64, 5)
    assert not output.exists()
    assert not output.with_suffix(output.suffix + ".part").exists()


def test_prepare_case_rejects_label_outside_frozen_domain(tmp_path: Path) -> None:
    image = np.arange(1, 8 * 8 * 8 + 1, dtype=np.float32).reshape(8, 8, 8)
    label = np.zeros((8, 8, 8), dtype=np.uint8)
    label[3, 3, 3] = 5
    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
    nib.save(nib.Nifti1Image(label, np.eye(4)), label_path)
    row = {"brats_case_id": "BraTS-GLI-99999-100", "brats_subject_id": "BraTS-GLI-99999", "ucsd_case_id": "sub-9999_ses-01", "local_image_path": str(image_path), "local_label_path": str(label_path)}
    with pytest.raises(prepare.ExternalPreparationError, match="unexpected raw labels"):
        prepare._prepare_case((0, row))


def test_prepare_case_retains_empty_wt_and_excludes_label4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.arange(1, 8 * 8 * 8 + 1, dtype=np.float32).reshape(8, 8, 8)
    label = np.zeros((8, 8, 8), dtype=np.uint8)
    label[3:5, 3:5, 3:5] = 4
    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
    nib.save(nib.Nifti1Image(label, np.eye(4)), label_path)
    images_path = tmp_path / "images.f32"
    labels_path = tmp_path / "labels.u8"
    np.memmap(images_path, dtype=np.float32, mode="w+", shape=(1, *prepare.MODEL_SHAPE)).flush()
    np.memmap(labels_path, dtype=np.uint8, mode="w+", shape=(1, *prepare.MODEL_SHAPE)).flush()
    monkeypatch.setattr(prepare, "n4_correct", lambda value, _shrink: value)
    prepare._CFG.clear()
    prepare._CFG["images_fd"] = os.open(images_path, os.O_WRONLY)
    prepare._CFG["labels_fd"] = os.open(labels_path, os.O_WRONLY)
    row = {"brats_case_id": "BraTS-GLI-99999-100", "brats_subject_id": "BraTS-GLI-99999", "ucsd_case_id": "sub-9999_ses-01", "local_image_path": str(image_path), "local_label_path": str(label_path)}
    try:
        _, metadata = prepare._prepare_case((0, row))
    finally:
        os.close(prepare._CFG.pop("images_fd"))
        os.close(prepare._CFG.pop("labels_fd"))
    assert metadata["raw_labels"] == [0, 4]
    assert metadata["source_wt_empty"] is True
    assert metadata["resized_wt_empty"] is True
    cached = np.memmap(labels_path, dtype=np.uint8, mode="r", shape=(1, *prepare.MODEL_SHAPE))
    assert int(cached.sum()) == 0


def _summary(case_values: list[tuple[str, str, float, bool | None]]) -> dict:
    per_case = [{"case_id": case, "subject_id": subject, "locator_dsc": dice, "automatic_point_hits_gt": point, "target_wt_positive": point is not None, "predicted_wt_voxels": 1} for case, subject, dice, point in case_values]
    positive = [row for row in per_case if row["target_wt_positive"]]
    by_subject: dict[str, list[float]] = {}
    for row in per_case:
        by_subject.setdefault(row["subject_id"], []).append(row["locator_dsc"])
    return {
        "per_case": per_case,
        "locator_dsc_mean": float(np.mean([row["locator_dsc"] for row in per_case])),
        "locator_subject_dsc_mean": float(np.mean([np.mean(values) for values in by_subject.values()])),
        "automatic_point_hits": sum(bool(row["automatic_point_hits_gt"]) for row in positive),
        "automatic_point_hit_rate_positive_only": float(np.mean([row["automatic_point_hits_gt"] for row in positive])),
    }


def test_compare_is_paired_by_case_and_subject() -> None:
    primary = _summary([("a", "s1", 0.8, True), ("b", "s1", 0.6, False), ("c", "s2", 0.9, True)])
    secondary = _summary([("a", "s1", 0.7, True), ("b", "s1", 0.5, True), ("c", "s2", 0.8, False)])
    result = evaluate.compare(primary, secondary)
    assert result["primary_minus_secondary"]["scan_dsc_mean"] == pytest.approx(0.1)
    assert result["primary_minus_secondary"]["subject_dsc_mean"] == pytest.approx(0.1)
    assert result["primary_minus_secondary"]["positive_point_hits"] == 0
    assert result["scan_cases_improved"] == 3
    assert result["subjects_improved"] == 2


def test_performance_gates_remain_frozen() -> None:
    primary = {"locator_dsc_mean": 0.70, "locator_subject_dsc_mean": 0.70, "automatic_point_hit_rate_positive_only": 0.90}
    comparison = {"primary_minus_secondary": {"scan_dsc_mean": 0.0, "subject_dsc_mean": 0.0, "positive_point_hits": -1}, "paired_scan_dsc_mean_95ci": [-0.003, 0.01], "paired_subject_dsc_mean_95ci": [-0.003, 0.01]}
    result = evaluate.gates(primary, comparison)
    assert result["all_performance_gates_passed"] is True
    comparison["paired_subject_dsc_mean_95ci"][0] = -0.003001
    assert evaluate.gates(primary, comparison)["all_performance_gates_passed"] is False
