import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts.verify_brats2024_source_provenance import (
    OFFICIAL_ZIP_MD5,
    EXPECTED_SUFFIXES,
    ProvenanceError,
    _verify_source,
    build_archive_inventory,
    build_local_inventory,
    collection_sha256,
    compare_inventories,
    verify_official_source,
    verify_official_source_to_path,
    write_report_exclusive,
)


CASES = ("BraTS-GLI-00005-100", "BraTS-GLI-00006-101")


def payload(case_id: str, suffix: str) -> bytes:
    return f"synthetic:{case_id}:{suffix}\n".encode("ascii")


def make_local_tree(root: Path, cases=CASES) -> Path:
    for index, case_id in enumerate(cases):
        case_dir = root / f"part{index + 1}" / f"Part{index + 1}" / case_id
        case_dir.mkdir(parents=True)
        for suffix in EXPECTED_SUFFIXES:
            (case_dir / f"{case_id}-{suffix}.nii").write_bytes(
                payload(case_id, suffix)
            )
    return root


def make_archive(path: Path, cases=CASES, overrides=None) -> Path:
    overrides = overrides or {}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BraTS2024/README.txt", "synthetic")
        for case_id in cases:
            for suffix in EXPECTED_SUFFIXES:
                content = overrides.get(
                    (case_id, suffix),
                    payload(case_id, suffix),
                )
                archive.writestr(
                    f"BraTS2024/{case_id}/{case_id}-{suffix}.nii",
                    content,
                )
    return path


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_synthetic_archive_and_local_tree_match(tmp_path):
    local_root = make_local_tree(tmp_path / "local")
    archive_path = make_archive(tmp_path / "official.zip")

    report = _verify_source(
        archive_path,
        local_root,
        expected_zip_md5=md5(archive_path),
        expected_cases=len(CASES),
    )

    assert report["status"] == "verified"
    assert report["comparison"] == {"matched": True, "matched_files": 10}
    assert (
        report["archive_inventory"]["collection_sha256"]
        == report["local_inventory"]["collection_sha256"]
    )
    assert [row["key"] for row in report["files"]] == sorted(
        row["key"] for row in report["files"]
    )
    assert all(len(row["sha256"]) == 64 for row in report["files"])


def test_public_entrypoint_requires_official_v2_md5(tmp_path):
    local_root = make_local_tree(tmp_path / "local")
    archive_path = make_archive(tmp_path / "not_official.zip")

    with pytest.raises(ProvenanceError, match=OFFICIAL_ZIP_MD5):
        verify_official_source(archive_path, local_root)


def test_content_mismatch_fails_closed(tmp_path):
    local_root = make_local_tree(tmp_path / "local")
    changed = {(CASES[0], "t1c"): b"different bytes"}
    archive_path = make_archive(tmp_path / "official.zip", overrides=changed)

    archive = build_archive_inventory(archive_path, expected_cases=len(CASES))
    local = build_local_inventory(local_root, expected_cases=len(CASES))
    with pytest.raises(ProvenanceError, match="content mismatch"):
        compare_inventories(archive, local)


def test_missing_local_suffix_is_rejected(tmp_path):
    local_root = make_local_tree(tmp_path / "local")
    missing = (
        local_root
        / "part1"
        / "Part1"
        / CASES[0]
        / f"{CASES[0]}-seg.nii"
    )
    missing.unlink()

    with pytest.raises(ProvenanceError, match="missing suffixes"):
        build_local_inventory(local_root, expected_cases=len(CASES))


def test_duplicate_archive_member_by_canonical_key_is_rejected(tmp_path):
    archive_path = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for case_id in CASES:
            for suffix in EXPECTED_SUFFIXES:
                name = f"first/{case_id}/{case_id}-{suffix}.nii"
                archive.writestr(name, payload(case_id, suffix))
        archive.writestr(
            f"second/{CASES[0]}/{CASES[0]}-seg.nii",
            payload(CASES[0], "seg"),
        )

    with pytest.raises(ProvenanceError, match="duplicate canonical member"):
        build_archive_inventory(archive_path, expected_cases=len(CASES))


def test_unexpected_archive_nifti_is_rejected(tmp_path):
    archive_path = make_archive(tmp_path / "official.zip")
    with zipfile.ZipFile(archive_path, "a") as archive:
        archive.writestr("BraTS2024/unexpected.nii", b"unexpected")

    with pytest.raises(ProvenanceError, match="unexpected NIfTI"):
        build_archive_inventory(archive_path, expected_cases=len(CASES))


def test_collection_root_ignores_source_paths(tmp_path):
    local_a = build_local_inventory(
        make_local_tree(tmp_path / "a"),
        expected_cases=len(CASES),
    )
    local_b = build_local_inventory(
        make_local_tree(tmp_path / "b"),
        expected_cases=len(CASES),
    )

    assert collection_sha256(local_a.values()) == collection_sha256(local_b.values())


def test_failed_official_verification_writes_no_report(tmp_path):
    local_root = make_local_tree(tmp_path / "local")
    archive_path = make_archive(tmp_path / "not_official.zip")
    output = tmp_path / "report.json"

    with pytest.raises(ProvenanceError, match="MD5 mismatch"):
        verify_official_source_to_path(archive_path, local_root, output)

    assert not output.exists()


def test_report_writer_is_exclusive(tmp_path):
    output = tmp_path / "report.json"
    report = {"status": "verified"}
    write_report_exclusive(report, output)

    assert json.loads(output.read_text()) == report
    with pytest.raises(ProvenanceError, match="refusing to overwrite"):
        write_report_exclusive(report, output)
