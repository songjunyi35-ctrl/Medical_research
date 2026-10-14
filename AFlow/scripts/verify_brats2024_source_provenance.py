#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable, Sequence


OFFICIAL_ZIP_MD5 = "1d910b17d6cd32e38aa6296b8dfb7c77"
OFFICIAL_CASE_COUNT = 1350
EXPECTED_SUFFIXES = ("t1n", "t1c", "t2w", "t2f", "seg")
SCHEMA_VERSION = 1
COLLECTION_ROOT_SCHEMA = "brats2024-gli-source-collection-v1"

CASE_ID_PATTERN = r"BraTS-GLI-\d{5}-\d{3}"
CASE_ID_RE = re.compile(rf"^{CASE_ID_PATTERN}$")
SOURCE_FILE_RE = re.compile(
    rf"^(?P<case_id>{CASE_ID_PATTERN})-"
    rf"(?P<suffix>{'|'.join(EXPECTED_SUFFIXES)})"
    r"(?P<extension>\.nii(?:\.gz)?)$"
)


class ProvenanceError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceFileRecord:
    key: str
    case_id: str
    suffix: str
    extension: str
    size_bytes: int
    sha256: str
    source_name: str


def _hash_stream(handle: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        size += len(chunk)
        digest.update(chunk)
    return size, digest.hexdigest()


def _hash_file(path: Path, algorithms: Sequence[str]) -> dict[str, str | int]:
    before = path.stat()
    digests = {name: hashlib.new(name) for name in algorithms}
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            for digest in digests.values():
                digest.update(chunk)
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or size != before.st_size:
        raise ProvenanceError(f"file changed while hashing: {path}")
    return {
        "size_bytes": size,
        **{name: digest.hexdigest() for name, digest in digests.items()},
    }


def _parse_source_filename(filename: str) -> tuple[str, str, str] | None:
    match = SOURCE_FILE_RE.fullmatch(filename)
    if match is None:
        return None
    return (
        match.group("case_id"),
        match.group("suffix"),
        match.group("extension"),
    )


def _canonical_key(case_id: str, suffix: str) -> str:
    return f"{case_id}/{suffix}"


def _validate_inventory(
    records: dict[str, SourceFileRecord],
    *,
    expected_cases: int,
    source: str,
) -> None:
    expected_files = expected_cases * len(EXPECTED_SUFFIXES)
    if len(records) != expected_files:
        raise ProvenanceError(
            f"{source} inventory has {len(records)} files; expected {expected_files}"
        )

    by_case: dict[str, set[str]] = {}
    for record in records.values():
        by_case.setdefault(record.case_id, set()).add(record.suffix)
    if len(by_case) != expected_cases:
        raise ProvenanceError(
            f"{source} inventory has {len(by_case)} cases; expected {expected_cases}"
        )
    expected_suffixes = set(EXPECTED_SUFFIXES)
    incomplete = {
        case_id: sorted(expected_suffixes - suffixes)
        for case_id, suffixes in by_case.items()
        if suffixes != expected_suffixes
    }
    if incomplete:
        examples = list(sorted(incomplete.items()))[:10]
        raise ProvenanceError(f"{source} cases have incomplete suffix sets: {examples}")


def collection_sha256(records: Iterable[SourceFileRecord]) -> str:
    digest = hashlib.sha256()
    digest.update(f"{COLLECTION_ROOT_SCHEMA}\n".encode("ascii"))
    for record in sorted(records, key=lambda item: item.key):
        line = (
            f"{record.key}\t{record.extension}\t{record.size_bytes}\t"
            f"{record.sha256}\n"
        )
        digest.update(line.encode("ascii"))
    return digest.hexdigest()


def build_archive_inventory(
    archive_path: Path,
    *,
    expected_cases: int = OFFICIAL_CASE_COUNT,
) -> dict[str, SourceFileRecord]:
    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise ProvenanceError(f"official archive is not a file: {archive_path}")

    records: dict[str, SourceFileRecord] = {}
    infos: dict[str, zipfile.ZipInfo] = {}
    unexpected_nifti: list[str] = []
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                normalized = info.filename.replace("\\", "/")
                filename = PurePosixPath(normalized).name
                parsed = _parse_source_filename(filename)
                if parsed is None:
                    if filename.endswith((".nii", ".nii.gz")):
                        unexpected_nifti.append(info.filename)
                    continue
                case_id, suffix, extension = parsed
                key = _canonical_key(case_id, suffix)
                if key in infos:
                    raise ProvenanceError(
                        f"archive has duplicate canonical member {key}: "
                        f"{infos[key].filename!r}, {info.filename!r}"
                    )
                unix_mode = info.external_attr >> 16
                if unix_mode and stat.S_ISLNK(unix_mode):
                    raise ProvenanceError(f"archive member is a symlink: {info.filename}")
                if info.flag_bits & 0x1:
                    raise ProvenanceError(f"archive member is encrypted: {info.filename}")
                infos[key] = info

            if unexpected_nifti:
                raise ProvenanceError(
                    "archive has unexpected NIfTI members: "
                    f"{sorted(unexpected_nifti)[:10]}"
                )

            # Validate names and counts before streaming a potentially large archive.
            placeholders = {
                key: SourceFileRecord(
                    key=key,
                    case_id=key.split("/", 1)[0],
                    suffix=key.split("/", 1)[1],
                    extension=_parse_source_filename(
                        PurePosixPath(info.filename.replace("\\", "/")).name
                    )[2],
                    size_bytes=info.file_size,
                    sha256="",
                    source_name=info.filename,
                )
                for key, info in infos.items()
            }
            _validate_inventory(
                placeholders,
                expected_cases=expected_cases,
                source="archive",
            )

            for key in sorted(infos):
                info = infos[key]
                filename = PurePosixPath(info.filename.replace("\\", "/")).name
                parsed = _parse_source_filename(filename)
                assert parsed is not None
                case_id, suffix, extension = parsed
                with archive.open(info, "r") as handle:
                    size, sha256 = _hash_stream(handle)
                if size != info.file_size:
                    raise ProvenanceError(
                        f"archive member size changed while reading: {info.filename}"
                    )
                records[key] = SourceFileRecord(
                    key=key,
                    case_id=case_id,
                    suffix=suffix,
                    extension=extension,
                    size_bytes=size,
                    sha256=sha256,
                    source_name=info.filename,
                )
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        if isinstance(exc, ProvenanceError):
            raise
        raise ProvenanceError(f"failed to read official archive: {exc}") from exc

    _validate_inventory(records, expected_cases=expected_cases, source="archive")
    return records


def build_local_inventory(
    local_root: Path,
    *,
    expected_cases: int = OFFICIAL_CASE_COUNT,
) -> dict[str, SourceFileRecord]:
    local_root = local_root.resolve()
    if not local_root.is_dir():
        raise ProvenanceError(f"local source root is not a directory: {local_root}")

    case_dirs = sorted(
        path
        for path in local_root.glob("part*/Part*/BraTS-GLI-*")
        if path.is_dir()
    )
    if not case_dirs:
        raise ProvenanceError(
            f"no BraTS cases found below {local_root}/part*/Part*/"
        )

    records: dict[str, SourceFileRecord] = {}
    seen_cases: dict[str, Path] = {}
    unexpected_nifti: list[str] = []
    for case_dir in case_dirs:
        case_id = case_dir.name
        if CASE_ID_RE.fullmatch(case_id) is None:
            raise ProvenanceError(f"invalid local case directory: {case_dir}")
        if case_id in seen_cases:
            raise ProvenanceError(
                f"duplicate local case {case_id}: {seen_cases[case_id]}, {case_dir}"
            )
        seen_cases[case_id] = case_dir

        paths_by_suffix: dict[str, tuple[Path, str]] = {}
        for path in sorted(case_dir.iterdir()):
            if not path.is_file() and not path.is_symlink():
                continue
            parsed = _parse_source_filename(path.name)
            if parsed is None:
                if path.name.endswith((".nii", ".nii.gz")):
                    unexpected_nifti.append(str(path))
                continue
            file_case_id, suffix, extension = parsed
            if file_case_id != case_id:
                raise ProvenanceError(
                    f"file/case id mismatch: directory={case_id}, file={path.name}"
                )
            if path.is_symlink():
                raise ProvenanceError(f"local source file is a symlink: {path}")
            if suffix in paths_by_suffix:
                raise ProvenanceError(
                    f"duplicate local suffix for {case_id}/{suffix}: "
                    f"{paths_by_suffix[suffix][0]}, {path}"
                )
            paths_by_suffix[suffix] = (path, extension)

        missing = sorted(set(EXPECTED_SUFFIXES) - set(paths_by_suffix))
        if missing:
            raise ProvenanceError(f"local case {case_id} is missing suffixes: {missing}")

        for suffix in EXPECTED_SUFFIXES:
            path, extension = paths_by_suffix[suffix]
            hashed = _hash_file(path, ("sha256",))
            key = _canonical_key(case_id, suffix)
            records[key] = SourceFileRecord(
                key=key,
                case_id=case_id,
                suffix=suffix,
                extension=extension,
                size_bytes=int(hashed["size_bytes"]),
                sha256=str(hashed["sha256"]),
                source_name=str(path),
            )

    if unexpected_nifti:
        raise ProvenanceError(
            f"local source has unexpected NIfTI files: {sorted(unexpected_nifti)[:10]}"
        )
    _validate_inventory(records, expected_cases=expected_cases, source="local")
    return records


def compare_inventories(
    archive_records: dict[str, SourceFileRecord],
    local_records: dict[str, SourceFileRecord],
) -> list[dict[str, str | int]]:
    archive_keys = set(archive_records)
    local_keys = set(local_records)
    if archive_keys != local_keys:
        raise ProvenanceError(
            "archive/local key sets differ: "
            f"archive_only={sorted(archive_keys - local_keys)[:10]}, "
            f"local_only={sorted(local_keys - archive_keys)[:10]}"
        )

    mismatches = []
    matched = []
    for key in sorted(archive_keys):
        archive = archive_records[key]
        local = local_records[key]
        differences = {}
        for field in ("extension", "size_bytes", "sha256"):
            archive_value = getattr(archive, field)
            local_value = getattr(local, field)
            if archive_value != local_value:
                differences[field] = {
                    "archive": archive_value,
                    "local": local_value,
                }
        if differences:
            mismatches.append({"key": key, "differences": differences})
            continue
        matched.append(
            {
                "key": key,
                "case_id": archive.case_id,
                "suffix": archive.suffix,
                "extension": archive.extension,
                "size_bytes": archive.size_bytes,
                "sha256": archive.sha256,
                "archive_member": archive.source_name,
                "local_path": local.source_name,
            }
        )

    if mismatches:
        raise ProvenanceError(
            f"archive/local content mismatch in {len(mismatches)} files: "
            f"{mismatches[:10]}"
        )

    archive_root = collection_sha256(archive_records.values())
    local_root = collection_sha256(local_records.values())
    if archive_root != local_root:
        raise ProvenanceError(
            f"collection roots differ despite matched files: {archive_root} != {local_root}"
        )
    return matched


def _verify_source(
    archive_path: Path,
    local_root: Path,
    *,
    expected_zip_md5: str,
    expected_cases: int,
) -> dict:
    archive_path = archive_path.resolve()
    local_root = local_root.resolve()
    archive_before = archive_path.stat()
    archive_hashes = _hash_file(archive_path, ("md5", "sha256"))
    actual_md5 = str(archive_hashes["md5"])
    if actual_md5 != expected_zip_md5:
        raise ProvenanceError(
            f"official archive MD5 mismatch: expected {expected_zip_md5}, got {actual_md5}"
        )

    archive_records = build_archive_inventory(
        archive_path,
        expected_cases=expected_cases,
    )
    local_records = build_local_inventory(
        local_root,
        expected_cases=expected_cases,
    )
    matched = compare_inventories(archive_records, local_records)

    archive_after = archive_path.stat()
    before_identity = (
        archive_before.st_dev,
        archive_before.st_ino,
        archive_before.st_size,
        archive_before.st_mtime_ns,
    )
    after_identity = (
        archive_after.st_dev,
        archive_after.st_ino,
        archive_after.st_size,
        archive_after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise ProvenanceError(f"official archive changed during verification: {archive_path}")

    archive_root = collection_sha256(archive_records.values())
    local_collection_root = collection_sha256(local_records.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "verified",
        "official_archive": {
            "path": str(archive_path),
            "size_bytes": int(archive_hashes["size_bytes"]),
            "md5": actual_md5,
            "required_md5": expected_zip_md5,
            "sha256": str(archive_hashes["sha256"]),
        },
        "local_root": str(local_root),
        "expected": {
            "cases": expected_cases,
            "files_per_case": len(EXPECTED_SUFFIXES),
            "files": expected_cases * len(EXPECTED_SUFFIXES),
            "suffixes": list(EXPECTED_SUFFIXES),
        },
        "archive_inventory": {
            "cases": len({record.case_id for record in archive_records.values()}),
            "files": len(archive_records),
            "collection_root_schema": COLLECTION_ROOT_SCHEMA,
            "collection_sha256": archive_root,
        },
        "local_inventory": {
            "cases": len({record.case_id for record in local_records.values()}),
            "files": len(local_records),
            "collection_root_schema": COLLECTION_ROOT_SCHEMA,
            "collection_sha256": local_collection_root,
        },
        "comparison": {
            "matched": True,
            "matched_files": len(matched),
        },
        "files": matched,
    }


def verify_official_source(archive_path: Path, local_root: Path) -> dict:
    return _verify_source(
        archive_path,
        local_root,
        expected_zip_md5=OFFICIAL_ZIP_MD5,
        expected_cases=OFFICIAL_CASE_COUNT,
    )


def write_report_exclusive(report: dict, output_path: Path) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise ProvenanceError(f"refusing to overwrite provenance report: {output_path}")
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output_path)
        except FileExistsError as exc:
            raise ProvenanceError(
                f"refusing to overwrite provenance report: {output_path}"
            ) from exc
        except OSError as exc:
            raise ProvenanceError(
                f"failed to publish provenance report without overwrite: {exc}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def verify_official_source_to_path(
    archive_path: Path,
    local_root: Path,
    output_path: Path,
) -> dict:
    report = verify_official_source(archive_path, local_root)
    write_report_exclusive(report, output_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed provenance verification for the official BraTS2024 GLI v2 "
            "training archive and the local 1350-case source tree."
        )
    )
    parser.add_argument("--official-zip", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = verify_official_source_to_path(
            args.official_zip,
            args.local_root,
            args.output,
        )
    except (OSError, ProvenanceError) as exc:
        print(f"provenance verification failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "official_archive_md5": report["official_archive"]["md5"],
                "files": report["comparison"]["matched_files"],
                "collection_sha256": report["local_inventory"]["collection_sha256"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
