#!/usr/bin/env python3
"""Build the private, code-and-records-only research migration archive."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tarfile
import tempfile
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
NAME = "research_code_migration_2026-10-14"
SKIP_DIRS = {
    ".git", ".pytest_cache", "__pycache__", ".venv", ".vendor",
    ".codex", ".agents", "node_modules", ".mypy_cache", ".ruff_cache",
}
SKIP_SUFFIXES = {
    ".pth", ".pt", ".ckpt", ".safetensors", ".f32", ".f16", ".u8",
    ".npz", ".bits", ".nii", ".zip", ".sums", ".xlsx", ".pyc",
    ".lock", ".so", ".bin",
}
AFLOW_SUFFIXES = {
    ".py", ".sh", ".md", ".txt", ".json", ".jsonl", ".csv",
    ".tsv", ".yaml", ".yml", ".toml", ".cpp", ".r", ".html",
    ".css", ".js", ".patch", ".orig", ".disabled", ".sha256",
    ".log", ".png", ".jpg", ".pdf", ".xml", ".tag",
}
ARTICLE_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".yaml", ".tex", ".bib",
    ".pdf", ".png",
}
SPECIAL_NAMES = {"LICENSE", "SHA256SUMS", ".gitignore", "Makefile"}
SECRET_MARKERS = (
    b"-----BEGIN PRIVATE KEY-----", b"-----BEGIN RSA PRIVATE KEY-----",
    b"-----BEGIN OPENSSH PRIVATE KEY-----", b"ghp_", b"github_pat_",
    b"hf_", b"AKIA", b"sk-proj-",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decide(path: Path, project: str) -> str | None:
    rel = path.relative_to(HERE / project)
    parts = rel.parts
    if any(part in SKIP_DIRS for part in parts):
        return "generated or local environment"
    if project == "AFlow" and rel.as_posix() == "config/config2.yaml":
        return "local credential configuration"
    if path.is_symlink():
        target = (path.parent / os.readlink(path)).resolve()
        if not target.is_relative_to(HERE / project):
            return "link leaves project"
        return None
    if not path.is_file():
        return "not a regular file"
    if path.suffix.lower() in SKIP_SUFFIXES or path.name.endswith(".nii.gz"):
        return "weight, data, cache, or generated binary"
    if project == "AFlow":
        if path.name.endswith(".md.gz") and "archive" in parts:
            return None
        if path.suffix.lower() not in AFLOW_SUFFIXES and path.name not in SPECIAL_NAMES and not path.suffix.startswith((".before_", ".attempt")):
            return "outside code and records allowlist"
        # Source-release clinical tables are dataset content, not research prose.
        if path.suffix.lower() in {".csv", ".tsv", ".xml"} and "source_metadata" in parts:
            return "copied source dataset metadata table"
        if "workspace" in parts and "metadata" in parts and "reproducibility_code" not in parts:
            if path.suffix.lower() in {".csv", ".tsv", ".xml", ".png", ".jpg", ".pdf"}:
                return "copied external dataset metadata or image"
    else:
        if path.suffix.lower() not in ARTICLE_SUFFIXES and path.name not in SPECIAL_NAMES:
            return "article build dependency or generated file"
    if path.stat().st_size > 20 * 1024 * 1024:
        return "over 20 MiB outside selected data manifest"
    return None


def inventory() -> tuple[list[tuple[Path, Path]], Counter, int]:
    included: list[tuple[Path, Path]] = []
    excluded: Counter = Counter()
    total_bytes = 0
    for project in ("AFlow", "article"):
        root = HERE / project
        for base, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            for name in sorted(files):
                path = Path(base) / name
                reason = decide(path, project)
                if reason:
                    excluded[reason] += 1
                    continue
                included.append((path, path.relative_to(HERE)))
                if not path.is_symlink():
                    total_bytes += path.stat().st_size
    included.sort(key=lambda pair: pair[1].as_posix())
    return included, excluded, total_bytes


def scan_markers(items: list[tuple[Path, Path]]) -> list[tuple[str, str]]:
    found = []
    text_suffixes = AFLOW_SUFFIXES - {".png", ".jpg", ".pdf"}
    for source, rel in items:
        if source.is_symlink() or source.suffix.lower() not in text_suffixes:
            continue
        data = source.read_bytes()
        for marker in SECRET_MARKERS:
            if marker in data:
                found.append((rel.as_posix(), marker.decode()))
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="create tar.gz; default is inventory only")
    parser.add_argument("--guide", type=Path, help="reviewed migration guide for the archive root")
    args = parser.parse_args()
    items, excluded, total_bytes = inventory()
    print(f"Selected {len(items)} paths, {total_bytes / 1048576:.1f} MiB before compression")
    for reason, count in sorted(excluded.items()):
        print(f"Excluded {count:5d}: {reason}")
    if not args.build:
        return
    if not args.guide or not args.guide.is_file():
        parser.error("--build requires an existing --guide file")
    markers = scan_markers(items)
    if markers:
        for path, marker in markers[:30]:
            print(f"Review possible credential marker {marker!r}: {path}")
        raise SystemExit("Credential marker review required before building")
    with tempfile.TemporaryDirectory(prefix="research_migration_") as temp:
        stage = Path(temp) / NAME
        stage.mkdir()
        rows = []
        for source, rel in items:
            destination = stage / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_symlink():
                destination.symlink_to(os.readlink(source))
                rows.append(f"SYMLINK\t{os.readlink(source)}\t{rel.as_posix()}")
            else:
                shutil.copy2(source, destination)
                rows.append(f"{sha256(destination)}\t{destination.stat().st_size}\t{rel.as_posix()}")
        shutil.copy2(args.guide, stage / "README_MIGRATION.md")
        rows.append(f"{sha256(stage / 'README_MIGRATION.md')}\t{(stage / 'README_MIGRATION.md').stat().st_size}\tREADME_MIGRATION.md")
        shutil.copy2(__file__, stage / "build_research_migration.py")
        rows.append(f"{sha256(stage / 'build_research_migration.py')}\t{(stage / 'build_research_migration.py').stat().st_size}\tbuild_research_migration.py")
        shutil.copy2(HERE / "verify_migration_bundle.py", stage / "verify_migration_bundle.py")
        rows.append(f"{sha256(stage / 'verify_migration_bundle.py')}\t{(stage / 'verify_migration_bundle.py').stat().st_size}\tverify_migration_bundle.py")
        (stage / "MANIFEST.sha256.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        temp_archive = Path(temp) / f"{NAME}.tar.gz"
        with tarfile.open(temp_archive, "w:gz", compresslevel=6, format=tarfile.PAX_FORMAT) as archive:
            archive.add(stage, arcname=NAME, recursive=True)
        output = HERE / temp_archive.name
        shutil.copy2(temp_archive, output)
        (HERE / f"{NAME}.tar.gz.sha256").write_text(
            f"{sha256(output)}  {output.name}\n", encoding="utf-8"
        )
        print(f"Wrote {output} ({output.stat().st_size / 1048576:.1f} MiB)")


if __name__ == "__main__":
    main()
