#!/usr/bin/env python3
"""Verify the files and links in an extracted research migration bundle."""

import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "MANIFEST.sha256.tsv"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    if not MANIFEST.is_file():
        raise SystemExit("Missing MANIFEST.sha256.tsv")
    listed = set()
    count = 0
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        expected, size_or_path, relative = line.split("\t", 2)
        if relative in listed:
            raise SystemExit(f"Duplicate manifest path: {relative}")
        listed.add(relative)
        path = ROOT / relative
        if not path.resolve().is_relative_to(ROOT):
            raise SystemExit(f"Path escapes bundle: {relative}")
        if expected == "SYMLINK":
            if not path.is_symlink() or os.readlink(path) != size_or_path:
                raise SystemExit(f"Link mismatch: {relative}")
        elif path.is_symlink() or not path.is_file():
            raise SystemExit(f"Missing or invalid file: {relative}")
        elif path.stat().st_size != int(size_or_path) or digest(path) != expected:
            raise SystemExit(f"Hash/size mismatch: {relative}")
        count += 1
    actual = {
        p.relative_to(ROOT).as_posix()
        for p in ROOT.rglob("*")
        if (p.is_file() or p.is_symlink()) and not any(
            part in {".git", "__pycache__", ".pytest_cache", ".venv"}
            for part in p.relative_to(ROOT).parts
        )
    }
    extra = actual - listed - {"MANIFEST.sha256.tsv"}
    if extra:
        raise SystemExit(f"Unlisted files: {sorted(extra)[:5]}")
    print(f"OK: {count} manifest entries verified")


if __name__ == "__main__":
    main()
