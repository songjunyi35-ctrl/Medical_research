#!/usr/bin/env python3
"""Apply frozen amendment01 to the not-yet-run consistency trainer."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts/train_t1ce_wt_flip_equivariance_consistency.py"
EXPECTED_BEFORE = "5faf3e4818a03757120ea75b73c3a788ba6a3e7894d104ebf41a6424fe6cb6b0"
AMENDMENT_SHA256 = "80604f241a6a220bee667019721cd76c5785f88a56f8ae17cc6fe99f9ea84ad4"


def once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected one {label}; found {text.count(old)}")
    return text.replace(old, new, 1)


def main() -> None:
    source = TARGET.read_text(encoding="utf-8")
    if hashlib.sha256(source.encode()).hexdigest() != EXPECTED_BEFORE:
        raise RuntimeError("trainer differs from frozen pre-amendment source")
    source = once(
        source,
        'PREREG_SHA256 = "7ae74cd1891a4a67efa064ca2dffd4560064657661fd4d824902aff02bac5809"',
        'PREREG_SHA256 = "7ae74cd1891a4a67efa064ca2dffd4560064657661fd4d824902aff02bac5809"\nAMENDMENT_SHA256 = "80604f241a6a220bee667019721cd76c5785f88a56f8ae17cc6fe99f9ea84ad4"',
        "amendment constant",
    )
    source = once(
        source,
        "denominator = original_probability.flatten(1).sum(dim=1) + reflected_probability.flatten(1).sum(dim=1)",
        "denominator = original_probability.square().flatten(1).sum(dim=1) + reflected_probability.square().flatten(1).sum(dim=1)",
        "squared soft-Dice denominator",
    )
    source = once(
        source,
        "        (args.prereg, PREREG_SHA256),\n        (args.native_artifact, prior.NATIVE_ARTIFACT_SHA256),",
        "        (args.prereg, PREREG_SHA256),\n        (args.amendment, AMENDMENT_SHA256),\n        (args.native_artifact, prior.NATIVE_ARTIFACT_SHA256),",
        "amendment input binding",
    )
    source = once(
        source,
        '"documents": {"idea_sha256": IDEA_SHA256, "prereg_sha256": PREREG_SHA256, "split_manifest_sha256": SPLIT_MANIFEST_SHA256},',
        '"documents": {"idea_sha256": IDEA_SHA256, "prereg_sha256": PREREG_SHA256, "amendment_sha256": AMENDMENT_SHA256, "split_manifest_sha256": SPLIT_MANIFEST_SHA256},',
        "checkpoint amendment binding",
    )
    source = once(
        source,
        '    parser.add_argument("--prereg", type=Path, default=record / "PREREG.md")\n',
        '    parser.add_argument("--prereg", type=Path, default=record / "PREREG.md")\n    parser.add_argument("--amendment", type=Path, default=record / "AMENDMENT_01.md")\n',
        "amendment CLI",
    )
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=TARGET.parent, delete=False) as handle:
        handle.write(source)
        temporary = Path(handle.name)
    temporary.replace(TARGET)
    print("applied_flip_consistency_amendment01")


if __name__ == "__main__":
    main()
