"""Shared inference-boundary checks used by the active T1ce track.

The former module also contained the retired four-modal manifest, workflow,
and normalization contracts.  Those implementations are preserved under the
retired-multimodal archive; this active module intentionally keeps only the
modality-independent supervision-leak guard required by ``t1ce_binary``.
"""
from __future__ import annotations

import re
from typing import Any, Mapping


FORBIDDEN_SUPERVISION_KEY_ALIASES = frozenset(
    {
        "seg",
        "seg3d",
        "segfile",
        "segpath",
        "segmentation",
        "segmentation3d",
        "segmentationfile",
        "segmentationpath",
        "label",
        "labels",
        "label3d",
        "labelfile",
        "labelfiles",
        "labelmap",
        "labelmaps",
        "labelpath",
        "labelpaths",
        "gt",
        "gt3d",
        "gtfile",
        "gtpath",
        "groundtruth",
        "groundtruth3d",
        "groundtruthfile",
        "groundtruthpath",
    }
)
FORBIDDEN_SUPERVISION_KEY_PREFIXES = ("seg", "label", "gt", "groundtruth")


def _canonical_field_name(key: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(key).casefold())


def _is_supervision_field(key: object) -> bool:
    normalized = _canonical_field_name(key)
    return normalized in FORBIDDEN_SUPERVISION_KEY_ALIASES or normalized.startswith(
        FORBIDDEN_SUPERVISION_KEY_PREFIXES
    )


def assert_no_supervision_fields(value: Any, path: str = "problem") -> None:
    """Recursively reject ground-truth field aliases at an inference boundary."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _is_supervision_field(key):
                raise ValueError(f"{path}.{key} is a forbidden supervision field")
            assert_no_supervision_fields(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_no_supervision_fields(child, f"{path}[{index}]")
