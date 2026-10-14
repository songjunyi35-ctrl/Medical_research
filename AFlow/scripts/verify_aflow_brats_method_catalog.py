#!/usr/bin/env python3
"""Verify the concise SegResNet catalog and its archived full history."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "AFLOW_BRATS_METHOD_CATALOG.md"
REQUIRED_FIELDS = (
    "**一句话：**",
    "**出发点：**",
    "**具体做法：**",
    "**结果：**",
    "**是否支持直觉：**",
    "**代价/下一步：**",
)


def validate_catalog(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("# AFlow BraTS 单T1ce SegResNet方法目录\n"):
        raise ValueError("unexpected method catalog title")
    if "## 先理解我们在做什么" not in text or "## 一页总览" not in text:
        raise ValueError("catalog lacks SegResNet introduction or overview")
    if "## 其他已试方向与负结果" not in text:
        raise ValueError("catalog lacks negative-result index")
    for marker in ("固定纯模型基线", "0.740032", "0.759179", "128³", "原生1mm", "0.775891"):
        if marker not in text:
            raise ValueError(f"catalog lacks baseline/grid/leader context: {marker}")
    if "## 方法1：师兄路线基础模型复现" in text:
        raise ValueError("historical interactive method cannot lead the SegResNet catalog")
    matches = list(re.finditer(r"^## 方法\d+：.+$", text, flags=re.MULTILINE))
    if len(matches) < 5:
        raise ValueError("catalog lacks the main SegResNet method cards")
    numbers = [int(re.search(r"方法(\d+)：", match.group(0)).group(1)) for match in matches]
    if numbers != list(range(1, len(matches) + 1)):
        raise ValueError("SegResNet method numbering is not sequential from 1")
    end_of_completed = text.find("\n## 其他已试方向与负结果")
    if end_of_completed < 0:
        raise ValueError("catalog lacks negative-result section")
    for index, match in enumerate(matches):
        start = match.start()
        next_start = matches[index + 1].start() if index + 1 < len(matches) else end_of_completed
        section = text[start:next_start]
        missing = [field for field in REQUIRED_FIELDS if field not in section]
        if missing:
            raise ValueError(f"{match.group(0)} lacks fields: {missing}")
        if "archive/" not in section:
            raise ValueError(f"{match.group(0)} lacks evidence link")
    if "EXPERIMENT_IDEA_TEMPLATE.md" not in text:
        raise ValueError("catalog does not link the experiment idea template")
    receipt = ROOT / "archive/reviews/2026-10-13_segresnet_method_catalog_restructure/RECEIPT.json"
    if not receipt.is_file():
        raise ValueError("full pre-rewrite catalog archive receipt is missing")
    saved = json.loads(receipt.read_text(encoding="utf-8"))
    original = ROOT / saved["original_exact_copy"]
    if not original.is_file() or hashlib.sha256(original.read_bytes()).hexdigest() != saved["original_exact_copy_sha256"]:
        raise ValueError("full historical catalog archive changed")
    browseable = ROOT / saved["browseable_historical_copy"]
    if not browseable.is_file() or hashlib.sha256(browseable.read_bytes()).hexdigest() != saved["browseable_copy_sha256"]:
        raise ValueError("browseable historical catalog archive changed")
    return len(matches)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args()
    count = validate_catalog(args.catalog)
    print(json.dumps({"status": "passed", "completed_method_cards": count}, indent=2))


if __name__ == "__main__":
    main()
