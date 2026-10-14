#!/usr/bin/env python3
"""Validate the compact root documentation layout and relative Markdown links."""
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROOT_MARKDOWN = {
    "README.md",
    "AGENTS.md",
    "AFLOW_BRATS_HANDOFF_CURRENT.md",
    "AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md",
    "AFLOW_BRATS_LEADERBOARD.md",
    "AFLOW_BRATS_METHOD_CATALOG.md",
    "AFLOW_BRATS_RESULTS.md",
    "AFLOW_BRATS_WEEKLY_REPORT.md",
    "EXPERIMENT_IDEA_TEMPLATE.md",
}


def validate_layout(root: Path = ROOT) -> dict:
    actual = {path.name for path in root.glob("*.md") if path.is_file()}
    unexpected = sorted(actual - ALLOWED_ROOT_MARKDOWN)
    missing = sorted(ALLOWED_ROOT_MARKDOWN - actual)
    if unexpected or missing:
        raise ValueError(
            f"root Markdown layout mismatch; unexpected={unexpected}, missing={missing}"
        )
    root_orig = sorted(path.name for path in root.glob("*.orig"))
    if root_orig:
        raise ValueError(f"legacy .orig files remain at root: {root_orig}")
    broken = []
    checked = 0
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for document in sorted(root.glob("*.md")):
        for target in pattern.findall(document.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#")):
                continue
            relative = target.split("#", 1)[0]
            if not relative:
                continue
            checked += 1
            if not (document.parent / relative).exists():
                broken.append(f"{document.name} -> {target}")
    if broken:
        raise ValueError("broken root-document links: " + "; ".join(broken))
    return {
        "root_markdown_files": len(actual),
        "relative_links_checked": checked,
        "root_orig_files": 0,
    }


def main() -> None:
    result = validate_layout()
    print(json.dumps({"status": "passed", **result}, indent=2))


if __name__ == "__main__":
    main()
