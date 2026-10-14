from __future__ import annotations

from scripts.verify_aflow_brats_doc_layout import validate_layout


def test_root_documentation_layout_and_links_are_clean() -> None:
    result = validate_layout()
    assert result["root_markdown_files"] == 9
    assert result["root_orig_files"] == 0
