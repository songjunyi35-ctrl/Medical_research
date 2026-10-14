from __future__ import annotations

from scripts.verify_aflow_brats_method_catalog import ROOT, validate_catalog


def test_current_method_catalog_has_complete_plain_language_cards() -> None:
    assert validate_catalog(ROOT / "AFLOW_BRATS_METHOD_CATALOG.md") >= 10


def test_agents_requires_plain_language_catalog_updates() -> None:
    instructions = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "Explain research for a beginner" in instructions
    assert "AFLOW_BRATS_METHOD_CATALOG.md" in instructions
    assert "EXPERIMENT_IDEA_TEMPLATE.md" in instructions
    assert "AFLOW_BRATS_HANDOFF_CURRENT.md" in instructions
