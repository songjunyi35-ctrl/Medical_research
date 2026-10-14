from __future__ import annotations

from pathlib import Path

import scripts.train_t1ce_wt_lawa6_stage_a as base
from scripts.train_t1ce_wt_lawa6_stage_b_final import (
    ARM,
    CHECKPOINT_SCHEMA,
    FULL972_SHA256,
    SCHEMA,
    SEED,
    STAGE_A_DECISION_SHA256,
    configure_frozen_base,
    validate_stage_a_decision,
)


ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "archive/research_records/2026-09-02_full972_lawa6_stability/STAGE_A_DECISION.json"


def test_stage_a_decision_authorizes_final_and_is_hash_bound() -> None:
    value = validate_stage_a_decision(DECISION)
    assert value["full972_final_authorized"] is True
    assert value["promotion_gates"]["passed"] == 10


def test_stage_b_specializes_frozen_base_without_changing_method() -> None:
    wrapper = ROOT / "scripts/train_t1ce_wt_lawa6_stage_b_final.py"
    names = (
        "SCHEMA",
        "CHECKPOINT_SCHEMA",
        "ARMS",
        "ARM_SEED",
        "FIT_SHA256",
        "SPLIT_SHA256",
        "FIT_CASES",
        "FIT_SUBJECTS",
        "__file__",
    )
    original = {name: getattr(base, name) for name in names}
    try:
        configure_frozen_base(wrapper)
        assert base.SCHEMA == SCHEMA
        assert base.CHECKPOINT_SCHEMA == CHECKPOINT_SCHEMA
        assert base.ARMS == (ARM,)
        assert base.ARM_SEED == {ARM: SEED}
        assert base.FIT_SHA256 == FULL972_SHA256
        assert base.SPLIT_SHA256 == STAGE_A_DECISION_SHA256
        assert base.FIT_CASES == 972
        assert base.FIT_SUBJECTS == 444
        assert Path(base.__file__).resolve() == wrapper.resolve()
    finally:
        for name, value in original.items():
            setattr(base, name, value)
