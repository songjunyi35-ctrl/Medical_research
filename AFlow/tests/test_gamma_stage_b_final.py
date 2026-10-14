from pathlib import Path
from scripts.train_t1ce_wt_gamma_aug_stage_b_final import STAGE_A_DECISION_SHA256, validate_decision

ROOT = Path(__file__).resolve().parents[1]


def test_gamma_stage_a_decision_authorizes_final():
    value = validate_decision(ROOT / "archive/research_records/2026-09-02_full972_gamma_aug_stability/STAGE_A_DECISION.json")
    assert value["full972_final_authorized"] is True
    assert value["gates"]["passed"] == 6
