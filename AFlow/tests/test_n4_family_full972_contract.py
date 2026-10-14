import json
from pathlib import Path

from scripts import n4_family_full972_protocol as protocol
from scripts import run_n4_family_full972 as runner


def _report(cache: str, *, mismatch: str | None = None):
    history = []
    for epoch in range(2):
        item = {
            "sampler_sha256": f"sampler-{epoch}",
            "augmentation_rng_before_sha256": f"before-{epoch}",
            "augmentation_rng_after_sha256": f"after-{epoch}",
            "first_batch": {"label": "same-label"} if epoch == 0 else None,
        }
        history.append(item)
    if mismatch:
        history[1][mismatch] = "different"
    return {
        "initial_state_sha256": "same-init",
        "sampling_weight_sha256": "same-weights",
        "history": history,
        "teacher_checkpoints": [{"role": "history", "sha256": "h"}, {"role": "ema", "sha256": "e"}],
        "cache_index_sha256": cache,
        "all_gates_passed": True,
    }


def test_full972_roles_and_seeds_are_frozen():
    assert protocol.ROLES == ("history", "ema", "canonical", "no_n4", "otsu")
    assert protocol.SEEDS["history"] == 20261341
    assert protocol.SEEDS["ema"] == 20261342
    assert {protocol.SEEDS[x] for x in ("canonical", "no_n4", "otsu")} == {20261343}
    assert (protocol.EPOCHS, protocol.DRAWS, protocol.STEPS) == (168, 972, 121)


def test_pairing_report_requires_all_epoch_rng_and_sampler(monkeypatch, tmp_path):
    monkeypatch.setattr(protocol, "WORK", tmp_path)
    for role, report in (
        ("canonical", _report("control")),
        ("no_n4", _report("candidate")),
    ):
        path = tmp_path / "formal" / role
        path.mkdir(parents=True)
        (path / "TRAIN_DONE.json").write_text(json.dumps(report))
    value = runner.pairing_report("formal", ["no_n4"])
    assert value["all_pairing_gates_passed"]
    bad = _report("candidate", mismatch="sampler_sha256")
    (tmp_path / "formal" / "no_n4" / "TRAIN_DONE.json").write_text(json.dumps(bad))
    value = runner.pairing_report("formal", ["no_n4"])
    assert not value["all_pairing_gates_passed"]


def test_trainer_has_no_evaluation_loader_and_freezes_single_factor():
    source = (Path(__file__).parents[1] / "scripts/train_n4_family_full972.py").read_text()
    assert "evaluate_" not in source
    assert '0.75 * teachers[0]' in source
    assert '0.25 * teachers[1]' in source
    assert 'base.AuditedWeightedSampler(weights, 16 if args.smoke else p.DRAWS, seed)' in source
    assert 'args.role in p.CANDIDATES and args.role not in eligible' in source


def test_eight_hour_gate_is_only_at_supervisor_stage_start():
    train_source = (Path(__file__).parents[1] / "scripts/train_n4_family_full972.py").read_text()
    run_source = (Path(__file__).parents[1] / "scripts/run_n4_family_full972.py").read_text()
    assert "p.verify_admission(require_time=False)" in train_source
    assert 'p.verify_admission(require_time=phase == "formal")' in run_source
