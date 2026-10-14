from pathlib import Path


def test_queue_order_is_fail_closed():
    text = Path("scripts/queue_utsw_no_n4_paired_external.py").read_text()
    text = text[text.index('family = p.full.RECORD / "FAMILY_READOUT_DONE.json"'):]
    order = [
        "FAMILY_READOUT_DONE.json", "performance_passed", "FULL972_FORMAL_RUNNER_DONE.json",
        "PRECEDENCE", "four_hour_gate", "prepare_utsw", "smoke_utsw", "download_utsw",
        "build_utsw", "evaluate_utsw",
    ]
    positions = [text.index(token) for token in order]
    assert positions == sorted(positions)
