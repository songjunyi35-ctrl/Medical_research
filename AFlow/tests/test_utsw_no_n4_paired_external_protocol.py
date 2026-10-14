from datetime import datetime, timezone
import hashlib

from scripts import utsw_no_n4_paired_external_protocol as p


def test_frozen_cohort_is_complete_and_exact():
    rows = p.cohort()
    assert len(rows) == 362
    assert len({row["subject_id"] for row in rows}) == 362
    assert sum(row["image_bytes"] + row["label_bytes"] for row in rows) == 1_813_086_599
    ids = "\n".join(row["subject_id"] for row in rows) + "\n"
    assert hashlib.sha256(ids.encode()).hexdigest() == p.MANUAL_IDS_SHA


def test_four_hour_gate_is_boundary_exact():
    assert p.four_hour_gate(datetime(2026, 9, 6, 10, 2, 39, tzinfo=timezone.utc))
    assert not p.four_hour_gate(datetime(2026, 9, 6, 10, 2, 40, tzinfo=timezone.utc))


def test_axes_and_accounting_are_frozen():
    assert len(p.AXES) == 8
    assert p.CASES * (1 + len(p.AXES)) * 2 == 6516
    assert p.BOOTSTRAP_SEED == 20261381 and p.BOOTSTRAP_SAMPLES == 10_000
    assert p.sha(p.CANONICAL_BRIDGE_COPY_A) == p.CANONICAL_BRIDGE_COPY_A_SHA
