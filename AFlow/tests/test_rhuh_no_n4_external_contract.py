from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from scripts import rhuh_no_n4_external_protocol as protocol
from scripts.build_rhuh_no_n4_external_cache import integer_labels
from scripts.evaluate_rhuh_no_n4_external import bootstrap, summarize


def rows():
    result = []
    for subject in range(40):
        for timepoint in (1, 2):
            result.append({
                "case_id": f"RHUH-{subject + 1:04d}_{timepoint}",
                "subject_id": f"RHUH-{subject + 1:04d}",
                "dice": 0.7 + subject / 1000 + timepoint / 10000,
                "point_hit": (subject + timepoint) % 5 != 0,
                "predicted_voxels": 10,
                "target_voxels": 11,
            })
    return result


def test_two_hour_gate_is_exact_and_timezone_aware():
    assert protocol.two_hour_gate(protocol.DEADLINE - timedelta(hours=2))
    assert not protocol.two_hour_gate(protocol.DEADLINE - timedelta(hours=2) + timedelta(seconds=1))
    with pytest.raises(ValueError):
        protocol.two_hour_gate(protocol.DEADLINE.replace(tzinfo=None))


def test_rhuh_label_union_accepts_published_numbering_variants():
    for values in ([0, 1, 2, 3], [0, 1, 2, 4], [0, 2]):
        array = np.asarray(values, dtype=np.float32).reshape(1, 1, -1)
        assert set(np.unique(integer_labels(array))) == set(values)
    with pytest.raises(ValueError):
        integer_labels(np.asarray([0, 5], dtype=np.float32).reshape(1, 1, -1))
    with pytest.raises(ValueError):
        integer_labels(np.asarray([0, 1.5], dtype=np.float32).reshape(1, 1, -1))


def test_summary_and_patient_cluster_bootstrap_are_fixed():
    data = rows()
    summary = summarize(data)
    assert summary["cases"] == 80 and summary["subjects"] == 40
    first = bootstrap(data)
    second = bootstrap(data)
    assert first == second
    assert first["unit"] == "patient_cluster_all_two_timepoints"
    assert first["samples"] == 10_000 and first["seed"] == 20261361


def test_external_source_freezes_flip8_and_no_repeat():
    source = (Path(__file__).parents[1] / "scripts/evaluate_rhuh_no_n4_external.py").read_text()
    assert "for axes in p.AXES" in source
    assert "forwards != 640" in source
    assert 'prediction = logits > 0' in source
    assert "refusing repeated RHUH external evaluation" in source
    queue = (Path(__file__).parents[1] / "scripts/queue_rhuh_no_n4_external.py").read_text()
    assert 'decisions["no_n4"]["performance_passed"] is not True' in queue
    assert "prepare_rhuh_no_n4_external.py" in queue
