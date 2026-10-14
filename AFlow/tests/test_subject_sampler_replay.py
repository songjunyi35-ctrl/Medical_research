import hashlib

import numpy as np
import pytest
import torch

from scripts.audit_subject_sampler_replay import independent_weights, replay


def fixture():
    cases = list('abcde')
    subjects = dict(zip(cases, ['x', 'x', 'x', 'y', 'z']))
    volumes = dict(zip(cases, [10, 100, 1000, 10000, 100000]))
    return cases, subjects, volumes


def test_independent_formula_subject_mass_and_ratio():
    cases, subjects, volumes = fixture()
    weights = independent_weights(cases, subjects, volumes)
    for subject in set(subjects.values()):
        slots = [i for i, c in enumerate(cases) if subjects[c] == subject]
        assert float(weights['subject_balanced'][slots].sum()) == pytest.approx(1)
    assert float(weights['subject_balanced'][0] / weights['subject_balanced'][1]) == pytest.approx(float(weights['control'][0] / weights['control'][1]))


def test_replay_keeps_rng_across_epochs_and_drops_last_four():
    cases, subjects, volumes = fixture()
    weights = independent_weights(cases, subjects, volumes)['control']
    actual = replay(weights, 42, 2, cases, subjects, volumes)
    generator = torch.Generator().manual_seed(42)
    for epoch in range(2):
        indices = torch.multinomial(weights, 972, replacement=True, generator=generator).numpy()
        expected = hashlib.sha256(np.asarray(indices[:968], dtype=np.int64).tobytes()).hexdigest()
        assert actual[epoch]['sampler_sha256'] == expected
        assert actual[epoch]['sampler_sha256'] != hashlib.sha256(indices.tobytes()).hexdigest()
    assert actual[0]['sampler_sha256'] != actual[1]['sampler_sha256']
    assert actual == replay(weights, 42, 2, cases, subjects, volumes)


def test_invalid_volume_rejected():
    with pytest.raises(ValueError):
        independent_weights(['a'], {'a': 'x'}, {'a': 0})
