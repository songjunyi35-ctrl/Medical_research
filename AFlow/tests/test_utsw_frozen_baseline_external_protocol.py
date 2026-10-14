from datetime import datetime,timezone
import torch
from scripts import utsw_frozen_baseline_external_protocol as p


def test_models_and_cohort_are_frozen():
    assert list(p.MODELS)==["full972_kd168","gcsam184"]
    assert [p.MODELS[x]["primary_view"] for x in p.MODELS]==["flip8","identity"]
    assert len(p.cohort())==362 and p.SOURCE_FILES==724 and p.SOURCE_BYTES==1_813_086_599


def test_checkpoint_payload_identity_and_state_hashes():
    for row in p.MODELS.values():
        payload=torch.load(row["path"],map_location="cpu",weights_only=False)
        assert payload["schema"]==row["schema"] and payload["epoch"]==row["epoch"]
        assert payload["model_state_sha256"]==row["state_sha256"]


def test_three_hour_gate_boundary():
    assert p.three_hour_gate(datetime(2026,9,6,11,2,39,tzinfo=timezone.utc))
    assert not p.three_hour_gate(datetime(2026,9,6,11,2,40,tzinfo=timezone.utc))
    assert 362*9*2==6516
