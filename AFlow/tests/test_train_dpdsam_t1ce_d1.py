import json

import numpy as np
import pytest
import torch
from torch import nn

from sammed3d import dpdsam_binary_model as binary_model
from scripts import train_dpdsam_t1ce_b0 as b0_runner
from scripts import train_dpdsam_t1ce_d1 as d1_runner


def test_sklearn_kmeans_plus_plus_is_fixed_and_nearest_centroid_consistent() -> None:
    descriptors = np.concatenate(
        [
            np.full((3, 3), value, dtype=np.float32)
            + np.arange(9, dtype=np.float32).reshape(3, 3) * 1e-3
            for value in (0.0, 10.0, 20.0, 30.0)
        ],
        axis=0,
    )
    first = d1_runner.fit_latent_domains(descriptors, domain_count=4, seed=17)
    second = d1_runner.fit_latent_domains(descriptors, domain_count=4, seed=17)

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2]["init"] == "k-means++"
    assert first[2]["n_init"] == 10
    assert first[2]["algorithm"] == "lloyd"
    nearest = np.square(descriptors[:, None] - first[0][None]).sum(-1).argmin(1)
    assert np.array_equal(first[1], nearest)


def test_domain_artifact_round_trip_binds_sources_and_contains_no_labels(tmp_path) -> None:
    descriptors = np.arange(24, dtype=np.float32).reshape(8, 3)
    centroids, labels, kmeans = d1_runner.fit_latent_domains(
        descriptors, domain_count=4, seed=19
    )
    sources = {
        "train_manifest_sha256": "train-sha",
        "split_manifest_sha256": "split-sha",
        "source_checkpoint_sha256": "checkpoint-sha",
    }
    written = d1_runner.write_domain_artifact(
        tmp_path,
        case_ids=[f"BraTS-GLI-{index:05d}-100" for index in range(8)],
        descriptors=descriptors,
        centroids=centroids,
        pseudo_labels=labels,
        kmeans=kmeans,
        sources=sources,
    )
    restored, pseudo_by_case, loaded = d1_runner.load_domain_artifact(
        tmp_path, expected_sources=sources
    )

    assert np.array_equal(restored, centroids)
    assert list(pseudo_by_case.values()) == labels.tolist()
    assert loaded["array_sha256"] == written["array_sha256"]
    metadata = json.loads(
        (tmp_path / "fit_only_latent_domains.json").read_text(encoding="utf-8")
    )
    assert metadata["fit_only"] is True
    assert metadata["reads_validation_images"] is False
    assert metadata["reads_any_labels"] is False
    with np.load(tmp_path / "fit_only_latent_domains.npz", allow_pickle=False) as arrays:
        assert set(arrays.files) == {
            "case_ids",
            "descriptors",
            "centroids",
            "pseudo_labels",
        }

    with pytest.raises(ValueError, match="source/configuration mismatch"):
        d1_runner.load_domain_artifact(
            tmp_path, expected_sources={**sources, "split_manifest_sha256": "changed"}
        )


def _dynamic_output() -> binary_model.DynamicBinarySAMOutput:
    logits = torch.zeros(2, 1, 2, 2, 2, requires_grad=True)
    domain_logits = torch.tensor(
        [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]], requires_grad=True
    )
    return binary_model.DynamicBinarySAMOutput(
        logits=logits,
        probabilities=torch.sigmoid(logits),
        iou_predictions=torch.zeros(2, 1),
        domain_logits=domain_logits,
        domain_affinities=torch.softmax(domain_logits, dim=-1),
        pseudo_domain_ids=torch.tensor([0, 1]),
        global_descriptor=torch.zeros(2, 3),
        prompt_prefix=torch.zeros(2, 1, 3),
    )


def test_phase1_freezes_only_cdr_and_keeps_adapter_ce_active() -> None:
    target = torch.zeros(2, 1, 2, 2, 2)
    phase1_loss, phase1 = d1_runner.d1_loss(
        _dynamic_output(), target, epoch=1, warmup_epochs=50, adapter_beta=0.1
    )
    later_loss, later = d1_runner.d1_loss(
        _dynamic_output(), target, epoch=51, warmup_epochs=50, adapter_beta=0.1
    )

    assert phase1["effective_adapter_beta"] == 0.1
    assert phase1_loss.item() == pytest.approx(
        phase1["segmentation"] + 0.1 * phase1["adapter_ce"]
    )
    assert later["effective_adapter_beta"] == 0.1
    assert later_loss.item() == pytest.approx(
        later["segmentation"] + 0.1 * later["adapter_ce"]
    )


def test_adapter_ce_uses_cached_fit_pseudo_label_when_supplied() -> None:
    output = _dynamic_output()
    target = torch.zeros(2, 1, 2, 2, 2)
    cached_ids = torch.tensor([1, 0])
    _, parts = d1_runner.d1_loss(
        output,
        target,
        epoch=51,
        warmup_epochs=50,
        adapter_beta=0.1,
        pseudo_domain_ids=cached_ids,
    )
    expected = torch.nn.functional.cross_entropy(output.domain_logits, cached_ids)
    assert parts["adapter_ce"] == pytest.approx(float(expected.detach()))


def test_d1_checkpoint_selection_contract_is_model_grid() -> None:
    metrics = {
        "model_grid": {"dsc": 0.61},
        "source_grid": {"dsc": 0.93},
    }
    assert b0_runner.validation_selection_score(metrics) == pytest.approx(0.61)
    assert d1_runner.BEST_CHECKPOINT_NAME == "best_model_grid_dsc.pth"


class _FakeD1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sam_model = nn.Linear(3, 2)
        self.domain_registry = nn.Linear(2, 2, bias=False)
        self.domain_adapter = nn.Linear(2, 4)
        self.prompt_generator = nn.Linear(4, 3)


def _manifest() -> dict:
    model = {
        "schema": binary_model.D1_SCHEMA,
        "adapter_config": {"schema_version": 1, "encoder_qv_lora": {"enabled": False}},
    }
    return {
        "schema": d1_runner.TRAINING_SCHEMA,
        "schema_version": d1_runner.TRAINING_SCHEMA_VERSION,
        "args": {"seed": 20260722, "prompt_tokens": 1, "domain_count": 4},
        "model": model,
        "latent_domains": {
            "array_sha256": "array-sha",
            "metadata_sha256": "metadata-sha",
        },
        "optimizer": {
            "learning_rate": 4e-4,
            "weight_decay": 0.1,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "amsgrad": False,
        },
    }


def _payload(
    model: _FakeD1,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    manifest: dict,
) -> dict:
    def cloned(module: nn.Module) -> dict[str, torch.Tensor]:
        return {name: value.clone() for name, value in module.state_dict().items()}

    return {
        "training_schema": d1_runner.TRAINING_SCHEMA,
        "training_schema_version": d1_runner.TRAINING_SCHEMA_VERSION,
        "model_state_dict": cloned(model.sam_model),
        "dpdsam_dynamic_state_dict": {
            "domain_registry": cloned(model.domain_registry),
            "domain_adapter": cloned(model.domain_adapter),
            "prompt_generator": cloned(model.prompt_generator),
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": None,
        "epoch": 1,
        "history": [{"epoch": 1}],
        "best": {"dsc": 0.2, "epoch": 1},
        "rng_state": b0_runner.capture_rng_state(generator),
        "implementation_manifest": manifest,
        "implementation_manifest_sha256": b0_runner.canonical_json_sha256(manifest),
        "dpdsam_binary_config": manifest["model"],
        "adapter_config": manifest["model"]["adapter_config"],
    }


def test_d1_resume_restores_dynamic_state_and_rejects_hash_drift() -> None:
    torch.manual_seed(31)
    model = _FakeD1()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)
    generator = torch.Generator().manual_seed(32)
    manifest = _manifest()
    payload = _payload(model, optimizer, generator, manifest)
    saved_adapter = {
        key: value.clone() for key, value in model.domain_adapter.state_dict().items()
    }
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    epoch, history, _ = d1_runner.restore_training_checkpoint(
        payload,
        model=model,  # type: ignore[arg-type]
        optimizer=optimizer,
        scaler=None,
        generator=generator,
        implementation_manifest=manifest,
    )
    assert epoch == 1
    assert history == [{"epoch": 1}]
    for name, value in model.domain_adapter.state_dict().items():
        assert torch.equal(value, saved_adapter[name])

    changed = json.loads(json.dumps(manifest))
    changed["latent_domains"]["array_sha256"] = "changed"
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        d1_runner.validate_resume_payload(payload, changed)
