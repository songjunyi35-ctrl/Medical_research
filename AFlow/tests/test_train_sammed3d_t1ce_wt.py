from __future__ import annotations

import pytest
import torch
from torch import nn

from scripts import train_sammed3d_t1ce_wt as trainer


def _contract_argv(*extra: str) -> list[str]:
    return [
        "--train-jsonl",
        "train.jsonl",
        "--val-jsonl",
        "val.jsonl",
        "--output-dir",
        "output",
        "--num-clicks",
        "3",
        "--click-method",
        "random",
        *extra,
    ]


def test_contract_keeps_decoder_only_reproduction_frozen() -> None:
    args = trainer.parse_contract_args(_contract_argv())

    assert args.strategy == "decoder_only"
    assert args.encoder_qv_lora_rank == 0
    assert args.encoder_qv_lora_alpha == trainer.peft.DEFAULT_LORA_ALPHA


def test_contract_allows_explicit_encoder_qv_lora() -> None:
    args = trainer.parse_contract_args(
        _contract_argv(
            "--encoder-qv-lora-rank",
            "4",
            "--encoder-qv-lora-alpha",
            "8",
        )
    )

    assert args.strategy == "decoder_only"
    assert args.encoder_qv_lora_rank == 4
    assert args.encoder_qv_lora_alpha == 8.0


def test_contract_accepts_whole_resize_and_train_only_augmentation() -> None:
    args = trainer.parse_contract_args(
        _contract_argv(
            "--spatial-preprocessing",
            "whole_resize",
            "--train-augmentation",
            "affine_elastic_noise",
        )
    )

    assert args.spatial_preprocessing == "whole_resize"
    assert args.train_augmentation == "affine_elastic_noise"


def test_wrapper_only_preprocessing_args_are_not_forwarded_to_generic() -> None:
    argv = _contract_argv(
        "--spatial-preprocessing",
        "whole_resize",
        "--train-augmentation=affine_elastic_noise",
        "--epochs",
        "2",
    )

    generic_argv = trainer.strip_wrapper_only_args(argv)

    assert "--spatial-preprocessing" not in generic_argv
    assert "--train-augmentation=affine_elastic_noise" not in generic_argv
    assert generic_argv[-2:] == ["--epochs", "2"]


def test_protocol_records_encoder_qv_lora_configuration(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    checkpoint = tmp_path / "base.pth"
    for path in (train, validation, checkpoint):
        path.write_bytes(b"test")
    args = trainer.parse_contract_args(
        [
            "--train-jsonl",
            str(train),
            "--val-jsonl",
            str(validation),
            "--output-dir",
            str(tmp_path / "output"),
            "--checkpoint",
            str(checkpoint),
            "--num-clicks",
            "3",
            "--click-method",
            "random",
            "--encoder-qv-lora-rank",
            "4",
            "--encoder-qv-lora-alpha",
            "8",
        ]
    )

    image_encoder = trainer.protocol_record(args)["optimization"]["image_encoder"]

    assert image_encoder == {
        "update": "qv_lora",
        "rank": 4,
        "alpha": 8.0,
        "blocks": list(trainer.peft.DEFAULT_LORA_BLOCKS),
        "projections": ["q", "v"],
    }


def test_protocol_records_whole_resize_and_augmentation(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    checkpoint = tmp_path / "base.pth"
    for path in (train, validation, checkpoint):
        path.write_bytes(b"test")
    args = trainer.parse_contract_args(
        [
            "--train-jsonl",
            str(train),
            "--val-jsonl",
            str(validation),
            "--output-dir",
            str(tmp_path / "output"),
            "--checkpoint",
            str(checkpoint),
            "--num-clicks",
            "3",
            "--click-method",
            "random",
            "--spatial-preprocessing",
            "whole_resize",
            "--train-augmentation",
            "affine_elastic_noise",
        ]
    )

    preprocessing = trainer.protocol_record(args)["preprocessing"]

    assert preprocessing["geometry"]["name"] == "whole_resize"
    assert preprocessing["geometry"]["uses_gt_for_localization"] is False
    assert preprocessing["augmentation"]["applies_to"] == "train split only"
    assert preprocessing["augmentation"]["validation_augmentation"] is False
    assert "no batch re-normalization" in preprocessing["normalization"]


def test_contract_rejects_augmented_overfit_validation() -> None:
    with pytest.raises(ValueError, match="validation must remain unaugmented"):
        trainer.parse_contract_args(
            _contract_argv(
                "--train-augmentation",
                "affine_elastic_noise",
                "--overfit",
            )
        )


@pytest.mark.parametrize(
    "extra",
    [
        ("--encoder-qv-lora-rank", "-1"),
        ("--encoder-qv-lora-rank", "4", "--encoder-qv-lora-alpha", "0"),
        ("--target-token",),
        ("--strategy", "full"),
    ],
)
def test_contract_rejects_unsafe_or_out_of_scope_encoder_modes(
    extra: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        trainer.parse_contract_args(_contract_argv(*extra))


def test_torchio_sample_normalization_matches_reference() -> None:
    image = torch.tensor([[[[[0.0, 1.0, 2.0]]]]])
    normalized = trainer.torchio_z_normalize_batch(image)
    expected = torch.tensor([[[[[-2.1213202, -0.7071068, 0.7071068]]]]])
    assert torch.allclose(normalized, expected)


def test_click_threshold_is_raw_logit_zero(monkeypatch) -> None:
    monkeypatch.setattr(trainer.np.random, "randint", lambda size: size - 1)
    logits = torch.tensor([[[[[0.0, 0.1]]]]])
    target = torch.tensor([[[[[1.0, 0.0]]]]])
    point, label = trainer.sample_uniform_error_click(logits, target)
    assert point.tolist() == [[[0, 0, 1]]]
    assert label.tolist() == [[0]]


class _PromptRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.counts = []

    def forward(self, points, boxes, masks):
        self.counts.append(int(points[0].shape[1]))
        batch = points[0].shape[0]
        return (
            torch.zeros(batch, points[0].shape[1], 2),
            torch.zeros(batch, 2, 1, 1, 1),
        )

    def get_dense_pe(self):
        return torch.zeros(1, 2, 1, 1, 1)


class _MaskDecoder(nn.Module):
    def forward(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output,
    ):
        batch = image_embeddings.shape[0]
        return torch.zeros(batch, 1, 1, 1, 1, requires_grad=True), None


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Identity()
        self.prompt_encoder = _PromptRecorder()
        self.mask_decoder = _MaskDecoder()


class _ZeroLoss(nn.Module):
    def forward(self, prediction, target):
        return prediction.sum() * 0


def test_interaction_accumulates_all_three_points(monkeypatch) -> None:
    clicks = iter(
        [
            (torch.tensor([[[0, 0, 0]]]), torch.tensor([[1]])),
            (torch.tensor([[[0, 0, 1]]]), torch.tensor([[0]])),
            (torch.tensor([[[0, 1, 0]]]), torch.tensor([[1]])),
        ]
    )
    monkeypatch.setattr(
        trainer, "sample_uniform_error_click", lambda *args: next(clicks)
    )
    model = _Model()
    trainer.cumulative_interaction_forward(
        model,
        torch.zeros(1, 1, 4, 4, 4),
        torch.ones(1, 1, 4, 4, 4),
        _ZeroLoss(),
        num_clicks=3,
        click_method="random",
        train_image_encoder=False,
    )
    assert model.prompt_encoder.counts == [1, 2, 3]
