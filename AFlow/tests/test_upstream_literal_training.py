from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest
import torch

from scripts import train_sammed3d_t1ce_wt_upstream_literal_va as upstream


def _args(**overrides):
    values = dict(
        mode="formal",
        epochs=200,
        batch_size=12,
        accumulation_steps=20,
        encoder_lr=8e-4,
        head_lr=8e-5,
        weight_decay=0.1,
        num_workers=12,
    )
    values.update(overrides)
    return Namespace(**values)


def test_formal_two_gpu_contract_preserves_source_loader_and_nominal_batch() -> None:
    args = _args()
    upstream.validate_args(args, world=2)
    assert 2 * args.batch_size * args.accumulation_steps == 480
    assert upstream.TRAIN_SEED == 2023
    assert 972 // (2 * args.batch_size) + 1 == 41


def test_smoke_preserves_formal_gradient_accumulation() -> None:
    upstream.validate_args(_args(mode="smoke"), world=2)


@pytest.mark.parametrize(
    ("field", "value"),
    [("batch_size", 6), ("accumulation_steps", 1), ("encoder_lr", 1e-4)],
)
def test_formal_contract_fails_closed(field, value) -> None:
    with pytest.raises(ValueError, match="upstream-literal"):
        upstream.validate_args(_args(**{field: value}), world=2)


def test_uniform_sampler_uses_upstream_raw_logit_point_five_threshold() -> None:
    target = torch.zeros(1, 1, 2, 2, 2)
    target[0, 0, 0, 0, 0] = 1
    logits = torch.zeros_like(target)
    logits[0, 0, 0, 0, 0] = 0.25  # positive under raw>0, negative under upstream >.5
    np.random.seed(3)
    point, label = upstream._sample_uniform_xor(logits, target, threshold=0.5)
    assert tuple(point[0, 0].tolist()) == (0, 0, 0)
    assert int(label[0, 0]) == 1


class _TinyImageEncoder(torch.nn.Module):
    def forward(self, image):
        return image.mean(dim=(-3, -2, -1), keepdim=True)


class _TinyPromptEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.point_history = []

    def forward(self, *, points, boxes, masks):
        self.point_history.append(points)
        batch = masks.shape[0]
        scalar = masks.mean(dim=(-3, -2, -1), keepdim=True)
        return scalar, scalar

    def get_dense_pe(self):
        return torch.zeros(1)


class _TinyMaskDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(-0.25))

    def forward(self, **kwargs):
        batch = kwargs["image_embeddings"].shape[0]
        return self.value.expand(batch, 1, 2, 2, 2), torch.zeros(batch, 1)


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = _TinyImageEncoder()
        self.prompt_encoder = _TinyPromptEncoder()
        self.mask_decoder = _TinyMaskDecoder()


def test_interaction_is_11_sum_with_random_and_final_no_point(monkeypatch) -> None:
    monkeypatch.setattr(upstream.engine, "CROP_SIZE", 8)
    model = _TinyModel()
    image = torch.ones(1, 1, 8, 8, 8)
    target = torch.zeros_like(image)
    target[:, :, 2:6, 2:6, 2:6] = 1
    loss_fn = lambda logits, gt: (logits - gt).square().mean()
    np.random.seed(7)
    logits, total_loss, record = upstream.upstream_train_interaction(
        model=model,
        image=image,
        target=target,
        loss_fn=loss_fn,
        random_insert=4,
    )
    assert logits.shape == target.shape
    assert record["interactions"] == 11
    assert record["no_point_iterations"] == [4, 10]
    assert len(model.prompt_encoder.point_history) == 11
    assert model.prompt_encoder.point_history[4] is None
    assert model.prompt_encoder.point_history[10] is None
    assert all(
        points is None or points[0].shape[1] == 1
        for points in model.prompt_encoder.point_history
    )
    expected_one = loss_fn(torch.full_like(target, -0.25), target)
    assert torch.allclose(total_loss, 11 * expected_one)


def test_current_monai_dicece_accepts_upstream_integer_gt_via_loss_boundary(monkeypatch) -> None:
    monkeypatch.setattr(upstream.engine, "CROP_SIZE", 8)
    model = _TinyModel()
    image = torch.ones(1, 1, 8, 8, 8)
    target = torch.zeros_like(image, dtype=torch.long)
    target[:, :, 2:6, 2:6, 2:6] = 1
    loss_fn = upstream.generic.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
    _, loss, _ = upstream.upstream_train_interaction(
        model=model, image=image, target=target, loss_fn=loss_fn, random_insert=4
    )
    assert bool(torch.isfinite(loss))


def test_threshold_replacement_keeps_length_and_replaces_small_case() -> None:
    class Dummy(torch.utils.data.Dataset):
        def __len__(self):
            return 3

        def __getitem__(self, index):
            label = torch.zeros(1, 16, 16, 16)
            label.view(-1)[: [10, 2000, 3000][index]] = 1
            return {"index": index, "label": label}

    wrapped = upstream.ForegroundThresholdReplacement(Dummy(), [10, 2000, 3000], 1000)
    np.random.seed(0)
    item = wrapped[0]
    assert len(wrapped) == 3
    assert item["threshold_replaced"] is True
    assert item["threshold_rejections"] >= 1
    assert item["selected_index"] in {1, 2}
