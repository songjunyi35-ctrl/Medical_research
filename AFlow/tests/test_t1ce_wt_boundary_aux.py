from __future__ import annotations

import torch

from sammed3d.t1ce_wt_boundary_aux import boundary_dice_per_case, boundary_target


def test_single_voxel_radius_one_boundary_is_three_cubed() -> None:
    label = torch.zeros(1, 1, 7, 7, 7)
    label[:, :, 3, 3, 3] = 1
    target = boundary_target(label, radius=1)
    assert int(target.sum()) == 27


def test_solid_cube_boundary_excludes_deep_interior() -> None:
    label = torch.zeros(1, 1, 9, 9, 9)
    label[:, :, 2:7, 2:7, 2:7] = 1
    target = boundary_target(label, radius=1)
    assert target[0, 0, 4, 4, 4] == 0
    assert target[0, 0, 2, 4, 4] == 1
    assert target[0, 0, 1, 4, 4] == 1


def test_boundary_dice_is_one_for_exact_logits() -> None:
    label = torch.zeros(2, 1, 7, 7, 7)
    label[:, :, 2:5, 2:5, 2:5] = 1
    target = boundary_target(label, radius=1)
    logits = torch.where(target > 0, torch.ones_like(target), -torch.ones_like(target))
    torch.testing.assert_close(
        boundary_dice_per_case(logits, target),
        torch.ones(2),
        rtol=0,
        atol=0,
    )


def test_invalid_boundary_radius_is_rejected() -> None:
    try:
        boundary_target(torch.zeros(1, 1, 4, 4, 4), radius=0)
    except ValueError as error:
        assert "radius" in str(error)
    else:
        raise AssertionError("invalid radius was accepted")
