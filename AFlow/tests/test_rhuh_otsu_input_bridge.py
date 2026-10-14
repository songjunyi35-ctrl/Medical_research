import numpy as np
import SimpleITK as sitk
import torch

from scripts.audit_rhuh_otsu_input_bridge import MODEL_SHAPE, otsu_bridge, passed


def synthetic_volume():
    zz, yy, xx = np.meshgrid(
        np.linspace(-1, 1, 32), np.linspace(-1, 1, 32), np.linspace(-1, 1, 32), indexing="ij"
    )
    radius = xx * xx + yy * yy + zz * zz
    array = np.full((32, 32, 32), -4.0, dtype=np.float32)
    support = radius < 0.72
    array[support] = (2.0 + 8.0 * np.exp(-3.0 * radius[support])).astype(np.float32)
    return array


def test_otsu_bridge_is_finite_support_exact_and_deterministic():
    torch.set_num_threads(1)
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    first_image, first_mask, first = otsu_bridge(synthetic_volume())
    second_image, second_mask, second = otsu_bridge(synthetic_volume())
    assert first_image.shape == MODEL_SHAPE and first_mask.shape == MODEL_SHAPE
    assert np.array_equal(first_image, second_image)
    assert np.array_equal(first_mask, second_mask)
    assert first["fit_mask"]["fit_mask_sha256"] == second["fit_mask"]["fit_mask_sha256"]
    assert first["support_exact_after_n4"] and first["output_finite"]
    assert passed(first, np.ones(3))


def test_otsu_bridge_rejects_nonconstant_boundary():
    array = synthetic_volume()
    array[0] = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
    try:
        otsu_bridge(array)
    except ValueError as error:
        assert "background" in str(error)
    else:
        raise AssertionError("nonconstant boundary should fail closed")
