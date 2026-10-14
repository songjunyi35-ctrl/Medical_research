import numpy as np

from scripts.audit_rhuh_no_n4_input_bridge import MODEL_SHAPE, bridge, case_passed


def synthetic_volume():
    array = np.full((24, 20, 16), -3.0, dtype=np.float32)
    values = np.linspace(-2.0, 8.0, 12 * 10 * 8, dtype=np.float32).reshape(12, 10, 8)
    array[6:18, 5:15, 4:12] = values
    return array


def test_bridge_recovers_background_and_standardizes_source_support():
    image, mask, metadata = bridge(synthetic_volume())
    assert image.shape == MODEL_SHAPE
    assert mask.shape == MODEL_SHAPE
    assert image.dtype == np.float32 and mask.dtype == np.uint8
    assert metadata["decoded_background_exact_zero"]
    assert abs(metadata["normalized_positive_mean"]) <= 1e-6
    assert abs(metadata["normalized_positive_population_std"] - 1.0) <= 1e-6
    assert case_passed(metadata, np.array([1.0, 1.0, 1.0]))


def test_bridge_is_byte_deterministic():
    first_image, first_mask, first = bridge(synthetic_volume())
    second_image, second_mask, second = bridge(synthetic_volume())
    assert np.array_equal(first_image, second_image)
    assert np.array_equal(first_mask, second_mask)
    assert first["output_image_sha256"] == second["output_image_sha256"]
    assert first["output_mask_sha256"] == second["output_mask_sha256"]


def test_bridge_rejects_nonconstant_border():
    array = synthetic_volume()
    array[0, :, :] = np.arange(array.shape[1] * array.shape[2], dtype=np.float32).reshape(array.shape[1:])
    try:
        bridge(array)
    except ValueError as error:
        assert "background" in str(error)
    else:
        raise AssertionError("nonconstant boundary should fail closed")
