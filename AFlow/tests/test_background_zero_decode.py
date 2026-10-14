import numpy as np
import pytest
from sammed3d.background_zero_decode import decode_background_zero


@pytest.mark.parametrize("scale,offset", [(1., 0.), (7., -3.), (0.125, -12.)])
def test_affine_encoding_preserves_signal_support(scale, offset):
    original = np.zeros((12, 14, 16), dtype=np.float32)
    original[3:-3, 3:-3, 3:-3] = np.linspace(1, 8, 6*8*10).reshape(6,8,10)
    encoded = (original / scale + offset).astype(np.float32)
    restored, meta = decode_background_zero(encoded)
    assert restored.dtype == np.float32 and restored.flags.c_contiguous
    assert restored.shape == original.shape
    assert np.array_equal(restored > 0, original > 0)
    assert np.max(np.abs(restored - original / scale)) <= 1e-5
    assert meta["border_mode_fraction"] == 1


@pytest.mark.parametrize("bad", ["constant", "nan", "below_background", "varied_faces"])
def test_rejects_unidentified_background(bad):
    original = np.zeros((12, 14, 16), dtype=np.float32)
    original[3:-3,3:-3,3:-3] = 1
    if bad == "constant": original[:] = 0
    elif bad == "nan": original[5,5,5] = np.nan
    elif bad == "below_background": original[5,5,5] = -1
    else: original[0,:,:] = np.arange(14*16).reshape(14,16) + 2
    with pytest.raises(ValueError): decode_background_zero(original)
