"""Strict constant-background decoding for linearly normalized, skull-stripped MRI."""
import numpy as np


def decode_background_zero(volume):
    array = np.asarray(volume)
    if array.ndim != 3 or min(array.shape) < 2 or not np.isfinite(array).all():
        raise ValueError("expected finite nondegenerate 3D image")
    border = np.concatenate((array[0].ravel(), array[-1].ravel(), array[:, 0].ravel(),
                             array[:, -1].ravel(), array[:, :, 0].ravel(), array[:, :, -1].ravel()))
    values, counts = np.unique(border, return_counts=True)
    winner = int(np.argmax(counts))
    background = float(values[winner])
    fraction = float(counts[winner] / counts.sum())
    if fraction < 0.95 or background != float(array.min()):
        raise ValueError("background must dominate >=95% of six faces and equal global minimum")
    decoded = np.ascontiguousarray(array.astype(np.float64) - background, dtype=np.float32)
    if not np.isfinite(decoded).all() or not (decoded > 0).any():
        raise ValueError("no finite positive image support")
    if not np.array_equal(decoded > 0, array > background):
        raise ValueError("float32 decoding changed support")
    return decoded, {"background_value": background, "border_mode_fraction": fraction,
                     "background_voxels": int((array == background).sum()),
                     "positive_voxels": int((decoded > 0).sum())}
