"""Deterministic label-only spatial fidelity metrics; never loads a model."""
import hashlib
import numpy as np
from scipy import ndimage
import torch
from torch.nn import functional as F
from surface_distance import metrics as surface_metrics

CONNECTIVITY = np.ones((3, 3, 3), dtype=bool)
BIN_EDGES = (1, 8, 27, 125, 1000)
BIN_NAMES = ('1', '2-8', '9-27', '28-125', '126-1000', '>1000')


def resize(array, shape, mode):
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32).copy())[None, None]
    kwargs = {'align_corners': False} if mode == 'trilinear' else {}
    return F.interpolate(tensor, size=tuple(int(x) for x in shape), mode=mode, **kwargs)[0, 0].numpy()


def dice(a, b):
    denominator = int(a.sum()) + int(b.sum())
    return 2 * int(np.count_nonzero(a & b)) / denominator if denominator else 1.


def measure(truth, prediction, spacing, with_surfaces=True):
    n_true, n_pred = int(truth.sum()), int(prediction.sum())
    if n_true == 0:
        raise ValueError('full native target must be nonempty')
    intersection = int(np.count_nonzero(truth & prediction))
    result = {'dice': dice(truth, prediction), 'intersection': intersection,
              'target_voxels': n_true, 'reconstructed_voxels': n_pred,
              'false_negative_voxels': n_true - intersection,
              'false_positive_voxels': n_pred - intersection,
              'relative_volume_error': (n_pred - n_true) / n_true,
              'absolute_relative_volume_error': abs(n_pred - n_true) / n_true,
              'reconstructed_native_sha256': hashlib.sha256(prediction.astype(np.uint8).tobytes()).hexdigest(),
              'empty_prediction': n_pred == 0}
    if with_surfaces:
        if n_pred == 0:
            result.update(hd95_mm=None, mean_surface_gt_to_pred_mm=None,
                          mean_surface_pred_to_gt_mm=None, surface_dice_1mm=0.)
        else:
            distances = surface_metrics.compute_surface_distances(truth, prediction, tuple(spacing))
            means = surface_metrics.compute_average_surface_distance(distances)
            result.update(hd95_mm=float(surface_metrics.compute_robust_hausdorff(distances, 95)),
                          mean_surface_gt_to_pred_mm=float(means[0]),
                          mean_surface_pred_to_gt_mm=float(means[1]),
                          surface_dice_1mm=float(surface_metrics.compute_surface_dice_at_tolerance(distances, 1.)))
    return result


def analyze(truth, origin, extent, cached, spacing=(1., 1., 1.), with_surfaces=True):
    truth = np.asarray(truth, dtype=bool)
    cached = np.asarray(cached)
    if truth.ndim != 3 or cached.ndim != 3 or not np.isin(cached, [0, 1]).all():
        raise ValueError('three-dimensional native and binary cache required')
    if len(origin) != 3 or len(extent) != 3 or any(o < 0 or e <= 0 or o + e > n for o, e, n in zip(origin, extent, truth.shape)):
        raise ValueError('crop outside native image')
    if len(spacing) != 3 or not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError('invalid spacing')
    if not truth.any():
        raise ValueError('empty native target')
    slices = tuple(slice(int(o), int(o + e)) for o, e in zip(origin, extent))
    cropped = truth[slices]
    rebuilt = resize(cropped, cached.shape, 'nearest-exact').astype(np.uint8)
    if not np.array_equal(rebuilt, cached):
        raise ValueError('inherited WT cache byte parity failed')

    ids, count = ndimage.label(truth, structure=CONNECTIVITY)
    if count >= 2**24:
        raise ValueError('component IDs exceed exact float32 integer range')
    sampled_ids = resize(ids[slices], cached.shape, 'nearest-exact').astype(np.int32)
    if not np.array_equal(sampled_ids > 0, cached > 0):
        raise ValueError('component ID and binary downsampling disagree')
    native_sizes = np.bincount(ids.ravel(), minlength=count + 1)
    crop_sizes = np.bincount(ids[slices].ravel(), minlength=count + 1)
    low_sizes = np.bincount(sampled_ids.ravel(), minlength=count + 1)
    low_components, low_count = ndimage.label(cached > 0, structure=CONNECTIVITY)
    source_links = np.zeros(count + 1, dtype=np.int64)
    merged_low_count = 0
    if low_count:
        pairs = np.unique(np.stack((low_components[cached > 0], sampled_ids[cached > 0]), axis=1), axis=0)
        low_links = np.bincount(pairs[:, 0], minlength=low_count + 1)
        source_links = np.bincount(pairs[:, 1], minlength=count + 1)
        merged_low_count = int(np.count_nonzero(low_links[1:] > 1))

    restorations = {}
    overlaps = {}
    for mode in ('nearest-exact', 'trilinear'):
        crop_prediction = resize(cached, extent, mode) > .5
        prediction = np.zeros(truth.shape, dtype=bool)
        prediction[slices] = crop_prediction
        metric = measure(truth, prediction, spacing, with_surfaces)
        metric['crop_internal_dice'] = dice(cropped, crop_prediction)
        restorations[mode] = metric
        overlaps[mode] = np.bincount(ids[prediction], minlength=count + 1)

    details = []
    bins = {name: {'components': 0, 'fully_crop_lost': 0, 'resampling_lost': 0,
                   'represented_at_128': 0, 'native_voxels': 0} for name in BIN_NAMES}
    for component in range(1, count + 1):
        size, inside, low = (int(x[component]) for x in (native_sizes, crop_sizes, low_sizes))
        group = BIN_NAMES[sum(size > edge for edge in BIN_EDGES)]
        row = {'id': component, 'native_voxels': size, 'crop_voxels': inside,
               'cached_voxels': low, 'size_bin': group, 'represented': low > 0,
               'fully_crop_lost': inside == 0, 'partially_cropped': 0 < inside < size,
               'resampling_lost': inside > 0 and low == 0,
               'coarse_component_links': int(source_links[component]),
               'restored_native_overlap_voxels': {m: int(x[component]) for m, x in overlaps.items()}}
        details.append(row)
        b = bins[group]
        b['components'] += 1
        b['native_voxels'] += size
        b['fully_crop_lost'] += int(row['fully_crop_lost'])
        b['resampling_lost'] += int(row['resampling_lost'])
        b['represented_at_128'] += int(row['represented'])
    full_size, crop_size = int(truth.sum()), int(cropped.sum())
    return {'native_shape': list(truth.shape), 'crop_origin': list(origin), 'crop_extent': list(extent),
            'spacing_mm': [float(x) for x in spacing], 'cache_byte_exact': True,
            'native_WT_voxels': full_size, 'crop_WT_voxels': crop_size,
            'crop_lost_WT_voxels': full_size - crop_size,
            'crop_only_dice': 2 * crop_size / (full_size + crop_size),
            'cached_WT_voxels': int(cached.sum()), 'restorations': restorations,
            'components': {'native_count_26': count, 'cached_count_26': low_count,
                           'represented_native_components': int(np.count_nonzero(low_sizes[1:])),
                           'resampling_lost_components': int(np.count_nonzero((crop_sizes[1:] > 0) & (low_sizes[1:] == 0))),
                           'fully_crop_lost_components': int(np.count_nonzero(crop_sizes[1:] == 0)),
                           'coarse_components_merging_multiple_native': merged_low_count,
                           'native_components_split_in_coarse_grid': int(np.count_nonzero(source_links[1:] > 1)),
                           'bins': bins, 'details': details}}
