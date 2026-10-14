"""Train-only paired patch-coverage diagnostic; no model or test-pixel access.

The unchanged brain-sampling branch cancels in the paired difference. First
compute that exact difference from native training labels. Only if necessary
gates pass do we need N4 image support to establish absolute old coverage.
"""
import argparse
import datetime
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'archive/research_records/2026-09-24_segresnet_lesion_balanced_patch_proposal'
SCRATCH = Path('/tmp/aflow_lesion_balanced_patch_20260924')
CANON = ROOT / 'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train'
CANVAS = np.array([160, 208, 160])
SEED = 20260921
EPOCHS = 20
CFG = {}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def read(path):
    return json.loads(Path(path).read_text())


def initialize(config):
    CFG.update(config)
    CFG['fd'] = os.open(config['native_labels'], os.O_WRONLY)
    CFG['canonical'] = np.memmap(CANON / 'labels.u8', mode='r', dtype='uint8', shape=(972, 128, 128, 128))


def covered_counts(coords, component_ids, start, count):
    inside = ((coords >= start) & (coords < start + 128)).all(axis=1)
    return np.bincount(component_ids[inside], minlength=count + 1)[1:]


def foreground_diagnostic(y, slot, epochs=EPOCHS):
    cc, count = ndimage.label(y, structure=np.ones((3, 3, 3), dtype='uint8'))
    if count == 0:
        return {'component_voxels': [], 'old_known': [], 'new_known': [], 'common_epochs': epochs,
                'old_boundary_known': [], 'new_boundary_known': [], 'boundary_voxels': []}
    coords = np.argwhere(y)
    ids = cc[tuple(coords.T)]
    sizes = np.bincount(ids, minlength=count + 1)[1:]
    groups = [np.flatnonzero(ids == i) for i in range(1, count + 1)]
    boundary = y & ~ndimage.binary_erosion(y, structure=ndimage.generate_binary_structure(3, 1), border_value=0)
    bcoords = np.argwhere(boundary)
    bids = cc[tuple(bcoords.T)]
    bsizes = np.bincount(bids, minlength=count + 1)[1:]
    old = np.zeros(count, dtype=np.float64)
    new = old.copy()
    oldb = old.copy()
    newb = old.copy()
    common = 0
    for epoch in range(1, epochs + 1):
        ro = np.random.default_rng(SEED + epoch * 10000 + slot)
        rn = np.random.default_rng(SEED + epoch * 10000 + slot)
        foreground = ro.random() < .5
        assert foreground == (rn.random() < .5)
        if not foreground:
            common += 1
            continue
        co = coords[ro.integers(len(coords))]
        group = groups[rn.integers(count)]
        cn = coords[group[rn.integers(len(group))]]
        so = np.clip(co - 64, 0, CANVAS - 128)
        sn = np.clip(cn - 64, 0, CANVAS - 128)
        old += covered_counts(coords, ids, so, count) / sizes / epochs
        new += covered_counts(coords, ids, sn, count) / sizes / epochs
        oldb += covered_counts(bcoords, bids, so, count) / bsizes / epochs
        newb += covered_counts(bcoords, bids, sn, count) / bsizes / epochs
    return {'component_voxels': sizes.tolist(), 'old_known': old.tolist(), 'new_known': new.tolist(),
            'common_epochs': common, 'old_boundary_known': oldb.tolist(),
            'new_boundary_known': newb.tolist(), 'boundary_voxels': bsizes.tolist()}


def worker(task):
    slot, row, geom, canon_slot = task
    assert row['split'] == 'train'
    path = Path(row['label_path'])
    img = nib.as_closest_canonical(nib.load(str(path), mmap=True))
    assert list(img.shape) == geom['native_shape']
    assert np.allclose(nib.affines.voxel_sizes(img.affine), 1., atol=1e-5, rtol=0)
    raw = np.asarray(img.dataobj)
    assert np.isfinite(raw).all() and set(np.unique(raw).tolist()) <= {0, 1, 2, 3, 4}
    label = np.isin(raw, [1, 2, 3])
    slices = tuple(slice(o, o + n) for o, n in zip(geom['origin'], geom['extent']))
    crop = label[slices]
    resize_indices = [np.floor((np.arange(128) + .5) * n / 128).astype(int) for n in crop.shape]
    assert np.array_equal(crop[np.ix_(*resize_indices)], CFG['canonical'][canon_slot]), row['case_id']
    offset = (CANVAS - np.array(crop.shape)) // 2
    assert (offset >= 0).all()
    y = np.zeros(tuple(CANVAS), dtype=bool)
    y[tuple(slice(o, o + n) for o, n in zip(offset, crop.shape))] = crop
    payload = y.astype('uint8').tobytes()
    assert os.pwrite(CFG['fd'], payload, slot * len(payload)) == len(payload)
    result = foreground_diagnostic(y, slot)
    result.update(case_id=row['case_id'], subject_id=row['subject_id'], slot=slot,
                  original_label_sha256=sha(path), native_label_sha256=hashlib.sha256(payload).hexdigest(),
                  outside_crop=int(label.sum() - crop.sum()), canonical_label_exact=True)
    return result


def summarize(cases):
    sizes = np.array([s for c in cases for s in c['component_voxels']])
    if not len(sizes):
        raise RuntimeError('no training foreground components')
    old = np.array([s for c in cases for s in c['old_known']])
    new = np.array([s for c in cases for s in c['new_known']])
    common = np.array([c['common_epochs'] / EPOCHS for c in cases for _ in c['component_voxels']])
    threshold = float(np.quantile(sizes, .25, method='linear'))
    small = sizes <= threshold
    delta = new - old
    scan_deltas = [float(np.mean(np.array(c['new_known']) - np.array(c['old_known'])))
                   for c in cases if c['component_voxels']]
    bd = np.array([n - o for c in cases for o, n in zip(c['old_boundary_known'], c['new_boundary_known'])])
    result = {'components': len(sizes), 'small_component_max_voxels': threshold,
              'small_components_including_quartile_ties': int(small.sum()),
              'single_component_scans': sum(len(c['component_voxels']) == 1 for c in cases),
              'multi_component_scans': sum(len(c['component_voxels']) > 1 for c in cases),
              'empty_scans': sum(not c['component_voxels'] for c in cases),
              'small_coverage_paired_delta_exact': float(delta[small].mean()),
              'all_coverage_paired_delta_exact': float(delta.mean()),
              'scan_equal_coverage_paired_delta_exact': float(np.mean(scan_deltas)),
              'small_boundary_coverage_paired_delta_exact': float(bd[small].mean()),
              'all_boundary_coverage_paired_delta_exact': float(bd.mean()),
              'old_small_absolute_coverage_bounds': [float(old[small].mean()), float((old + common)[small].mean())],
              'new_small_absolute_coverage_bounds': [float(new[small].mean()), float((new + common)[small].mean())],
              'absolute_coverage_not_measured': True,
              'paired_delta_proof': 'Identical brain-branch RNG and inputs give identical crops; per-region contributions cancel. Foreground contributions divided by all20 epochs, not only foreground epochs.'}
    necessary = {'small_gain_at_least_0_05': result['small_coverage_paired_delta_exact'] >= .05,
                 'all_drop_no_more_than_0_01': result['all_coverage_paired_delta_exact'] >= -.01,
                 'old_lower_bound_below_0_95': result['old_small_absolute_coverage_bounds'][0] < .95,
                 'multi_component_exists': result['multi_component_scans'] > 0}
    result['necessary_gates'] = necessary
    result['decision'] = 'requires_absolute_coverage_stage' if all(necessary.values()) else 'do_not_train'
    result['training_admitted'] = False
    return result


def selftest():
    y = np.zeros(tuple(CANVAS), dtype=bool)
    y[4:7, 4:7, 4:7] = True
    y[145:150, 190:195, 145:150] = True
    d = foreground_diagnostic(y, 3)
    # Brute-force full20-epoch coverage, including the unchanged brain branch.
    cc, count = ndimage.label(y, structure=np.ones((3, 3, 3)))
    fg = np.argwhere(y)
    bg = np.argwhere(np.ones_like(y))
    full = [np.zeros(count), np.zeros(count)]
    for e in range(1, EPOCHS + 1):
        for arm in range(2):
            rng = np.random.default_rng(SEED + e * 10000 + 3)
            if rng.random() < .5:
                if arm == 0:
                    center = fg[rng.integers(len(fg))]
                else:
                    ident = int(rng.integers(count)) + 1
                    selected = np.argwhere(cc == ident)
                    center = selected[rng.integers(len(selected))]
            else:
                center = bg[rng.integers(len(bg))]
            start = np.clip(center - 64, 0, CANVAS - 128)
            sub = cc[tuple(slice(s, s + 128) for s in start)]
            full[arm] += np.bincount(sub.ravel(), minlength=count + 1)[1:] / np.array(d['component_voxels']) / EPOCHS
    assert np.allclose(full[1] - full[0], np.array(d['new_known']) - np.array(d['old_known']), atol=1e-14)
    import torch
    for shape in [(133, 162, 132), (156, 202, 159), (128, 128, 128)]:
        a = np.random.default_rng(1).integers(0, 2, size=shape, dtype='uint8')
        ix = [np.floor((np.arange(128) + .5) * n / 128).astype(int) for n in shape]
        actual = torch.nn.functional.interpolate(torch.from_numpy(a).float()[None, None], size=(128,)*3, mode='nearest-exact')[0, 0].numpy()
        assert np.array_equal(a[np.ix_(*ix)], actual)
    # One-component foreground selections and off-centre coverage bounds.
    single = y.copy(); single[145:150, 190:195, 145:150] = False
    one = foreground_diagnostic(single, 0)
    assert np.allclose(one['old_known'], one['new_known'])
    assert foreground_diagnostic(np.zeros_like(y), 0)['component_voxels'] == []
    result = {'utc': now(), 'status': 'passed', 'checks': ['paired_branch_cancellation_vs_bruteforce', 'nearest_exact_resize_three_shapes', 'single_component', 'empty_foreground'], 'patient_pixels_access': False}
    write(R / 'DIAGNOSTIC_SYNTHETIC_TEST.json', result)
    print(json.dumps(result))


def run(processes):
    freeze = read(R / 'DIAGNOSTIC_FREEZE.json')
    for relative, expected in freeze['files'].items():
        assert sha(ROOT / relative) == expected, relative
    assert read(R / 'DIAGNOSTIC_SYNTHETIC_TEST.json')['status'] == 'passed'
    access = R / 'DIAGNOSTIC_PREACCESS.json'
    if access.exists():
        raise RuntimeError('Existing access: inspect recovery explicitly; no automatic duplicate')
    index = read(CANON / 'index.json')
    rows = [json.loads(s) for s in (ROOT / freeze['train_manifest']).read_text().splitlines() if s]
    assert len(rows) == 972 and set(index['case_ids']) == {r['case_id'] for r in rows}
    assert shutil.disk_usage(ROOT).free >= 20 * 2**30
    assert shutil.disk_usage('/tmp').free >= 100 * 2**30
    SCRATCH.mkdir(parents=True, exist_ok=True)
    target = SCRATCH / 'train_labels.u8'
    with target.open('xb') as stream:
        stream.truncate(972 * int(CANVAS.prod()))
    write(access, {'utc': now(), 'worker': 1109780, 'roles': ['train_labels', 'existing_train_crop_geometry', 'canonical_train_labels'],
                   'new_test_pixels_access': False, 'validation_pixels_access': False, 'model_forward': False,
                   'freeze_sha256': sha(R / 'DIAGNOSTIC_FREEZE.json')})
    assert sha(CANON / 'labels.u8') == index['labels_sha256']
    slots = {c: i for i, c in enumerate(index['case_ids'])}
    tasks = [(i, row, index['crop_geometry'][row['case_id']], slots[row['case_id']]) for i, row in enumerate(rows)]
    started = time.monotonic()
    cases = []
    with mp.get_context('spawn').Pool(processes, initializer=initialize, initargs=({'native_labels': str(target)},)) as pool:
        for case in pool.imap_unordered(worker, tasks, chunksize=1):
            cases.append(case)
            write(R / 'PIPELINE_STATUS.json', {'utc': now(), 'state': 'diagnosing', 'running': True, 'pid': os.getpid(),
                  'worker': 1109780, 'completed_scans': len(cases), 'total_scans': 972, 'elapsed_seconds': time.monotonic() - started,
                  'training_started': False, 'new_test_pixels_access': False})
            if len(cases) % 20 == 0:
                print(json.dumps({'utc': now(), 'completed': len(cases), 'seconds': time.monotonic() - started}), flush=True)
    cases.sort(key=lambda c: c['slot'])
    assert len({c['case_id'] for c in cases}) == 972 and len({c['subject_id'] for c in cases}) == 444
    native_sha = sha(target)
    assert native_sha == freeze['historical_native_train_labels_sha256'], native_sha
    write(R / 'DIAGNOSTIC_PER_CASE.json', {'utc': now(), 'cases': cases})
    result = summarize(cases)
    result.update(utc=now(), scans=972, subjects=444, epochs_simulated=EPOCHS,
                  native_labels_sha256=native_sha, native_labels_historical_exact=True,
                  train_label_pixels_access=True, validation_pixels_access=False, new_test_pixels_access=False,
                  model_forward=False, training_updates=0, elapsed_seconds=time.monotonic() - started,
                  per_case_sha256=sha(R / 'DIAGNOSTIC_PER_CASE.json'))
    write(R / 'DIAGNOSTIC_RESULT.json', result)
    write(R / 'PIPELINE_STATUS.json', {'utc': now(), 'state': result['decision'], 'running': False,
          'worker': 1109780, 'training_started': False, 'new_test_pixels_access': False, 'result_sha256': sha(R / 'DIAGNOSTIC_RESULT.json')})
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['selftest', 'run'])
    parser.add_argument('--processes', type=int, default=12)
    args = parser.parse_args()
    try:
        selftest() if args.mode == 'selftest' else run(args.processes)
    except BaseException:
        import traceback
        write(R / f'FAILURE_{args.mode}_{int(time.time())}.json', {'utc': now(), 'traceback': traceback.format_exc()})
        raise
