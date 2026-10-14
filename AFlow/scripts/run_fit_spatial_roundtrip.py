"""Frozen all-fit CPU audit of native label -> inherited128 -> native fidelity."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.metadata
import json
import multiprocessing as mp
from pathlib import Path
import resource
import time
import traceback

import nibabel as nib
import numpy as np
import torch
from surface_distance import metrics as surface_metrics
from scripts import structured_background_protocol as p
from sammed3d import spatial_roundtrip_audit as core

RECORD = p.ROOT / 'archive/research_records/2026-09-08_fit_spatial_roundtrip'
SOURCE_RECEIPT_SHA = 'cb7a89d933e7fd3f21250ba002b9c082c80018274b800c5f820ea7eed8be9554'
CONFIG = {}


def now():
    return datetime.now(timezone.utc).isoformat()


def init(config):
    torch.set_num_threads(1)
    CONFIG.update(config)
    CONFIG['labels'] = np.memmap(config['labels_path'], dtype=np.uint8, mode='r', shape=(972, 128, 128, 128))


def canonical_header_geometry(shape, affine):
    transform = nib.orientations.ornt_transform(nib.orientations.io_orientation(affine), nib.orientations.axcodes2ornt(('R', 'A', 'S')))
    new_shape = np.asarray(shape)[np.argsort(transform[:, 0])].tolist()
    return new_shape, affine @ nib.orientations.inv_ornt_aff(transform, shape)


def worker(task):
    started = time.monotonic()
    row, geometry, slot, expected_sha = task
    label_path = Path(row['label_path'])
    p.checked(label_path, expected_sha)
    image = nib.load(row['image_path'])  # Header/proxy only: do not materialize image dataobj.
    label_stored = nib.load(str(label_path))
    if image.header.get_xyzt_units()[0] != 'mm' or label_stored.header.get_xyzt_units()[0] != 'mm':
        raise ValueError('unverified physical distance units')
    image_shape, image_affine = canonical_header_geometry(image.shape, image.affine)
    label = nib.as_closest_canonical(label_stored)
    if image_shape != list(label.shape) or list(label.shape) != geometry['native_shape']:
        raise ValueError('canonical header/label/cache shape mismatch')
    if not np.allclose(image_affine, label.affine, atol=1e-5, rtol=0) or not np.isfinite(label.affine).all():
        raise ValueError('image/label affine mismatch')
    if nib.aff2axcodes(label.affine) != ('R', 'A', 'S'):
        raise ValueError('canonical orientation')
    spacing = nib.affines.voxel_sizes(label.affine)
    axes = label.affine[:3, :3] / spacing
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError('nonorthogonal grid')
    raw = np.asarray(label.dataobj)
    if not np.isfinite(raw).all() or not np.array_equal(raw, np.rint(raw)) or not set(np.unique(raw)) <= {0, 1, 2, 3, 4}:
        raise ValueError('invalid native label domain')
    truth = np.isin(raw, [1, 2, 3])
    result = core.analyze(truth, geometry['origin'], geometry['extent'], CONFIG['labels'][slot], spacing)
    return {'case_id': row['case_id'], 'subject_id': row['subject_id'],
            'source_label_sha256': expected_sha, 'canonical_affine': label.affine.tolist(),
            **result, 'compute_seconds': time.monotonic() - started,
            'worker_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


def stable(row):
    return {k: v for k, v in row.items() if k not in ('compute_seconds', 'worker_peak_rss_bytes')}


def summarize(rows):
    result = {}
    for mode in ('nearest-exact', 'trilinear'):
        summary = {}
        for key in ('dice', 'crop_internal_dice', 'absolute_relative_volume_error', 'relative_volume_error',
                    'hd95_mm', 'surface_dice_1mm', 'mean_surface_gt_to_pred_mm', 'mean_surface_pred_to_gt_mm'):
            values = [r['restorations'][mode][key] for r in rows]
            finite = [v for v in values if v is not None]
            grouped = defaultdict(list)
            for row, value in zip(rows, values):
                if value is not None:
                    grouped[row['subject_id']].append(value)
            summary[key] = {'scan_mean': float(np.mean(finite)) if finite else None,
                            'subject_mean': float(np.mean([np.mean(v) for v in grouped.values()])) if grouped else None,
                            'min_p05_median_p95_max': np.quantile(finite, [0, .05, .5, .95, 1]).tolist() if finite else None,
                            'undefined_scans': len(values) - len(finite)}
        summary['empty_reconstructions'] = sum(r['restorations'][mode]['empty_prediction'] for r in rows)
        result[mode] = summary
    bins = {name: {key: sum(r['components']['bins'][name][key] for r in rows)
                   for key in ('components', 'fully_crop_lost', 'resampling_lost', 'represented_at_128', 'native_voxels')}
            for name in core.BIN_NAMES}
    return {'restorations': result, 'component_bins': bins,
            'native_components_26': sum(r['components']['native_count_26'] for r in rows),
            'resampling_lost_components': sum(r['components']['resampling_lost_components'] for r in rows),
            'cases_with_resampling_lost_component': sum(r['components']['resampling_lost_components'] > 0 for r in rows),
            'fully_crop_lost_components': sum(r['components']['fully_crop_lost_components'] for r in rows),
            'crop_lost_WT_voxels': sum(r['crop_lost_WT_voxels'] for r in rows),
            'cases_with_crop_loss': sum(r['crop_lost_WT_voxels'] > 0 for r in rows),
            'coarse_components_merging_multiple_native': sum(r['components']['coarse_components_merging_multiple_native'] for r in rows),
            'native_components_split_in_coarse_grid': sum(r['components']['native_components_split_in_coarse_grid'] for r in rows),
            'spacing_mm_counts': {str(k): v for k, v in Counter(tuple(r['spacing_mm']) for r in rows).items()}}


def main():
    if (RECORD / 'PREACCESS.json').exists():
        raise FileExistsError('audit already started; do not restart or overwrite')
    fit, _ = p.check_roles()
    p.checked(p.CACHE / 'index.json', p.INDEX_SHA)
    index = json.loads((p.CACHE / 'index.json').read_text())
    p.checked(p.CACHE / 'labels.u8', index['labels_sha256'])
    source_receipt = p.RECORD / 'AUX_CACHE_RECEIPT.json'
    p.checked(source_receipt, SOURCE_RECEIPT_SHA)
    source_hashes = {r['case_id']: r['source_label_sha256'] for r in json.loads(source_receipt.read_text())['details']}
    assert set(source_hashes) == {r['case_id'] for r in fit}
    lookup = {case: i for i, case in enumerate(index['case_ids'])}
    tasks = [(r, index['crop_geometry'][r['case_id']], lookup[r['case_id']], source_hashes[r['case_id']]) for r in fit]
    files = [RECORD / 'IDEA.md', RECORD / 'PREREG.md', Path(__file__), Path(core.__file__),
             p.ROOT / 'tests/test_spatial_roundtrip_audit.py', Path(p.__file__),
             Path(surface_metrics.__file__), Path(surface_metrics.__file__).with_name('lookup_tables.py'),
             p.FIT, p.CACHE / 'index.json', source_receipt]
    p.write_new(RECORD / 'PREACCESS.json', {'created_utc': now(), 'status': 'frozen_before_fit_label_arrays',
                'fit_sha256': p.FIT_SHA, 'cache_index_sha256': p.INDEX_SHA, 'labels_sha256': index['labels_sha256'],
                'files': {str(f): p.sha(f) for f in files}, 'source_label_hashes': source_hashes,
                'libraries': {k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'torch', 'nibabel', 'surface-distance')},
                'fit_label_arrays_at_write': 0, 'first_case_header_probe_before_write': fit[0]['case_id'],
                'workers': 4, 'bootstrap': None, 'new_forwards': 0})
    config = {'labels_path': str(p.CACHE / 'labels.u8')}
    ctx = mp.get_context('spawn')
    started = time.monotonic()
    rows = []
    raw_path = RECORD / 'RAW_CASES.jsonl'
    def progress(stage):
        value = {'updated_utc': now(), 'stage': stage, 'completed_cases': len(rows), 'expected_cases': 684,
                 'elapsed_seconds': time.monotonic() - started}
        temp = RECORD / 'PROGRESS.tmp'
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(RECORD / 'PROGRESS.json')
    try:
        progress('fixed_first_fit_case')
        with ctx.Pool(1, initializer=init, initargs=(config,)) as pool:
            first = pool.apply_async(worker, (tasks[0],)).get(timeout=180)
        p.write_new(RECORD / 'FIRST_CASE.json', first)
        progress('all_fit_cases')
        with raw_path.open('x') as raw_file, ctx.Pool(4, initializer=init, initargs=(config,)) as pool:
            iterator = pool.imap(worker, tasks)
            while len(rows) < 684:
                if time.monotonic() - started > 1800:
                    raise TimeoutError('30 minute diagnostic budget')
                try:
                    row = iterator.next(timeout=20)
                except mp.TimeoutError:
                    progress('all_fit_cases')
                    continue
                if not rows and stable(row) != stable(first):
                    raise ValueError('first-case independent-process repeat not exact')
                body = json.dumps(row, sort_keys=True, allow_nan=False) + '\n'
                if raw_file.tell() + len(body.encode()) > 64 * 1024**2:
                    raise RuntimeError('raw report size budget exceeded')
                raw_file.write(body)
                raw_file.flush()
                rows.append(row)
                if len(rows) % 25 == 0:
                    progress('all_fit_cases')
                    print(f'fit spatial audit {len(rows)}/684', flush=True)
        assert [r['case_id'] for r in rows] == [r['case_id'] for r in fit]
        assert len({r['subject_id'] for r in rows}) == 313
        aggregate = summarize(rows)
        pre = json.loads((RECORD / 'PREACCESS.json').read_text())
        for path, digest in pre['files'].items():
            p.checked(Path(path), digest)
        result = {'created_utc': now(), 'status': 'completed_fit_only_diagnostic', 'cases': 684, 'subjects': 313,
                  'summary': aggregate, 'all_cache_byte_exact': True, 'first_case_cross_process_exact': True,
                  'preaccess_sha256': p.sha(RECORD / 'PREACCESS.json'), 'raw_sha256': p.sha(raw_path),
                  'elapsed_seconds': time.monotonic() - started,
                  'max_worker_peak_rss_bytes': max(r['worker_peak_rss_bytes'] for r in rows),
                  'raw_report_bytes': raw_path.stat().st_size,
                  'access': {'fit_labels_and_image_headers': True, 'fit_image_pixels': False,
                             'development_arrays_or_metrics': False, 'test_outer_confirmation_external': False,
                             'model_or_checkpoint_loaded': False, 'new_forwards': 0},
                  'interpretation': 'Fixed GT-label operator fidelity, not a learned-model score or attainable Dice upper bound.'}
        p.write_new(RECORD / 'RESULT.json', result)
        progress('completed_analysis_archival_pending')
        print(json.dumps({k: v for k, v in result.items() if k != 'summary'}, indent=2), flush=True)
    except BaseException as error:
        p.write_new(RECORD / 'FAILURE.json', {'created_utc': now(), 'error': repr(error),
                    'traceback': traceback.format_exc(), 'completed_cases': len(rows),
                    'first_case_report_exists': (RECORD / 'FIRST_CASE.json').exists(),
                    'fit_label_access_attempted': True, 'development_or_test_access': False, 'new_forwards': 0})
        progress('failed_do_not_restart_automatically')
        raise


if __name__ == '__main__':
    main()
