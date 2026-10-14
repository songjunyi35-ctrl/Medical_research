#!/usr/bin/env python3
"""Raw-label scan, first-case probe, and canonical 128^3 cache for UCSD BraTS-2024-test 56."""
from __future__ import annotations
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from sammed3d import t1ce_binary
from scripts.build_no_n4_clean_cache import arrays
from scripts import ucsd_brats2024test_protocol as p

CFG = {}
ORIG = t1ce_binary._case_id
RAW_LABEL_SCAN = p.RECORD / 'RAW_LABEL_SCAN.json'


def init(cfg):
    CFG.update(cfg)
    CFG['allowed'] = set(cfg['allowed'])
    torch.set_num_threads(1)
    import SimpleITK as sitk
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    t1ce_binary._case_id = lambda v: v if v in CFG['allowed'] else ORIG(v)
    CFG['ifd'] = os.open(cfg['images'], os.O_WRONLY)
    CFG['lfd'] = os.open(cfg['labels'], os.O_WRONLY)


def arrsha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def smap():
    return {x['case_id']: x for x in json.load(open(p.SOURCE_RECEIPT))['selected']['files']}


def scan_labels():
    if RAW_LABEL_SCAN.exists():
        raise FileExistsError(RAW_LABEL_SCAN)
    rows = p.cases()
    sources = smap()
    out = {}
    for row in rows:
        s = sources[row['case_id']]
        mask_path = Path(s['mask_path'])
        p.checked(mask_path, s['mask_sha256'])
        raw = np.asanyarray(nib.load(str(mask_path)).dataobj)
        rounded = np.rint(raw)
        values = sorted(int(x) for x in np.unique(rounded))
        wt = int(np.isin(rounded, [1, 2, 3]).sum())
        out[row['case_id']] = {'values': values, 'raw_wt_voxels': wt,
                               'integer_exact': bool(np.allclose(raw, rounded, rtol=0, atol=1e-5)),
                               'domain_ok': set(values) <= {0, 1, 2, 3, 4}}
    gates = {'all_integer_exact': all(v['integer_exact'] for v in out.values()),
             'all_domain_ok': all(v['domain_ok'] for v in out.values()),
             'empty_wt_cases': sorted(c for c, v in out.items() if v['raw_wt_voxels'] == 0)}
    status = 'passed' if gates['all_integer_exact'] and gates['all_domain_ok'] and not gates['empty_wt_cases'] else 'failed'
    p.write_new(RAW_LABEL_SCAN, {'schema': 'aflow.ucsd_brats2024test.raw_label_scan', 'status': status,
                                 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                                 'cases': len(out), 'gates': gates, 'per_case': out,
                                 'image_pixels_decoded': False, 'model_inference': False,
                                 'runner_sha256': p.sha(Path(__file__))})
    print(json.dumps({'status': status, 'gates': gates}, indent=2))
    if status != 'passed':
        raise RuntimeError('raw label scan failed; stop before cache/model per PREREG')


def prepare(row, sources):
    s = sources[row['case_id']]
    ip, mpath = Path(s['image_path']), Path(s['mask_path'])
    p.checked(ip, s['image_sha256'])
    p.checked(mpath, s['mask_sha256'])
    ni = nib.as_closest_canonical(nib.load(str(ip), mmap=True))
    nm = nib.as_closest_canonical(nib.load(str(mpath), mmap=True))
    err = float(np.max(np.abs(ni.affine - nm.affine)))
    if ni.shape != nm.shape or err > 1e-5:
        raise ValueError('geometry ' + row['case_id'])
    image = np.ascontiguousarray(np.asanyarray(ni.dataobj), dtype=np.float32)
    raw = np.ascontiguousarray(np.asanyarray(nm.dataobj))
    rounded = np.rint(raw)
    values = sorted(int(x) for x in np.unique(rounded))
    if not np.isfinite(image).all() or not (image > 0).any() or not np.allclose(raw, rounded, rtol=0, atol=1e-5) or not set(values) <= {0, 1, 2, 3, 4}:
        raise ValueError('domain ' + row['case_id'])
    label = np.ascontiguousarray(np.isin(rounded, [1, 2, 3]), dtype=np.uint8)
    oi, ol, geo = arrays(row['case_id'], image, label, nib.affines.voxel_sizes(ni.affine), use_n4=True)
    crop = tuple(slice(o, o + e) for o, e in zip(geo['origin'], geo['extent']))
    cropped = int(label[crop].sum())
    if cropped != int(label.sum()) or not ol.any():
        raise ValueError('WT crop ' + row['case_id'])
    return oi, ol, {'native_shape': list(ni.shape), 'spacing_mm': [float(x) for x in nib.affines.voxel_sizes(ni.affine)],
                    'axis_codes': list(nib.aff2axcodes(ni.affine)), 'affine_error': err, 'raw_values': values,
                    'raw_wt': int(label.sum()), 'cropped_wt': cropped, 'resized_wt': int(ol.sum()), 'geometry': geo,
                    'image_sha256': arrsha(oi), 'label_sha256': arrsha(ol)}


def worker(task):
    slot, row, sources = task
    i, l, d = prepare(row, sources)
    vox = 128 ** 3
    if os.pwrite(CFG['ifd'], i.tobytes(), slot * vox * 4) != vox * 4 or os.pwrite(CFG['lfd'], l.tobytes(), slot * vox) != vox:
        raise IOError('short write')
    return slot, row['case_id'], row['subject_id'], d


def probe():
    if p.FIRST_PROBE.exists():
        raise FileExistsError(p.FIRST_PROBE)
    scan = json.load(open(RAW_LABEL_SCAN))
    if scan['status'] != 'passed':
        raise RuntimeError('label scan not passed')
    rows = p.cases()
    sources = smap()
    torch.set_num_threads(1)
    import SimpleITK as sitk
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    first = rows[0]['case_id']
    t1ce_binary._case_id = lambda v: v if v == first else ORIG(v)
    i, l, d = prepare(rows[0], sources)
    p.write_new(p.FIRST_PROBE, {'schema': 'aflow.ucsd_brats2024test.first_probe',
                                'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
                                'case_id': first, **d, 'output_image_shape': list(i.shape),
                                'output_label_values': sorted(int(x) for x in np.unique(l)),
                                'label_scan_sha256': p.sha(RAW_LABEL_SCAN),
                                'model_inference': False, 'other_cases_preprocessed': False,
                                'runner_sha256': p.sha(Path(__file__))})
    print(json.dumps(json.load(open(p.FIRST_PROBE)), indent=2))


def build(workers):
    rows = p.cases()
    sources = smap()
    probe_doc = json.load(open(p.FIRST_PROBE))
    needed = p.CASES * 128 ** 3 * 5
    if probe_doc['status'] != 'passed' or p.CACHE.exists() or p.CACHE_RECEIPT.exists():
        raise ValueError('state')
    if shutil.disk_usage(p.CACHE.parent if p.CACHE.parent.exists() else '/tmp').free < needed + (2 << 30):
        raise RuntimeError('disk')
    p.CACHE.mkdir(parents=True)
    images = p.CACHE / 'images.f32'
    labels = p.CACHE / 'labels.u8'
    for path, size in ((images, needed * 4 // 5), (labels, needed // 5)):
        with path.open('xb') as f:
            f.truncate(size)
    cfg = {'allowed': [r['case_id'] for r in rows], 'images': str(images), 'labels': str(labels)}
    start = time.monotonic()
    out = []
    with mp.get_context('spawn').Pool(workers, initializer=init, initargs=(cfg,)) as pool:
        for x in pool.imap_unordered(worker, [(i, r, sources) for i, r in enumerate(rows)], chunksize=1):
            out.append(x)
            if len(out) % 8 == 0 or len(out) == p.CASES:
                print(f'cached {len(out)}/{p.CASES}', flush=True)
    out.sort()
    details = {c: d for _, c, _, d in out}
    im = np.memmap(images, dtype=np.float32, mode='r', shape=(p.CASES, 128, 128, 128))
    la = np.memmap(labels, dtype=np.uint8, mode='r', shape=(p.CASES, 128, 128, 128))
    fd = details[rows[0]['case_id']]
    gates = {'all_finite_binary_nonempty': all(np.isfinite(im[i]).all() and np.all((la[i] == 0) | (la[i] == 1)) and la[i].any() for i in range(p.CASES)),
             'first_probe_image_exact': fd['image_sha256'] == probe_doc['image_sha256'],
             'first_probe_label_exact': fd['label_sha256'] == probe_doc['label_sha256'],
             'no_raw_wt_crop_loss': all(x['raw_wt'] == x['cropped_wt'] for x in details.values()),
             'all_resized_wt_nonempty': all(x['resized_wt'] > 0 for x in details.values())}
    if not all(gates.values()):
        raise RuntimeError(str(gates))
    index = {'schema': 'aflow.ucsd_brats2024test.cache', 'schema_version': 1,
             'created_at_utc': datetime.now(timezone.utc).isoformat(), 'split': 'public_ucsd_brats2024test56',
             'cases': p.CASES, 'subjects': p.SUBJECTS,
             'case_ids': [r['case_id'] for r in rows], 'subject_ids': [r['subject_id'] for r in rows],
             'images_sha256': p.sha(images), 'labels_sha256': p.sha(labels), 'model_shape': [128] * 3,
             'details': details, 'target': 'post-treatment WT raw labels1/2/3 exclude RC4',
             'preprocessing': {'canonical_ras': True, 'n4_shrink': 2, 'iterations': [50] * 4, 'threads': 1,
                               'image_only_crop': True, 'positive_zscore': True, 'resolution': 128},
             'gates': gates, 'cohort_sha256': p.COHORT_SHA, 'source_receipt_sha256': p.sha(p.SOURCE_RECEIPT),
             'first_probe_sha256': p.sha(p.FIRST_PROBE), 'builder_sha256': p.sha(Path(__file__)),
             'elapsed_seconds': time.monotonic() - start, 'model_inference': False}
    p.write_new(p.CACHE / 'index.json', index)
    receipt = {'schema': 'aflow.ucsd_brats2024test.cache_receipt', 'status': 'passed', 'all_gates_passed': True,
               'cases': p.CASES, 'subjects': p.SUBJECTS, 'index_sha256': p.sha(p.CACHE / 'index.json'),
               'images_sha256': index['images_sha256'], 'labels_sha256': index['labels_sha256'], 'gates': gates,
               'elapsed_seconds': index['elapsed_seconds'], 'model_inference': False}
    p.write_new(p.CACHE_RECEIPT, receipt)
    print(json.dumps(receipt, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=('labels', 'probe', 'build'))
    ap.add_argument('--workers', type=int, default=14)
    a = ap.parse_args()
    p.docs()
    if a.action == 'labels':
        scan_labels()
    elif a.action == 'probe':
        probe()
    else:
        build(a.workers)


if __name__ == '__main__':
    main()
