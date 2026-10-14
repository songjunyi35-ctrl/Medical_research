#!/usr/bin/env python3
"""One-time frozen panel on the public UCSD BraTS-GLI 2024 test cohort (56 scans / 41 subjects)."""
from __future__ import annotations
import argparse
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import scripts.evaluate_ucsd_ptgbm_external_confirmation as model_io
from scripts import ucsd_brats2024test_protocol as p


class Data(Dataset):
    def __init__(self):
        receipt = json.loads(p.CACHE_RECEIPT.read_text())
        if receipt.get('all_gates_passed') is not True:
            raise ValueError('cache receipt failed')
        p.checked(p.CACHE / 'index.json', receipt['index_sha256'])
        self.index = json.loads((p.CACHE / 'index.json').read_text())
        if self.index.get('cases') != p.CASES or self.index.get('subjects') != p.SUBJECTS or self.index.get('split') != 'public_ucsd_brats2024test56':
            raise ValueError('cache identity')
        p.checked(p.CACHE / 'images.f32', self.index['images_sha256'])
        p.checked(p.CACHE / 'labels.u8', self.index['labels_sha256'])
        shape = (p.CASES, 128, 128, 128)
        self.images = np.memmap(p.CACHE / 'images.f32', dtype=np.float32, mode='r', shape=shape)
        self.labels = np.memmap(p.CACHE / 'labels.u8', dtype=np.uint8, mode='r', shape=shape)

    def __len__(self):
        return p.CASES

    def __getitem__(self, i):
        return {'image': torch.from_numpy(np.array(self.images[i]))[None],
                'label': torch.from_numpy(np.array(self.labels[i]))[None],
                'case_id': self.index['case_ids'][i], 'subject_id': self.index['subject_ids'][i]}


def row(logits, target, case_id, subject_id):
    pred = logits[0, 0] > 0
    truth = target[0, 0] > 0.5
    intersection = int((pred & truth).sum())
    denominator = int(pred.sum() + truth.sum())
    point = int(logits[0, 0].flatten().argmax())
    return {'case_id': str(case_id), 'subject_id': str(subject_id), 'locator_dsc': 2 * intersection / denominator,
            'automatic_point_hits_gt': bool(truth.flatten()[point]), 'automatic_point_flat_index': point,
            'predicted_wt_voxels': int(pred.sum()), 'target_wt_voxels': int(truth.sum()), 'intersection_voxels': intersection}


def subject_means(rows, key='locator_dsc'):
    grouped = defaultdict(list)
    for x in rows:
        grouped[x['subject_id']].append(float(x[key]))
    return {s: statistics.fmean(v) for s, v in sorted(grouped.items())}


def summarize(rows):
    if len(rows) != p.CASES or len({x['case_id'] for x in rows}) != p.CASES:
        raise ValueError('summary cohort')
    means = subject_means(rows)
    if len(means) != p.SUBJECTS:
        raise ValueError('subject count')
    hits = sum(x['automatic_point_hits_gt'] for x in rows)
    return {'cases': p.CASES, 'subjects': p.SUBJECTS,
            'locator_dsc_mean': statistics.fmean(x['locator_dsc'] for x in rows),
            'locator_subject_dsc_mean': statistics.fmean(means.values()),
            'automatic_point_hits': hits, 'automatic_point_hit_rate_positive_only': hits / p.CASES,
            'per_case': rows}


def interval(values, seed):
    a = np.asarray(values, dtype=np.float64)
    draw = np.random.default_rng(seed).integers(0, len(a), size=(p.BOOTSTRAP_SAMPLES, len(a)))
    return [float(x) for x in np.quantile(a[draw].mean(1), (.025, .975))]


def absolute(summary, seed):
    return {'scan_dsc_mean_95ci': interval([x['locator_dsc'] for x in summary['per_case']], seed),
            'subject_dsc_mean_95ci': interval(list(subject_means(summary['per_case']).values()), seed + 1)}


def compare(left, right, seed, name):
    if [x['case_id'] for x in left['per_case']] != [x['case_id'] for x in right['per_case']]:
        raise ValueError('paired order')
    delta_rows = [{'case_id': l['case_id'], 'subject_id': l['subject_id'],
                   'locator_dsc': r['locator_dsc'] - l['locator_dsc']}
                  for l, r in zip(left['per_case'], right['per_case'], strict=True)]
    scan_delta = [x['locator_dsc'] for x in delta_rows]
    subj_delta = list(subject_means(delta_rows).values())
    point = right['automatic_point_hits'] - left['automatic_point_hits']
    return {name: {'scan_dice': statistics.fmean(scan_delta), 'subject_dice': statistics.fmean(subj_delta), 'point_hits': point},
            'paired_scan_dice_95ci': interval(scan_delta, seed),
            'paired_subject_dice_95ci': interval(subj_delta, seed + 1),
            'cases_improved': sum(x > 0 for x in scan_delta), 'cases_worse': sum(x < 0 for x in scan_delta),
            'cases_tied': sum(x == 0 for x in scan_delta),
            'subjects_improved': sum(x > 0 for x in subj_delta), 'subjects_worse': sum(x < 0 for x in subj_delta),
            'subjects_tied': sum(x == 0 for x in subj_delta)}


@torch.inference_mode()
def evaluate_full(model, loader, device):
    identity_rows, flip_rows, forwards = [], [], 0
    for batch in loader:
        image = batch['image'].to(device, dtype=torch.float32)
        target = batch['label'].to(device, dtype=torch.float32)
        total = None
        identity = None
        for axes in p.AXES:
            view = torch.flip(image, axes) if axes else image
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(view).segmentation_logits.float()
            logits = torch.flip(logits, axes) if axes else logits
            if identity is None:
                identity = logits.clone()
                total = torch.zeros_like(logits)
            total.add_(logits)
            forwards += len(image)
        for i, case in enumerate(batch['case_id']):
            identity_rows.append(row(identity[i:i + 1], target[i:i + 1], case, batch['subject_id'][i]))
            flip_rows.append(row((total / 8)[i:i + 1], target[i:i + 1], case, batch['subject_id'][i]))
    return {'identity': summarize(identity_rows), 'flip8': summarize(flip_rows)}, forwards


@torch.inference_mode()
def evaluate_identity(model, loader, device):
    rows, forwards = [], 0
    for batch in loader:
        image = batch['image'].to(device, dtype=torch.float32)
        target = batch['label'].to(device, dtype=torch.float32)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(image).segmentation_logits.float()
        forwards += len(image)
        for i, case in enumerate(batch['case_id']):
            rows.append(row(logits[i:i + 1], target[i:i + 1], case, batch['subject_id'][i]))
    return summarize(rows), forwards


def interp(comparison, key):
    delta = comparison[key]['scan_dice']
    lo = comparison['paired_scan_dice_95ci'][0]
    if delta > 0 and lo > 0:
        return 'supported_on_ucsd_brats2024test56'
    return 'positive_but_uncertain' if delta > 0 else 'not_supported_or_reversed'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    p.docs()
    pre = json.loads(p.MODEL_PREACCESS.read_text())
    cache = json.loads(p.CACHE_RECEIPT.read_text())
    smoke = p.RECORD / 'SMOKE_RECEIPT.json'
    if (pre.get('status') != 'frozen_after_cache_before_model' or pre['access']['model_inference'] is not False
            or pre['code']['evaluator_sha256'] != p.sha(Path(__file__))
            or pre['cache_receipt_sha256'] != p.sha(p.CACHE_RECEIPT)
            or pre['smoke_receipt_sha256'] != p.sha(smoke)):
        raise ValueError('preaccess binding')
    out = p.RECORD / 'RAW_UCSD_BRATS2024TEST56.json'
    started = p.RECORD / 'RAW_UCSD_BRATS2024TEST56_STARTED.json'
    if out.exists() or started.exists():
        raise FileExistsError('refusing repeated evaluation')
    device = torch.device(a.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    loader = DataLoader(Data(), batch_size=8, shuffle=False, num_workers=0)
    p.write_new(started, {'schema': 'aflow.ucsd_brats2024test.started',
                          'created_at_utc': datetime.now(timezone.utc).isoformat(),
                          'preaccess_sha256': p.sha(p.MODEL_PREACCESS),
                          'checkpoint_loaded_at_write': False, 'model_forwards_at_write': 0})
    full, full_payload = model_io.load_model(p.FULL, p.FULL_SHA, 'primary', device)
    torch.cuda.reset_peak_memory_stats(device)
    begin = time.monotonic()
    full_summary, n_full = evaluate_full(full, loader, device)
    del full
    torch.cuda.empty_cache()
    gcsam, gcsam_payload = model_io.load_model(p.GCSAM, p.GCSAM_SHA, 'secondary', device)
    gcsam_summary, n_gc = evaluate_identity(gcsam, loader, device)
    forwards = n_full + n_gc
    if forwards != p.CASES * 9:
        raise RuntimeError('forward count mismatch')
    flip = compare(full_summary['identity'], full_summary['flip8'], p.SEED_PAIRED_FLIP8, 'flip8_minus_identity')
    gcomp = compare(full_summary['identity'], gcsam_summary, p.SEED_PAIRED_GCSAM, 'gcsam_minus_full')
    report = {'schema': 'aflow.ucsd_brats2024test.frozen_panel', 'schema_version': 1,
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'evidence_scope': 'one-time public post-challenge UCSD BraTS-GLI 2024 test cohort; 56 scans/41 subjects; patient-disjoint from 1350-case union and prior UCSD main release by official v3 mapping; UCSD is a BraTS2024 source site so NOT institution-disjoint; not hidden blind',
              'preregistered': True, 'not_test_tuned': True, 'not_pristine_blind': True,
              'protocol': {'target': 'post-treatment WT raw labels {1,2,3} exclude RC 4', 'threshold': 0.0,
                           'full972_views': ['identity', 'fixed_flip8_equal_raw_logit_mean'], 'gcsam_views': ['identity'],
                           'sam': False, 'postprocessing': False, 'model_ensemble': False,
                           'bootstrap_samples': p.BOOTSTRAP_SAMPLES},
              'data': {'cases': p.CASES, 'subjects': p.SUBJECTS, 'cohort_sha256': p.COHORT_SHA,
                       'cache_index_sha256': cache['index_sha256'], 'images_sha256': cache['images_sha256'],
                       'labels_sha256': cache['labels_sha256'], 'union_patient_overlap': 0,
                       'excluded_subject': 'UCSD-PTGBM-0038 (3 studies, union overlap via official mapping)'},
              'full972_kd168': {'checkpoint_sha256': p.FULL_SHA, 'epoch': full_payload['epoch'],
                                'identity': full_summary['identity'],
                                'identity_bootstrap': absolute(full_summary['identity'], p.SEED_ABS_FULL_IDENTITY),
                                'flip8': full_summary['flip8'],
                                'flip8_bootstrap': absolute(full_summary['flip8'], p.SEED_ABS_FULL_FLIP8)},
              'gcsam184': {'checkpoint_sha256': p.GCSAM_SHA, 'epoch': gcsam_payload['epoch'],
                           'identity': gcsam_summary,
                           'identity_bootstrap': absolute(gcsam_summary, p.SEED_ABS_GCSAM)},
              'comparisons': {'full_flip8_minus_identity': flip, 'gcsam_identity_minus_full_identity': gcomp},
              'interpretation_gates': {'full_flip8_minus_identity': interp(flip, 'flip8_minus_identity'),
                                       'gcsam_identity_minus_full_identity': interp(gcomp, 'gcsam_minus_full')},
              'performance_promotion_gate': None, 'result_must_not_trigger_tuning_or_repeat': True,
              'forwards': forwards, 'elapsed_seconds': time.monotonic() - begin,
              'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(device),
              'access': {'ucsd_brats2024test56': True, 'ucsd75': False, 'internal_test270': False,
                         'fixed_validation108': False, 'outer': False, 'confirmation': False,
                         'africa95': False, 'cptac33': False, 'waw_pre152': False, 'waw_post31': False},
              'training_source_caveat': 'local BraTS2024 Kaggle mirror; official Synapse parity unresolved',
              'evaluator_sha256': p.sha(Path(__file__))}
    p.write_new(out, report)
    print(json.dumps({'full_identity': {k: full_summary['identity'][k] for k in ('locator_dsc_mean', 'locator_subject_dsc_mean', 'automatic_point_hits')},
                      'full_flip8': {k: full_summary['flip8'][k] for k in ('locator_dsc_mean', 'locator_subject_dsc_mean', 'automatic_point_hits')},
                      'gcsam': {k: gcsam_summary[k] for k in ('locator_dsc_mean', 'locator_subject_dsc_mean', 'automatic_point_hits')},
                      'comparisons': report['comparisons'],
                      'interpretation': report['interpretation_gates'], 'forwards': forwards}, indent=2))


if __name__ == '__main__':
    main()
