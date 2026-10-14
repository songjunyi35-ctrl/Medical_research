"""Preflight, reproduce the old native cache, then evaluate once; no retry loop."""
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.multiscale_e166_test270_20260929_common import *
from scripts import build_segresnet_native_res_test_cache as original


def preflight():
    verify()
    if shutil.disk_usage(ROOT).free < 20 * (1 << 30):
        raise RuntimeError('persistent free space below20GiB')
    needed = 270 * int(np.prod(TEST_CANVAS)) * 5
    if shutil.disk_usage('/tmp').free < needed + 4 * (1 << 30):
        raise RuntimeError('worker scratch space insufficient')
    split_rows = {k: read_rows(p) for k, p in MANIFEST.items()}
    expected = {'train': (972, 444), 'validation': (108, 50), 'test': (270, 119)}
    subjects = {}
    for key, rows in split_rows.items():
        subjects[key] = {r['subject_id'] for r in rows}
        if (len(rows), len(subjects[key])) != expected[key] or len({r['case_id'] for r in rows}) != len(rows):
            raise RuntimeError('split identity/count mismatch: ' + key)
    if any(subjects[a] & subjects[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
        raise RuntimeError('patient leakage')
    references()  # Archived per-case scores only; no reference model forward.
    shape = (2, 3, 4)
    label = np.zeros(shape, np.float32); label[0, 1, 2] = 1
    logits = np.full(shape, -4., np.float32); logits[0, 1, 2] = 4
    padded_label, offset = pad_to_canvas(label)
    padded_logits, _ = pad_to_canvas(logits, fill=99.)
    metric = case_metrics(torch.from_numpy(padded_logits)[None], torch.from_numpy(padded_label)[None],
                          {'canvas_offset': offset, 'extent': list(shape)})
    if metric['locator_dsc'] != 1. or metric['automatic_point'] != [0, 1, 2] or not metric['automatic_point_hits_gt']:
        raise RuntimeError('padding excluded synthetic metric failed')
    # Zero is background under the frozen strict raw-logit threshold.
    zero = torch.zeros_like(torch.from_numpy(padded_logits)[None])
    zero_metric = case_metrics(zero, torch.from_numpy(padded_label)[None], {'canvas_offset': offset, 'extent': list(shape)})
    if zero_metric['locator_dsc'] > 1e-5:
        raise RuntimeError('zero-logit threshold semantics changed')
    if torch.cuda.device_count() != 2:
        raise RuntimeError('expected the assigned two A100 GPUs')
    devices = []
    for i in range(2):
        with torch.cuda.device(i):
            x = torch.eye(32, device=f'cuda:{i}')
            if not torch.equal(x @ x, x):
                raise RuntimeError('CUDA arithmetic failed')
            devices.append(torch.cuda.get_device_name(i))
            del x
    numeric_mode(); seed_all(20260819); numeric_mode(); torch.set_num_threads(2)
    model = load_model(torch.device('cuda:0'))
    torch.cuda.reset_peak_memory_stats(0)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(torch.zeros((2, 1, *TEST_CANVAS), device='cuda:0')).segmentation_logits
    if tuple(output.shape) != (2, 1, *TEST_CANVAS) or not torch.isfinite(output).all():
        raise RuntimeError('native bf16 batch2 smoke failed')
    torch.cuda.synchronize(0)
    peak = torch.cuda.max_memory_allocated(0)
    del model, output
    torch.cuda.empty_cache()
    write(R / 'PREFLIGHT.json', {'utc': now(), 'status': 'passed', 'worker': 1114913,
          'gpu_names': devices, 'synthetic_only': True, 'patient_model_forward': False,
          'split_counts_and_isolation': True, 'crop_padding_and_threshold_checks': True,
          'checkpoint_strict_load_epoch': 166, 'bf16_batch2_peak_allocated_bytes': peak,
          'project_free_bytes': shutil.disk_usage(ROOT).free, 'scratch_free_bytes': shutil.disk_usage('/tmp').free})


def cache():
    verify()
    write_new(R / 'NATIVE_CACHE_PREACCESS.json', {'utc': now(), 'worker': 1114913,
              'test_image_label_pixel_access': True, 'candidate_model_forward': False,
              'purpose': 'fixed historical native payload reproduction and canonical SHA verification',
              'freeze_sha256': sha(R / 'FREEZE.json')})
    if CACHE.exists():
        raise RuntimeError('existing scratch cache: inspect instead of automatic overwrite')
    expected = {'images.f32': 'cf834d2b4e6cdaa414689c8723d532178980f6ad459324d13a903321f1064888',
                'labels.u8': 'ecb28a45607d54cf453ece3208ef8261ea30f496e747f919242ac1172445f222'}
    actual = {name: sha(CANONICAL_TEST / name) for name in expected}
    if actual != expected:
        raise RuntimeError('canonical test payload changed')
    write(R / 'CANONICAL_RECEIPT.json', {'utc': now(), 'status': 'passed', 'payload_sha256': actual,
          'index_sha256': sha(CANONICAL_TEST / 'index.json'), 'test_access': True})
    rows = read_rows(MANIFEST['test']); canonical = read(CANONICAL_TEST / 'index.json')
    if canonical['case_ids'] != [r['case_id'] for r in rows]:
        raise RuntimeError('canonical case order mismatch')
    CACHE.mkdir(parents=True)
    n = int(np.prod(TEST_CANVAS))
    for name, width in [('images.f32', 4), ('labels.u8', 1)]:
        with (CACHE / name).open('xb') as stream:
            stream.truncate(270 * n * width)
    config = {'images': str(CACHE / 'images.f32'), 'labels': str(CACHE / 'labels.u8'), 'canon_dir': str(CANONICAL_TEST)}
    tasks = [(i, row, canonical['crop_geometry'][row['case_id']]) for i, row in enumerate(rows)]
    geometry = {}; tick = time.monotonic()
    with mp.get_context('spawn').Pool(24, initializer=original.init, initargs=(config,)) as pool:
        for done, (slot, cid, geo, error) in enumerate(pool.imap_unordered(original.worker, tasks, chunksize=1), 1):
            if error:
                raise RuntimeError(cid + ': ' + error)
            if not (geo['canonical_image_within_tolerance'] and geo['canonical_label_exact'] and
                    geo['canonical_geometry_exact'] and geo['positive_voxels'] > 0):
                raise RuntimeError(cid + ': canonical reproduction failed')
            geometry[cid] = geo
            status('building_native_test_cache', done=done, total=270, seconds=time.monotonic() - tick)
    payload = {'images_sha256': sha(CACHE / 'images.f32'), 'labels_sha256': sha(CACHE / 'labels.u8')}
    historical = read(OLD_NATIVE / 'CACHE_RECEIPT.json')
    if any(payload[k] != historical[k] for k in payload):
        raise RuntimeError('native payload SHA differs from historical test grid')
    index = {'schema': SCHEMA + '.test', 'schema_version': 1, 'split': 'test', 'test_access': True,
             'canvas': list(TEST_CANVAS), 'count': 270, 'case_ids': [r['case_id'] for r in rows],
             'manifest_sha256': sha(MANIFEST['test']), 'geometry': geometry,
             'image_dtype': 'float32', 'label_dtype': 'uint8', **payload,
             'preprocessing': {'uses_gt_for_localization': False, 'spatial': 'braincrop_native_canvas',
                               'n4_shrink_factor': 2, 'normalization': 'canonical nonzero population zscore', 'padding': 'centered zero'},
             'historical_payload_exact': True, 'builder_sha256': sha(Path(__file__))}
    write(CACHE / 'index.json', index)
    write(R / 'CACHE_RECEIPT.json', {'utc': now(), 'status': 'passed', 'worker': 1114913, 'cache': str(CACHE),
          'index_sha256': sha(CACHE / 'index.json'), **payload, 'historical_payload_exact': True,
          'historical_receipt_sha256': sha(OLD_NATIVE / 'CACHE_RECEIPT.json'),
          'count': 270, 'subjects': 119, 'canvas': list(TEST_CANVAS), 'bytes': 270 * n * 5,
          'build_seconds': time.monotonic() - tick, 'all_canonical_labels_geometry_exact': True,
          'image_max_abs_diff': max(g['canonical_image_max_abs_diff'] for g in geometry.values()),
          'positives_outside_crop_total': sum(g['positives_outside_crop'] for g in geometry.values())})


def evaluate():
    verify()
    if (R / 'RAW_TEST270.json').exists():
        return read(R / 'RAW_TEST270.json')  # Only statistics may resume from committed raw.
    receipt = read(R / 'CACHE_RECEIPT.json')
    if not receipt['historical_payload_exact'] or sha(CACHE / 'index.json') != receipt['index_sha256']:
        raise RuntimeError('cache receipt verification failed')
    numeric_mode(); seed_all(20260819); numeric_mode(); torch.set_num_threads(2)
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    model = load_model(device)
    dataset = NativeTestDataset(CACHE, [r['case_id'] for r in read_rows(MANIFEST['test'])])
    write_new(R / 'MODEL_PREACCESS.json', {'utc': now(), 'worker': 1114913, 'epoch': 166,
              'checkpoint_sha256': CHECKPOINT_SHA, 'selection_sha256': sha(R / 'SELECTION_RECEIPT.json'),
              'freeze_sha256': sha(R / 'FREEZE.json'), 'cache_receipt_sha256': sha(R / 'CACHE_RECEIPT.json'),
              'precision': 'bfloat16', 'batch': 2, 'threshold_raw_logit': 0.,
              'one_main_output_full_brain_forward_per_scan': True, 'test_access': True,
              'test_tuned_development': True, 'current_test_retuning': False,
              'grid': 'native 1mm crop, 160x208x168 canvas; padding excluded',
              'no_automatic_patient_forward_repeat': True})
    torch.cuda.reset_peak_memory_stats(device); tick = time.monotonic(); rows = []
    with torch.inference_mode():
        for start in range(0, 270, 2):
            items = [dataset[i] for i in range(start, start + 2)]
            image = torch.stack([x['image'] for x in items]).to(device)
            label = torch.stack([x['label'] for x in items]).to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(image).segmentation_logits
            if not torch.isfinite(logits).all():
                raise RuntimeError('nonfinite test logits')
            for k, item in enumerate(items):
                metric = case_metrics(logits[k], label[k], dataset.geometry[item['case_id']])
                metric.update(case_id=item['case_id'], subject_id=item['subject_id'])
                rows.append(metric)
            # Preserve progress even if a later batch fails; never restart inference automatically.
            write(R / 'INFERENCE_JOURNAL.json', {'utc': now(), 'completed_scans': len(rows), 'per_case': rows})
            if len(rows) % 30 == 0:
                status('evaluating_test270', done=len(rows), total=270)
    torch.cuda.synchronize(device); validate_rows(rows)
    raw = {'utc': now(), 'status': 'complete_candidate_inference', 'method': ID, 'epoch': 166,
           'checkpoint_sha256': CHECKPOINT_SHA, 'summary': summarize_cases(rows), 'test_access': True,
           'test_tuned_development': True, 'current_test_used_for_selection_or_retuning': False,
           'not_pristine_blind': True, 'not_external': True, 'new_outer_confirmation_access': False,
           'grid': 'native 1mm crop; 160x208x168 canvas; padding excluded',
           'runtime': {'inference_seconds': time.monotonic() - tick,
                       'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                       'gpu': torch.cuda.get_device_name(device), 'evaluation_gpus': 1,
                       'batch': 2, 'torch_version': torch.__version__}}
    write(R / 'RAW_TEST270.json', raw)  # Raw is durable before any comparisons/selection.
    del model; torch.cuda.empty_cache()
    return raw


def statistics(raw):
    verify(); rows = raw['summary']['per_case']; validate_rows(rows)
    refs = references()
    result = summary_ci(rows)
    result.update(status='complete_test270', utc=now(), method=ID, epoch=166,
                  checkpoint_sha256=CHECKPOINT_SHA, point_hits=sum(r['automatic_point_hits_gt'] for r in rows),
                  point_denominator=270, grid=raw['grid'], runtime=raw['runtime'],
                  comparisons={k: paired(rows, v) for k, v in refs.items()},
                  reference_summaries={k: summary_ci(v) for k, v in refs.items()},
                  test_access=True, test_tuned_development=True, prior_test_informed_development=True,
                  current_test_used_for_selection_or_retuning=False, not_pristine_blind=True, not_external=True,
                  new_outer_confirmation_access=False, validation_gate_passed=False,
                  evaluation_status='user-authorized fixed exploratory test after failed validation gate',
                  numeric_comparability_note='Same native payload and scoring function; old combo166 used the same deterministic bf16 batch2 mode. Native173/pure193 are historical reports with earlier numeric mode; no reference re-forward.',
                  selection='108 validation highest Scan chose166; test did not select epoch',
                  forward_cost={'main_model_forwards_per_scan': 1, 'canvas_voxels': int(np.prod(TEST_CANVAS)),
                                'input_voxel_equivalents_128_cubed': float(np.prod(TEST_CANVAS) / 128**3),
                                'auxiliary_heads_in_deployment': False, 'not_a_measured_latency_ratio': True})
    files = [CHECKPOINT, R / 'SELECTION_RECEIPT.json', R / 'PREACCESS.json', R / 'MODEL_PREACCESS.json',
             R / 'PREFLIGHT.json', R / 'CANONICAL_RECEIPT.json', R / 'NATIVE_CACHE_PREACCESS.json',
             R / 'CACHE_RECEIPT.json', R / 'RAW_TEST270.json', R / 'FREEZE.json']
    result['evidence'] = {str(p.relative_to(ROOT)): sha(p) for p in files}
    write(R / 'RESULT.json', result)
    status('evaluation_complete_publication_pending', scan_dice=result['scan_dice'],
           subject_dice=result['subject_dice'], point_hits=result['point_hits'], new_test_model_access=True)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else 'run'
    if stage == 'statistics':
        statistics(read(R / 'RAW_TEST270.json')); return
    write_new(R / 'RUN_STARTED.json', {'utc': now(), 'pid': os.getpid(), 'worker': 1114913})
    status('preflight')
    preflight()
    status('cache_preparation')
    cache()
    status('candidate_evaluation')
    statistics(evaluate())


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        write(R / 'FAILURE.json', {'utc': now(), 'error': repr(exc), 'traceback': traceback.format_exc(),
              'candidate_model_access': (R / 'MODEL_PREACCESS.json').exists(),
              'candidate_raw_saved': (R / 'RAW_TEST270.json').exists(), 'no_automatic_retry': True})
        status('failed_no_automatic_retry', error=repr(exc)); raise
