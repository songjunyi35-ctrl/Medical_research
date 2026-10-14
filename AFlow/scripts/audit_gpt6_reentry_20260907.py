"""Read-only input audit; write a new receipt, never load models or image arrays."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'archive/reviews/2026-09-07_gpt6_reentry/AUDIT.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ci(values, seed):
    values = np.asarray(values, dtype=np.float64)
    draw = np.random.default_rng(seed).integers(0, len(values), (10000, len(values)))
    return np.quantile(values[draw].mean(axis=1), [0.025, 0.975]).tolist()


def subject_values(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['subject_id']].append(row['locator_dsc'])
    return [statistics.fmean(grouped[key]) for key in sorted(grouped)]


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    source = ROOT / 'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl'
    split_path = ROOT / 'archive/research_records/2026-09-05_no_n4_clean_teacher_replication/SPLIT.json'
    assert sha(source) == '6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65'
    assert sha(split_path) == '68975292cc10ab581657a97789050a28cc03d4a3bc3b0991682b7284bfa9c3f2'
    rows = read_rows(source)
    counts = Counter(row['subject_id'] for row in rows)
    split = json.loads(split_path.read_text())
    excluded = set()
    for item in split['exclusions']:
        path = ROOT / item['path']
        assert sha(path) == item['sha256']
        excluded.update(row.get('subject_id') or '-'.join(row['case_id'].split('-')[:3]) for row in read_rows(path))
    for fold in ('e', 'f'):
        item = split['folds'][fold]['inner']
        path = ROOT / item['path']
        assert sha(path) == item['sha256']
        excluded.update(row['subject_id'] for row in read_rows(path))
    remaining = set(counts) - excluded
    record = ROOT / 'archive/research_records/2026-09-06_ucsd_brats2024test_frozen_panel'
    raw_path = record / 'RAW_UCSD_BRATS2024TEST56.json'
    raw = json.loads(raw_path.read_text())
    verification = json.loads((record / 'VERIFICATION_RECEIPT.json').read_text())
    hashes = {'raw': sha(raw_path), 'train_manifest': sha(source), 'historical_split': sha(split_path)}
    assert hashes['raw'] == verification['raw_report_sha256']
    for name, key in [('SOURCE_RECEIPT.json', 'source_receipt_sha256'), ('CACHE_RECEIPT.json', 'cache_receipt_sha256'), ('MODEL_PREACCESS.json', 'model_preaccess_sha256')]:
        hashes[name] = sha(record / name)
        assert hashes[name] == verification[key]
    summaries = [raw['full972_kd168']['identity'], raw['full972_kd168']['flip8'], raw['gcsam184']['identity']]
    absolutes = [raw['full972_kd168']['identity_bootstrap'], raw['full972_kd168']['flip8_bootstrap'], raw['gcsam184']['identity_bootstrap']]
    checks = {}
    for name, summary, absolute, seed in zip(['full_identity', 'full_flip8', 'gcsam_identity'], summaries, absolutes, [20261601, 20261611, 20261621]):
        per_case = summary['per_case']
        assert len(per_case) == 56 and len({row['subject_id'] for row in per_case}) == 41
        assert all(abs(row['locator_dsc'] - 2 * row['intersection_voxels'] / (row['predicted_wt_voxels'] + row['target_wt_voxels'])) < 1e-12 for row in per_case)
        checks[name] = {
            'scan_mean_exact': statistics.fmean(row['locator_dsc'] for row in per_case) == summary['locator_dsc_mean'],
            'subject_mean_exact': statistics.fmean(subject_values(per_case)) == summary['locator_subject_dsc_mean'],
            'point_exact': sum(row['automatic_point_hits_gt'] for row in per_case) == summary['automatic_point_hits'],
            'scan_ci_exact': ci([row['locator_dsc'] for row in per_case], seed) == absolute['scan_dsc_mean_95ci'],
            'subject_ci_exact': ci(subject_values(per_case), seed + 1) == absolute['subject_dsc_mean_95ci'],
        }
    for name, right, seed in [('full_flip8_minus_identity', summaries[1], 20261701), ('gcsam_identity_minus_full_identity', summaries[2], 20261801)]:
        delta = []
        for left_row, right_row in zip(summaries[0]['per_case'], right['per_case'], strict=True):
            assert (left_row['case_id'], left_row['subject_id']) == (right_row['case_id'], right_row['subject_id'])
            delta.append({**left_row, 'locator_dsc': right_row['locator_dsc'] - left_row['locator_dsc']})
        stored = raw['comparisons'][name]
        checks[name] = {
            'scan_ci_exact': ci([row['locator_dsc'] for row in delta], seed) == stored['paired_scan_dice_95ci'],
            'subject_ci_exact': ci(subject_values(delta), seed + 1) == stored['paired_subject_dice_95ci'],
        }
    assert all(all(group.values()) for group in checks.values())
    result = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
        'immutable_role_capacity': {'train_scans': len(rows), 'train_subjects': len(counts), 'remaining_subjects': sorted(remaining), 'remaining_scans': sum(counts[s] for s in remaining)},
        'fable_saved_report_recomputation': checks, 'hashes': hashes,
        'access': {'image_arrays': False, 'checkpoint_loaded': False, 'model_forward': 0, 'new_evaluation': False, 'role_reassignment_applied': False},
        'limitation': 'Saved counts/receipt hashes rechecked; no independent rerun of predictions, publisher audit, or image geometry. Historical scan bootstrap treats scans independently; subject CI aggregates repeated scans.'
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
