"""Read only existing label-derived counts in frozen cache indexes; no pixel bytes."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'archive/reviews/2026-09-24_test270_image_profile'
INDEX = {
    'train': ROOT / 'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train/index.json',
    'validation': ROOT / 'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_ema_20260819/validation108/index.json',
    'test': ROOT / 'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/test270/index.json',
}
FEATURES = R / 'FEATURES.json'
REPORT = ROOT / 'archive/research_records/2026-09-21_t1ce_test270_10h/reports/ref_native173.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    protocol = json.loads((R / 'LABEL_METADATA_PROTOCOL.json').read_text())
    for rel, expected in protocol['files_sha256'].items():
        if sha(ROOT / rel) != expected:
            raise ValueError('frozen source changed: ' + rel)
    receipt = R / 'LABEL_METADATA_PREACCESS.json'
    if receipt.exists():
        raise RuntimeError('label-derived metadata already accessed in this stage')
    write(receipt, {'utc': now(), 'scope': 'existing per_case_positive_voxels from train/validation/test cache indexes and old fixed173 per-case score',
                    'new_label_pixel_access': False, 'test_label_derived_metadata_access': True,
                    'test_tuned': True, 'new_model_forward': False, 'new_training': False,
                    'protocol_sha256': sha(R / 'LABEL_METADATA_PROTOCOL.json')})
    data = json.loads(FEATURES.read_text())['splits']
    scores = {r['case_id']: r['locator_dsc'] for r in json.loads(REPORT.read_text())['per_case']}
    volumes = {}
    details = {}
    for split, path in INDEX.items():
        source = json.loads(path.read_text())
        counts = source['per_case_positive_voxels']
        cases = data[split]
        if set(counts) != {r['case_id'] for r in cases}:
            raise ValueError('count/report cohort mismatch')
        v = np.array([counts[r['case_id']] for r in cases], dtype=float)
        if not np.isfinite(v).all() or not (v > 0).all():
            raise ValueError('invalid positive count')
        volumes[split] = v
        details[split] = {k: float(n) for k, n in zip(['p05', 'p10', 'p25', 'median', 'p75', 'p90', 'p95'],
            np.quantile(v, [.05, .10, .25, .50, .75, .90, .95]))}
        details[split]['min'] = float(v.min())
        details[split]['max'] = float(v.max())
    threshold = float(np.quantile(volumes['train'], .10))
    small = {}
    for split, v in volumes.items():
        small[split] = {'scans_at_or_below_train_p10': int((v <= threshold).sum()),
                        'fraction': float(np.mean(v <= threshold))}
    test_dice = np.array([scores[r['case_id']] for r in data['test']], dtype=float)
    assoc = spearmanr(volumes['test'], test_dice)
    result = {'utc': now(), 'status': 'complete', 'raster': 'existing preprocessed 128^3 label cache; not native 1mm tumor volume',
              'volume_quantiles_voxels': details, 'training_p10_voxels': threshold,
              'share_at_or_below_train_p10': small,
              'test_volume_vs_historical_fixed173_scan_dice_spearman': float(assoc.statistic),
              'test_label_derived_metadata_access': True, 'new_label_pixel_access': False,
              'new_model_forward': False, 'new_training': False,
              'interpretation_limit': 'Association between volume and Dice does not show that resampling cases or changing loss would improve performance.'}
    write(R / 'LABEL_METADATA_RESULT.json', result)
    print(json.dumps({'train_median': details['train']['median'], 'test_median': details['test']['median'],
                      'train_small_fraction': small['train']['fraction'], 'test_small_fraction': small['test']['fraction'],
                      'rho': float(assoc.statistic)}))


if __name__ == '__main__':
    main()
