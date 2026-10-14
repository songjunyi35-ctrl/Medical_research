"""Compare frozen train/validation/test image-only cache features; no model calls."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'archive/reviews/2026-09-24_test270_image_profile'
CACHES = {
    'train': ROOT / 'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train',
    'validation': ROOT / 'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_ema_20260819/validation108',
    'test': ROOT / 'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/test270',
}
MANIFEST = {
    'train': ROOT / 'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl',
    'validation': ROOT / 'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/validation_t1ce_wt.jsonl',
    'test': ROOT / 'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/sealed_test/test_cases_t1ce.jsonl',
}
REPORT = ROOT / 'archive/research_records/2026-09-21_t1ce_test270_10h/reports/ref_native173.json'
NAMES = ['extent_x', 'extent_y', 'extent_z', 'nonzero_fraction',
         'q01', 'q05', 'q25', 'q50', 'q75', 'q95', 'q99',
         'contrast_q95_q05', 'high_signal_fraction', 'mean_abs_neighbor_change']


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def feature(image, geometry):
    # The same fixed spatial lattice is used for every split. Image cache is
    # 128^3 in every cohort; no label array or source test volume is opened.
    sampled = image[::4, ::4, ::4].astype(np.float32, copy=False)
    foreground = sampled[sampled != 0]
    if foreground.size < 1000 or not np.isfinite(foreground).all():
        raise ValueError('invalid nonzero image support')
    q = np.quantile(foreground, [.01, .05, .25, .50, .75, .95, .99])
    neighbor = float(np.mean([np.abs(np.diff(sampled, axis=a)).mean() for a in range(3)]))
    return [*map(float, geometry['extent']), float(foreground.size / sampled.size),
            *map(float, q), float(q[5] - q[1]),
            float(np.mean(foreground > 1.5)), neighbor]


def extract(split, cache):
    index = json.loads((cache / 'index.json').read_text())
    rows = [json.loads(s) for s in MANIFEST[split].read_text().splitlines() if s]
    mapping = {r['case_id']: r for r in rows}
    expected = {'train': (972, 444), 'validation': (108, 50), 'test': (270, 119)}[split]
    if len(mapping) != len(rows) or len(rows) != expected[0] or len({r['subject_id'] for r in rows}) != expected[1]:
        raise ValueError('manifest cohort mismatch')
    if index['count'] != expected[0] or set(index['case_ids']) != set(mapping):
        raise ValueError('cache cohort mismatch')
    if index['preprocessing']['uses_gt_for_localization'] or index['test_access'] != (split == 'test'):
        raise ValueError('cache role mismatch')
    if sha(cache / 'images.f32') != index['images_sha256']:
        raise ValueError('image cache SHA mismatch')
    images = np.memmap(cache / 'images.f32', mode='r', dtype='float32', shape=(expected[0], 128, 128, 128))
    output = []
    for i, case_id in enumerate(index['case_ids']):
        vector = feature(images[i], index['crop_geometry'][case_id])
        if len(vector) != len(NAMES):
            raise ValueError('feature count mismatch')
        row = mapping[case_id]
        output.append({'case_id': case_id, 'subject_id': row['subject_id'],
                       'source_part': row['source_part'], 'timepoint': row['timepoint'],
                       'features': dict(zip(NAMES, vector))})
    return output


def analyze(groups):
    train, val, test = (groups[s] for s in ['train', 'validation', 'test'])
    arrays = {s: np.array([[r['features'][k] for k in NAMES] for r in groups[s]], dtype=float)
              for s in groups}
    tr, te = arrays['train'], arrays['test']
    x = np.concatenate([tr, te]); y = np.array([0] * len(tr) + [1] * len(te))
    subjects = np.array([r['subject_id'] for r in train + test])
    split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260924)
    prediction = np.full(len(y), np.nan)
    for fit, hold in split.split(x, y, groups=subjects):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1, class_weight='balanced', random_state=20260924))
        model.fit(x[fit], y[fit]); prediction[hold] = model.predict_proba(x[hold])[:, 1]
    if not np.isfinite(prediction).all():
        raise ValueError('incomplete out-of-fold prediction')
    auc = float(roc_auc_score(y, prediction))
    historical = json.loads(REPORT.read_text())
    scores = {r['case_id']: r['locator_dsc'] for r in historical['per_case']}
    if set(scores) != {r['case_id'] for r in test}:
        raise ValueError('historical reference cohort mismatch')
    target_dice = np.array([scores[r['case_id']] for r in test])
    per_feature = {}
    for j, name in enumerate(NAMES):
        median_train, median_val, median_test = map(float, [np.median(arrays[s][:, j]) for s in ['train', 'validation', 'test']])
        q25, q75 = np.quantile(tr[:, j], [.25, .75]); iqr = float(q75 - q25)
        if iqr == 0:
            standardized_gap = None
        else:
            standardized_gap = float((median_test - median_train) / iqr)
        correlation = spearmanr(te[:, j], target_dice)
        per_feature[name] = {'train_median': median_train, 'validation_median': median_val,
                             'test_median': median_test, 'train_iqr': iqr,
                             'test_minus_train_over_train_iqr': standardized_gap,
                             'spearman_vs_historical_fixed173_test_dice': float(correlation.statistic)}
    # Descriptive screen only; no feature or augmentation is auto-selected.
    strong = [name for name, stats in per_feature.items()
              if stats['test_minus_train_over_train_iqr'] is not None
              and abs(stats['test_minus_train_over_train_iqr']) >= .5
              and abs(stats['spearman_vs_historical_fixed173_test_dice']) >= .3]
    source_counts = {s: {p: sum(r['source_part'] == p for r in groups[s]) for p in ['Part1', 'Part2', 'Part3']}
                     for s in groups}
    return {'utc': now(), 'status': 'complete', 'auc_train_vs_test_5fold_subject_grouped': auc,
            'feature_names': NAMES, 'per_feature': per_feature, 'source_part_counts': source_counts,
            'descriptive_screen': {'criteria': 'domain AUC >=0.75 and one feature median shift >=0.5 training IQR and |Spearman vs historical fixed173 test Dice| >=0.3',
                                   'features_meeting_both_univariate_checks': strong,
                                   'screen_passed': auc >= .75 and bool(strong)},
            'historical_fixed173_test_report_sha256': sha(REPORT),
            'test_labels_read_this_run': False, 'historical_test_score_used_descriptively': True,
            'new_model_forward': False, 'new_training': False,
            'interpretation_limit': 'Domain discrimination and test-error association are descriptive, not a causal treatment effect.'}


def main():
    protocol = json.loads((R / 'PROTOCOL.json').read_text())
    for relative, expected in protocol['frozen_files_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError('frozen file changed: ' + relative)
    if (R / 'PREACCESS.json').exists():
        raise RuntimeError('existing test-image access; inspect before repeat')
    write(R / 'PREACCESS.json', {'utc': now(), 'scope': 'cached preprocessed image voxels from train972, validation108, test270; historical fixed173 per-case scores',
                                'test_image_pixels_access': True, 'test_labels_read': False,
                                'new_model_forward': False, 'training_started': False,
                                'protocol_sha256': sha(R / 'PROTOCOL.json')})
    groups = {}
    for split, cache in CACHES.items():
        groups[split] = extract(split, cache)
        write(R / 'STATUS.json', {'utc': now(), 'state': 'image_features_read', 'completed_split': split,
                                  'scans_completed': sum(len(x) for x in groups.values()),
                                  'new_model_forward': False, 'training_started': False})
    all_subjects = [set(r['subject_id'] for r in groups[s]) for s in ['train', 'validation', 'test']]
    if any(all_subjects[i] & all_subjects[j] for i, j in [(0, 1), (0, 2), (1, 2)]):
        raise ValueError('patient isolation violated')
    write(R / 'FEATURES.json', {'utc': now(), 'subsample': 'fixed [::4,::4,::4] spatial lattice', 'splits': groups})
    result = analyze(groups)
    result['features_sha256'] = sha(R / 'FEATURES.json')
    write(R / 'RESULT.json', result)
    write(R / 'STATUS.json', {'utc': now(), 'state': 'analysis_complete', 'screen_passed': result['descriptive_screen']['screen_passed'],
                              'test_image_pixels_access': True, 'test_labels_read': False,
                              'training_started': False, 'new_model_forward': False})
    print(json.dumps({'auc': result['auc_train_vs_test_5fold_subject_grouped'],
                      'screen': result['descriptive_screen'], 'feature_count': len(NAMES)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
