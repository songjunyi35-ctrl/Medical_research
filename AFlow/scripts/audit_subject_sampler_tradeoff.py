"""Fit-metadata-only description of the two already registered sampling policies."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from scripts import structured_background_protocol as p
from scripts.build_t1ce_wt_native_volume_artifact import native_sampling_weights
from sammed3d.subject_balanced_sampling import normalize_subject_weights


def describe(case_ids, subject_map, native_volumes):
    weights = native_sampling_weights(case_ids, native_volumes)
    balanced = normalize_subject_weights(case_ids, weights, subject_map).numpy()
    original = weights.numpy()
    policies = {'native': original / original.sum(),
                'subject_equal': balanced / balanced.sum()}
    ordered = sorted(range(len(case_ids)), key=lambda i: (native_volumes[case_ids[i]], case_ids[i]))
    assert len(case_ids) % 4 == 0
    groups = np.array_split(ordered, 4)
    quartiles = []
    for i, slots in enumerate(groups):
        quartiles.append({'quartile': i + 1, 'scans': len(slots),
                         'native_voxels_min': int(native_volumes[case_ids[slots[0]]]),
                         'native_voxels_max': int(native_volumes[case_ids[slots[-1]]]),
                         'policies': {k: {'probability_mass': float(v[slots].sum()),
                                         'expected_draws_per968': float(v[slots].sum() * 968)}
                                      for k, v in policies.items()}})
    summary = {}
    for name, probabilities in policies.items():
        masses = defaultdict(float)
        for case, prob in zip(case_ids, probabilities):
            masses[subject_map[case]] += float(prob)
        summary[name] = {'total_probability': float(probabilities.sum()),
                         'expected_unique_scans_per968': float(np.sum(1 - (1 - probabilities) ** 968)),
                         'expected_unique_subjects_per968': float(sum(1 - (1 - mass) ** 968 for mass in masses.values())),
                         'max_to_min_scan_probability': float(probabilities.max() / probabilities.min()),
                         'subject_probability_min': min(masses.values()),
                         'subject_probability_max': max(masses.values())}
    relative = policies['subject_equal'] / policies['native']
    return {'quartiles': quartiles, 'policies': summary,
            'relative_scan_probability_min': float(relative.min()),
            'relative_scan_probability_max': float(relative.max())}


def main():
    record = p.ROOT / 'archive/reviews/2026-09-07_fit_subject_sampling'
    output = record / 'SAMPLER_TRADEOFF.json'
    if output.exists():
        raise FileExistsError(output)
    # Do not use check_roles here: this audit needs only the frozen fit manifest.
    p.checked(p.FIT, p.FIT_SHA)
    p.checked(p.NATIVE, p.NATIVE_SHA)
    fit = p.rows(p.FIT)
    assert len(fit) == 684 and len({r['subject_id'] for r in fit}) == 313
    volumes = json.loads(p.NATIVE.read_text())['native_wt_voxels']
    result = describe([r['case_id'] for r in fit],
                      {r['case_id']: r['subject_id'] for r in fit}, volumes)
    result.update({'created_utc': datetime.now(timezone.utc).isoformat(),
                   'status': 'descriptive_fit_metadata_only_no_method_change',
                   'fit_sha256': p.FIT_SHA, 'native_sha256': p.NATIVE_SHA,
                   'plan_sha256': p.sha(record / 'SAMPLER_TRADEOFF_PLAN.md'),
                   'script_sha256': p.sha(Path(__file__)),
                   'access': {'development_metrics': False, 'new_model_forwards': 0,
                              'image_or_label_arrays': False},
                   'source_dev_preaccess_exists_at_audit': (p.RECORD / 'DEV_PREACCESS.json').exists()})
    p.write_new(output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
