"""Read-only, test-tuned label geometry audit on the canonical 128^3 cache."""
import datetime
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'archive/reviews/2026-09-24_test270_image_profile'
CACHE = {
    'train': ROOT / 'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train',
    'validation': ROOT / 'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_ema_20260819/validation108',
    'test': ROOT / 'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/test270',
}
COUNTS = {'train': 972, 'validation': 108, 'test': 270}
LOCAL = {}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def initialize(split):
    LOCAL['split'] = split
    LOCAL['index'] = json.loads((CACHE[split] / 'index.json').read_text())
    LOCAL['labels'] = np.memmap(CACHE[split] / 'labels.u8', mode='r', dtype='uint8',
                                shape=(COUNTS[split], 128, 128, 128))


def case(slot):
    labels = LOCAL['labels'][slot]
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('nonbinary WT label')
    mask = labels != 0
    positive = int(mask.sum())
    identifier = LOCAL['index']['case_ids'][slot]
    if positive != int(LOCAL['index']['per_case_positive_voxels'][identifier]) or positive <= 0:
        raise ValueError('cached label count mismatch')
    connected, n = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype='uint8'))
    counts = np.bincount(connected.ravel(), minlength=n + 1)[1:]
    coords = np.argwhere(mask)
    near = np.any((coords < 5) | (coords >= 123), axis=1)
    edge_voxels = int(near.sum())
    return {'case_id': identifier, 'subject_id': identifier.rsplit('-', 1)[0],
            'positive_voxels_128': positive, 'components26': int(n),
            'components_le5vox': int(np.sum(counts <= 5)),
            'component_le5vox_total_positive_voxels': int(counts[counts <= 5].sum()),
            'largest_component_fraction': float(counts.max() / positive),
            'positive_voxels_in_outer5_cube_slices': edge_voxels,
            'fraction_positive_in_outer5_cube_slices': float(edge_voxels / positive),
            'min_distance_positive_to_cube_edge': int(np.minimum(coords, 127 - coords).min())}


def summarize(cases):
    return {'scans': len(cases), 'subjects': len({c['subject_id'] for c in cases}),
            'median_components26': float(np.median([c['components26'] for c in cases])),
            'fraction_multicomponent': float(np.mean([c['components26'] > 1 for c in cases])),
            'total_components_le5vox': sum(c['components_le5vox'] for c in cases),
            'total_microfragment_voxels': sum(c['component_le5vox_total_positive_voxels'] for c in cases),
            'total_positive_voxels': sum(c['positive_voxels_128'] for c in cases),
            'fraction_scans_with_microfragments': float(np.mean([c['components_le5vox'] > 0 for c in cases])),
            'fraction_scans_with_positive_near_cube_edge': float(np.mean([c['positive_voxels_in_outer5_cube_slices'] > 0 for c in cases])),
            'median_fraction_positive_near_cube_edge': float(np.median([c['fraction_positive_in_outer5_cube_slices'] for c in cases]))}


def main():
    protocol = json.loads((R / 'LABEL_GEOMETRY_PROTOCOL.json').read_text())
    for relative, expected in protocol['files_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError('frozen input changed: ' + relative)
    receipt = R / 'LABEL_GEOMETRY_PREACCESS.json'
    if receipt.exists():
        raise RuntimeError('existing label pixel access; inspect before repeated run')
    write(receipt, {'utc': now(), 'test_tuned': True, 'scope': 'canonical 128^3 train/validation/test label cache geometry only',
                    'test_label_pixels_access': True, 'new_image_pixels_access': False,
                    'new_model_forward': False, 'training_started': False,
                    'protocol_sha256': sha(R / 'LABEL_GEOMETRY_PROTOCOL.json')})
    output = {}
    for split in ['train', 'validation', 'test']:
        index = json.loads((CACHE[split] / 'index.json').read_text())
        if sha(CACHE[split] / 'labels.u8') != index['labels_sha256']:
            raise RuntimeError(split + ' label cache SHA differs')
        tic = time.monotonic()
        records = []
        with mp.get_context('spawn').Pool(8, initializer=initialize, initargs=(split,)) as pool:
            for record in pool.imap_unordered(case, range(COUNTS[split]), chunksize=1):
                records.append(record)
                if len(records) % 50 == 0:
                    write(R / 'LABEL_GEOMETRY_STATUS.json', {'utc': now(), 'split': split,
                          'completed': len(records), 'total': COUNTS[split], 'running': True,
                          'new_model_forward': False, 'training_started': False})
        records.sort(key=lambda c: c['case_id'])
        if len(records) != COUNTS[split] or len({c['case_id'] for c in records}) != COUNTS[split]:
            raise RuntimeError('incomplete case set')
        output[split] = records
        print(json.dumps({'split': split, 'scans': len(records), 'seconds': time.monotonic() - tic}), flush=True)
    write(R / 'LABEL_GEOMETRY_CASES.json', {'utc': now(), 'splits': output})
    result = {'utc': now(), 'status': 'complete', 'raster': 'N4 braincrop resized to 128^3 in all cohorts; cube-edge proximity is not native brain-boundary proximity',
              'summary': {split: summarize(cases) for split, cases in output.items()},
              'cases_sha256': sha(R / 'LABEL_GEOMETRY_CASES.json'),
              'test_label_pixels_access': True, 'test_tuned': True,
              'new_model_forward': False, 'training_started': False,
              'interpretation_limit': 'Connected components <=5 voxels are geometrical fragments, not verified clinical lesions or safe deletion targets.'}
    write(R / 'LABEL_GEOMETRY_RESULT.json', result)
    write(R / 'LABEL_GEOMETRY_STATUS.json', {'utc': now(), 'running': False, 'status': 'complete',
          'training_started': False, 'new_model_forward': False,
          'result_sha256': sha(R / 'LABEL_GEOMETRY_RESULT.json')})


if __name__ == '__main__':
    main()
