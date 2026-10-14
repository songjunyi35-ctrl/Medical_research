"""Build only fit RC labels; verify rebuilt WT against the inherited binary cache."""
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F
from scripts import structured_background_protocol as p

CONFIG = {}


def init(config):
    torch.set_num_threads(1)
    CONFIG.update(config)
    CONFIG['fd'] = os.open(config['output'], os.O_WRONLY)
    CONFIG['wt'] = np.memmap(config['wt_path'], dtype=np.uint8, mode='r', shape=(972,128,128,128))


def worker(task):
    slot, row, geometry, original_slot = task
    path = Path(row['label_path'])
    digest = p.sha(path)
    nii = nib.as_closest_canonical(nib.load(str(path)))
    raw = np.asarray(nii.dataobj)
    if list(raw.shape) != geometry['native_shape'] or not np.isfinite(raw).all():
        raise ValueError('label shape/finite')
    rounded = np.rint(raw)
    if not np.allclose(raw, rounded, atol=1e-5, rtol=0) or not set(np.unique(rounded)).issubset({0,1,2,3,4}):
        raise ValueError('raw label domain')
    crop = tuple(slice(a, a+b) for a,b in zip(geometry['origin'], geometry['extent']))
    raw = rounded[crop]
    stacked = np.stack((np.isin(raw, [1,2,3]), raw == 4)).astype(np.float32)
    resized = F.interpolate(torch.from_numpy(stacked)[None], size=(128,128,128), mode='nearest-exact')[0].numpy().astype(np.uint8)
    if not np.array_equal(resized[0], CONFIG['wt'][original_slot]) or np.any(resized[0] & resized[1]):
        raise ValueError('WT byte parity or RC disjointness')
    body = resized[1].tobytes()
    assert os.pwrite(CONFIG['fd'], body, slot * 128**3) == len(body)
    return {'case_id': row['case_id'], 'source_label_sha256': digest, 'rc_array_sha256': hashlib.sha256(body).hexdigest(), 'rc_voxels': int(resized[1].sum()), 'wt_byte_exact': True}


def main():
    fit, _ = p.check_roles()
    p.checked(p.CACHE / 'index.json', p.INDEX_SHA)
    index = json.loads((p.CACHE / 'index.json').read_text())
    p.checked(p.CACHE / 'labels.u8', index['labels_sha256'])
    p.checked(p.CACHE / 'images.f32', index['images_sha256'])
    output = p.WORK / 'fit_aux/rc.u8'
    if output.exists() or (p.RECORD / 'AUX_CACHE_RECEIPT.json').exists():
        raise FileExistsError('cache already built')
    p.WORK.mkdir(parents=True, exist_ok=True)
    needed = len(fit) * 128**3
    if shutil.disk_usage(p.WORK).free < needed + 2*1024**3:
        raise RuntimeError('disk reserve')
    output.parent.mkdir()
    with output.open('xb') as f:
        f.truncate(needed)
    lookup = {cid:i for i,cid in enumerate(index['case_ids'])}
    tasks = [(i,r,index['crop_geometry'][r['case_id']],lookup[r['case_id']]) for i,r in enumerate(fit)]
    start = time.monotonic()
    config = {'output': str(output), 'wt_path': str(p.CACHE / 'labels.u8')}
    result = []
    with mp.get_context('spawn').Pool(8, initializer=init, initargs=(config,)) as pool:
        for row in pool.imap_unordered(worker, tasks):
            result.append(row)
            if len(result)%100 == 0:
                print(f'RC fit cache {len(result)}/{len(fit)}', flush=True)
    result.sort(key=lambda r:r['case_id'])
    p.write_new(p.RECORD / 'AUX_CACHE_RECEIPT.json', {'status':'passed', 'fit_sha256':p.FIT_SHA, 'cases':684, 'subjects':313, 'rc_file':str(output), 'rc_sha256':p.sha(output), 'bytes':needed, 'source_cache_index_sha256':p.INDEX_SHA, 'source_images_sha256':index['images_sha256'], 'source_labels_sha256':index['labels_sha256'], 'all_wt_byte_exact':True, 'fit_cases_with_rc':sum(r['rc_voxels']>0 for r in result), 'details':result, 'elapsed_seconds':time.monotonic()-start, 'dev_label4_access':False, 'model_forwards':0, 'builder_sha256':p.sha(Path(__file__))})
    print('fit-only auxiliary cache passed', flush=True)


if __name__ == '__main__':
    main()
