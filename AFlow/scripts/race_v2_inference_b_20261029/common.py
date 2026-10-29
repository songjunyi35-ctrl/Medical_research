"""New-stage w32/RACE validation fusion. No historical launcher or training."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / 'archive/research_records/2026-10-29_race_v2_inference_b'
STAGE = ROOT / 'archive/research_records/2026-10-29_race_v2_stage'
TMP = Path('/tmp/aflow_race_v2_20261029')
OUT = TMP / 'inference_b'
A = TMP / 'inference_a'
CHECKPOINT = ROOT / 'archive/research_records/2026-10-28_segresnet_w32_e171_test270/checkpoint_epoch171.pth'
CHECKPOINT_SHA = '02a60b2d37da0d8de8018c221e1c00566106bd21d0d713b95ac3d9f94d87fade'
RACE_SHA = '4bd6c35f7e12ecdf1540bf88128f9d3f9853340d801b32e77779fc4ba56eb3d7'
MANIFEST = ROOT / 'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/validation_t1ce_wt.jsonl'
RECIPE = ROOT / 'archive/research_records/2026-10-29_race_e164_inference_test270/SELECTED_RECIPE.json'
CANVAS = (160, 208, 160)

def now(): return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for value in iter(lambda: handle.read(8 << 20), b''): h.update(value)
    return h.hexdigest()
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)
def exclusive(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle: json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
def rows(): return [json.loads(x) for x in MANIFEST.read_text().splitlines() if x.strip()]
def slices(g): return tuple(slice(int(o), int(o)+int(e)) for o,e in zip(g['canvas_offset'],g['extent']))
def guard():
    assert shutil.disk_usage(ROOT).free >= 20 * 2**30, 'persistent reserve below 20 GiB'
    assert sum(p.stat().st_size for p in R.rglob('*') if p.is_file()) <= 64 * 2**20, 'B text budget exceeded'
    assert shutil.disk_usage(TMP).free >= 70 * 2**30, 'scratch reserve below 70 GiB'
    # Current user's confirmed first 24 hours; extensions require root amendment.
    assert time.time() < 1793336374, 'worker1154121 initial 24-hour deadline reached'
    authorization = STAGE / 'AUTHORIZATION.json'
    if authorization.exists():
        deadline = read(authorization).get('deadline_unix')
        if deadline is not None: assert time.time() < deadline, 'new-stage resource deadline reached'
def numeric_mode():
    import torch
    torch.set_num_threads(2); torch.manual_seed(20261029)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True
def gpu_lock():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1', 'B owns physical GPU1 exclusively'
    folder = TMP / 'locks'; folder.mkdir(parents=True, exist_ok=True)
    handle = (folder / 'worker1154121_gpu1.lock').open('a+')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    uuid = subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    metadata = dict(at=now(),worker=1154121,physical_gpu=1,gpu_uuid=uuid,pid=os.getpid(),owner='inference_b',
                    unique_queue=True,lock_path=str(folder/'worker1154121_gpu1.lock'))
    handle.seek(0); handle.truncate(); handle.write(json.dumps(metadata)); handle.flush()
    write(R / 'GPU_QUEUE.json', metadata)
    return handle
def verify():
    guard(); amendment=R/'EXECUTION_FREEZE_V3.json'
    frozen = read(amendment if amendment.exists() else R/'EXECUTION_FREEZE_V2.json')
    for path,digest in frozen['source_sha256'].items(): assert sha(ROOT/path)==digest,path
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    assert sha(MANIFEST)==frozen['manifest_sha256']
    assert sha(RECIPE)==frozen['recipe_sha256']
def data(cache):
    import numpy as np
    cache=Path(cache); index=read(cache/'index.json'); cohort=rows()
    assert len(cohort)==108 and len({r['subject_id'] for r in cohort})==50
    assert index['case_ids']==[r['case_id'] for r in cohort]
    assert tuple(index['canvas'])==CANVAS and index['split']=='validation'
    count=108*int(np.prod(CANVAS))
    assert (cache/'images.f32').stat().st_size==count*4 and (cache/'labels.u8').stat().st_size==count
    historical=read(ROOT/'archive/research_records/2026-10-28_race_seg/DATA_validation.json')
    ready=read(cache.parent/'CACHE_READY.json')
    assert ready['all_four_payloads_exact'] and ready['historical_geometry_exact']
    assert ready['payload_sha256']==historical['digests']==index['digests']
    assert ready['index_sha256']==sha(cache/'index.json')
    assert index['geometry']==historical['geometry']
    assert sha(cache/'images.f32')==historical['digests']['images.f32']
    assert sha(cache/'labels.u8')==historical['digests']['labels.u8']
    return index,np.memmap(cache/'images.f32',np.float32,'r',shape=(108,*CANVAS)),np.memmap(cache/'labels.u8',np.uint8,'r',shape=(108,*CANVAS))
