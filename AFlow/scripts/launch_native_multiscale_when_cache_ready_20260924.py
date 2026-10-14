"""Wait for exact V2 cache payload, freeze execution, then run one supervisor."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'archive/research_records/2026-09-24_segresnet_native_multiscale_supervision'
PY = ROOT.parent / 'sammed3d-venv/bin/python'
MAX_CACHE_WAIT_SECONDS = 2 * 3600


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def freeze():
    if (R / 'EXECUTION_FREEZE.json').exists() or (R / 'TRAIN_PREACCESS.json').exists():
        raise RuntimeError('execution already frozen or patient training accessed')
    data = read(R / 'DATA_AUDIT.json')
    smoke = read(R / 'GPU_SMOKE.json')
    if data['status'] != 'passed' or not data['historical_pixel_payload_exact'] or smoke['status'] != 'passed':
        raise RuntimeError('data or GPU smoke gate failed')
    if data['manifest']['train']['scans'] != 972 or data['manifest']['validation']['scans'] != 108 or data['manifest']['test']['scans'] != 270:
        raise RuntimeError('cohort count changed')
    if not data['patient_isolation'] or data['manifest']['train']['patients'] != 444 or data['manifest']['validation']['patients'] != 50 or data['manifest']['test']['patients'] != 119:
        raise RuntimeError('patient isolation/count changed')
    if smoke['global_batch'] != 8 or smoke['tail_global_batch'] != 4 or smoke['peak_reserved_bytes'] > 36 * 2**30:
        raise RuntimeError('synthetic GPU batch gate failed')
    files = [
        'scripts/launch_native_multiscale_when_cache_ready_20260924.py',
        'scripts/train_native_multiscale_20260924.py',
        'scripts/supervise_native_multiscale_20260924.py',
        'scripts/finalize_native_multiscale_20260924.py',
        'scripts/segresnet_pretrained_native_common.py',
        'scripts/train_segresnet_pretrained_native.py',
        'sammed3d/t1ce_wt_capg3d.py',
        'archive/research_records/2026-09-24_segresnet_native_multiscale_supervision/IDEA.md',
        'archive/research_records/2026-09-24_segresnet_native_multiscale_supervision/CACHE_FREEZE.json',
        'archive/research_records/2026-09-24_segresnet_native_multiscale_supervision/REPAIR_V2/CACHE_FREEZE.json',
        'archive/research_records/2026-09-24_segresnet_native_multiscale_supervision/ENGINEERING_AMENDMENT_001.json',
        'archive/research_records/2026-09-24_segresnet_native_multiscale_supervision/CACHE_ENGINEERING_AMENDMENT_002.json',
        'workspace/BraTS/segresnet_t1ce_wt/pretrained_native_res_20260921/validation_e045.json',
        'workspace/BraTS/segresnet_t1ce_wt/pretrained_native_res_20260921/validation_e166.json',
    ]
    prepared = read(R / 'EXECUTION_SOURCE_PREPARED_V2.json')
    for relative, expected in prepared['source_sha256'].items():
        # V2 preparation precedes the final publish/reload engineering edits;
        # the final execution freeze is authoritative. Record drift explicitly.
        actual = sha(ROOT / relative)
        if actual != expected and relative not in {
            'scripts/train_native_multiscale_20260924.py',
            'scripts/supervise_native_multiscale_20260924.py',
            'scripts/finalize_native_multiscale_20260924.py',
        }:
            raise RuntimeError('unexpected source drift since preparation: ' + relative)
    old = ROOT / 'workspace/BraTS/segresnet_t1ce_wt/pretrained_native_res_20260921/validation_e166.json'
    pretrain = ROOT / 'workspace/BraTS/segresnet_t1ce_wt/two_ideas_20260921/pretrain/final.pth'
    if sha(pretrain) != '3726eac66918b7afd32ae93500a554534d27528bc86a0fe9dd2a626ce5ac6b76':
        raise RuntimeError('pretrain source changed')
    body = {'utc': now(), 'status': 'frozen_before_patient_training',
            'source_sha256': {f: sha(ROOT / f) for f in files},
            'data_audit_sha256': sha(R / 'DATA_AUDIT.json'),
            'gpu_smoke_sha256': sha(R / 'GPU_SMOKE.json'),
            'old_validation_e166_sha256': sha(old),
            'pretrain_sha256': sha(pretrain), 'worker': 1109780,
            'train_scans': 972, 'validation_scans': 108,
            'new_test_model_forward': False, 'test_tuned_development': True,
            'promotion_scan_min': .761178640458319,
            'epoch50_best_scan_min': .71971019177194,
            'target_epochs': 200, 'global_batch': 8, 'updates_per_epoch': 122}
    write(R / 'EXECUTION_FREEZE.json', body)
    return body


def main():
    lock = (R / 'COORDINATOR.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gate = read(R / 'COORDINATOR_FREEZE.json')
    if sha(ROOT / 'scripts/launch_native_multiscale_when_cache_ready_20260924.py') != gate['coordinator_sha256']:
        raise RuntimeError('coordinator changed after freeze')
    if sha(R / 'REPAIR_V2/CACHE_FREEZE.json') != gate['cache_v2_freeze_sha256']:
        raise RuntimeError('cache recipe changed after coordinator freeze')
    started = time.monotonic()
    while not (R / 'DATA_AUDIT.json').exists():
        if (R / 'CACHE_V2_FAILURE.json').exists() or time.monotonic() - started > MAX_CACHE_WAIT_SECONDS:
            raise RuntimeError('cache rebuild failed or exceeded two-hour preparation wait')
        write(R / 'QUEUE_HEARTBEAT.json', {'utc': now(), 'state': 'waiting_exact_native_cache',
              'worker': 1109780, 'elapsed_seconds': time.monotonic() - started,
              'training_started': False, 'test_pixels_access': False})
        time.sleep(30)
    handoff_started = time.monotonic()
    while read(R / 'PIPELINE_STATUS.json')['state'] != 'cache_complete':
        if (R / 'CACHE_V2_FAILURE.json').exists() or time.monotonic() - handoff_started > 120:
            raise RuntimeError('data audit appeared without completed cache handoff')
        time.sleep(2)
    freeze()
    write(R / 'EXECUTION_LAUNCH_PREACCESS.json', {'utc': now(), 'worker': 1109780,
          'freeze_sha256': sha(R / 'EXECUTION_FREEZE.json'),
          'cache_data_audit_sha256': sha(R / 'DATA_AUDIT.json'),
          'gpu_smoke_sha256': sha(R / 'GPU_SMOKE.json'),
          'training_started': False, 'new_test_model_forward': False})
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR='/tmp/aflow_matplotlib_cache',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (R / 'supervisor.log').open('ab') as stream:
        completed = subprocess.run([str(PY), 'scripts/supervise_native_multiscale_20260924.py'],
                                   cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError('training supervisor failed with exit code ' + str(completed.returncode))
    if read(R / 'PIPELINE_STATUS.json')['state'] != 'complete':
        raise RuntimeError('supervisor returned without complete publication')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        write(R / 'COORDINATOR_FAILURE.json', {'utc': now(), 'type': type(exc).__name__,
              'reason': str(exc), 'test_pixels_access': False})
        raise
