"""Exclusive continuation queue: verified cache and epoch50 state -> epoch51."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'archive/research_records/2026-09-29_segresnet_native_multiscale_resume50'
OUT = ROOT / 'workspace/BraTS/segresnet_t1ce_wt/native_multiscale_resume50_20260929'
OLD_OUT = ROOT / 'workspace/BraTS/segresnet_t1ce_wt/native_multiscale_20260924'
PY = ROOT.parent / 'sammed3d-venv/bin/python'
RESUME_SHA = '93a969ad592849e2d057111c0111837726cb44b9bd126668aa6567faa6dba8a6'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for data in iter(lambda: stream.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def freeze_execution():
    if (R / 'EXECUTION_FREEZE.json').exists() or (R / 'RESUME_PREACCESS.json').exists():
        raise RuntimeError('continuation already frozen or started')
    if sha(OUT / 'latest.pth') != RESUME_SHA or sha(OLD_OUT / 'latest.pth') != RESUME_SHA:
        raise RuntimeError('epoch50 full-state source or copied initialization changed')
    cache = read(R / 'DATA_AUDIT.json')
    smoke = read(R / 'GPU_SMOKE.json')
    if cache['status'] != 'passed' or not cache['historical_pixel_payload_exact'] or not cache['patient_isolation']:
        raise RuntimeError('cache or patient isolation gate failed')
    if smoke['status'] != 'passed' or smoke['resume_epoch'] != 50 or smoke['model_source_sha256'] != RESUME_SHA:
        raise RuntimeError('resumed full-state GPU smoke failed')
    if smoke['global_batch'] != 8 or smoke['tail_global_batch'] != 4 or smoke['peak_reserved_bytes'] > 64 * 2**30:
        raise RuntimeError('batch or A10080GB engineering memory gate failed')
    if read(R / 'SMOKE_EXIT_VERIFIED.json')['status'] != 'passed':
        raise RuntimeError('synthetic CUDA processes have not exited')
    files = [
        'scripts/launch_multiscale_resume50_20260929.py',
        'scripts/train_native_multiscale_resume50_20260929.py',
        'scripts/train_native_multiscale_resume50_20260929_v2.py',
        'scripts/supervise_native_multiscale_resume50_20260929.py',
        'scripts/supervise_native_multiscale_resume50_20260929_v2.py',
        'scripts/finalize_native_multiscale_resume50_20260929.py',
        'scripts/scaffold_multiscale_resume50_20260929.py',
        'scripts/amend_resume_smoke_memory_20260929.py',
        'scripts/build_native_multiscale_resume50_cache_20260929.py',
        'scripts/segresnet_pretrained_native_common.py',
        'scripts/train_segresnet_pretrained_native.py',
        'sammed3d/t1ce_wt_capg3d.py',
        'archive/research_records/2026-09-29_segresnet_native_multiscale_resume50/IDEA.md',
        'archive/research_records/2026-09-29_segresnet_native_multiscale_resume50/BUDGET_AMENDMENT_001.json',
        'archive/research_records/2026-09-29_segresnet_native_multiscale_resume50/HARDWARE_AMENDMENT_002.json',
        'archive/research_records/2026-09-29_segresnet_native_multiscale_resume50/RESUME_INITIALIZATION.json',
        'archive/research_records/2026-09-29_segresnet_native_multiscale_resume50/SOURCE_DERIVATION.json',
        'archive/research_records/2026-09-29_segresnet_native_multiscale_resume50/ENGINEERING_AMENDMENT_003.json',
        'archive/research_records/2026-09-29_segresnet_native_multiscale_resume50/CACHE_FREEZE.json',
        'workspace/BraTS/segresnet_t1ce_wt/native_multiscale_20260924/latest.pth',
        'workspace/BraTS/segresnet_t1ce_wt/pretrained_native_res_20260921/validation_e045.json',
        'workspace/BraTS/segresnet_t1ce_wt/pretrained_native_res_20260921/validation_e166.json',
    ]
    derived = read(R / 'SOURCE_DERIVATION.json')['derived_sources_sha256']
    for name, expected in derived.items():
        if sha(ROOT / name) != expected:
            raise RuntimeError('derived continuation source changed: ' + name)
    for name, expected in read(R / 'ENGINEERING_AMENDMENT_003.json')['new_sources_sha256'].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError('A10080GB engineering source changed: ' + name)
    write(R / 'EXECUTION_FREEZE.json', {'utc': now(), 'status': 'frozen_before_epoch51_patient_training',
          'source_sha256': {name: sha(ROOT / name) for name in files},
          'data_audit_sha256': sha(R / 'DATA_AUDIT.json'),
          'gpu_smoke_sha256': sha(R / 'GPU_SMOKE.json'),
          'old_validation_e166_sha256': sha(ROOT / files[-1]),
          'pretrain_sha256': '3726eac66918b7afd32ae93500a554534d27528bc86a0fe9dd2a626ce5ac6b76',
          'resume_checkpoint_sha256': RESUME_SHA, 'resume_optimizer_updates': 6100,
          'worker': 1114913, 'start_epoch': 51, 'target_epochs': 200,
          'new_epochs': 150, 'new_updates': 18300, 'global_batch': 8,
          'epoch50_budget_gate_superseded_by_user': True,
          'promotion_scan_min': .761178640458319,
          'new_test_model_forward': False, 'adaptive_budget_amendment': True})


def main():
    lock = (R / 'COORDINATOR.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gate = read(R / 'COORDINATOR_FREEZE.json')
    if sha(Path(__file__)) != gate['coordinator_sha256']:
        raise RuntimeError('coordinator source changed')
    if sha(R / 'CACHE_FREEZE.json') != gate['cache_freeze_sha256']:
        raise RuntimeError('cache freeze changed')
    started = time.monotonic()
    while not all((R / name).exists() for name in ['DATA_AUDIT.json', 'GPU_SMOKE.json', 'SMOKE_EXIT_VERIFIED.json']):
        if (R / 'CACHE_FAILURE.json').exists() or time.monotonic() - started > 2 * 3600:
            raise RuntimeError('cache/smoke preparation failed or exceeded2h')
        write(R / 'QUEUE_HEARTBEAT.json', {'utc': now(), 'state': 'waiting_resume_cache_and_smoke',
              'worker': 1114913, 'original_complete_epochs': 50,
              'resume_training_started': False, 'test_pixels_access': False})
        time.sleep(30)
    for _ in range(60):
        if read(R / 'PIPELINE_STATUS.json')['state'] == 'cache_complete':
            break
        time.sleep(2)
    else:
        raise RuntimeError('cache completion handoff missing')
    freeze_execution()
    write(R / 'EXECUTION_LAUNCH_PREACCESS.json', {'utc': now(), 'worker': 1114913,
          'start_epoch': 51, 'freeze_sha256': sha(R / 'EXECUTION_FREEZE.json'),
          'resume_checkpoint_sha256': RESUME_SHA, 'new_test_model_forward': False})
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR='/tmp/aflow_matplotlib_cache',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (R / 'supervisor.log').open('ab') as stream:
        completed = subprocess.run([str(PY), 'scripts/supervise_native_multiscale_resume50_20260929_v2.py'],
                                   cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode or read(R / 'PIPELINE_STATUS.json')['state'] != 'complete':
        raise RuntimeError('continuation supervisor/publication failed; inspect without retraining completed epochs')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        write(R / 'COORDINATOR_FAILURE.json', {'utc': now(), 'reason': repr(exc),
              'worker': 1114913, 'test_pixels_access': False})
        raise
