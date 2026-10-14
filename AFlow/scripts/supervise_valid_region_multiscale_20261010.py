"""Exclusive 2-GPU native training supervisor with bounded engineering resume."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'archive/research_records/2026-10-10_segresnet_valid_region_supervision'
OUT = ROOT / 'workspace/BraTS/segresnet_t1ce_wt/valid_region_20261010'
PY = ROOT.parent / 'sammed3d-venv/bin/python'
MAX_WALL_SECONDS = 14 * 3600
MAX_ENGINEERING_RESTARTS = 2


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def verify():
    if read(R / 'DATA_AUDIT.json')['status'] != 'passed' or read(R / 'GPU_SMOKE.json')['status'] != 'passed':
        raise RuntimeError('prerequisite not passed')
    freeze = read(R / 'EXECUTION_FREEZE.json')
    for relative, expected in freeze['source_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError('frozen source changed: ' + relative)
    if sha(R / 'DATA_AUDIT.json') != freeze['data_audit_sha256']:
        raise RuntimeError('frozen cache audit changed')
    if sha(R / 'GPU_SMOKE.json') != freeze['gpu_smoke_sha256']:
        raise RuntimeError('frozen GPU smoke changed')


def main():
    lock = (R / 'SUPERVISOR.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify()
    if (R / 'TRAIN_STOP.json').exists():
        raise RuntimeError('research training already finished; no restart')
    started = time.monotonic()
    for attempt in range(1, MAX_ENGINEERING_RESTARTS + 2):
        if attempt > 1 and not (OUT / 'latest.pth').exists():
            raise RuntimeError('no complete epoch checkpoint for controlled resume')
        log = R / f'train_attempt{attempt}.log'
        command = [str(PY), '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                   'scripts/train_valid_region_multiscale_20261010_v2.py', 'train']
        env = os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR='/tmp/aflow_matplotlib_cache',
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with log.open('ab') as stream:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write(R / f'LAUNCH_ATTEMPT{attempt}.json', {'utc': now(), 'attempt': attempt, 'worker': 1127648,
              'pid': proc.pid, 'command': command, 'freeze_sha256': sha(R / 'EXECUTION_FREEZE.json'),
              'resume_from_latest': attempt > 1, 'test_pixels_access': False})
        write(R / 'PIPELINE_STATUS.json', {'utc': now(), 'state': 'training', 'running': True,
              'worker': 1127648, 'attempt': attempt, 'pid': proc.pid, 'test_pixels_access': False})
        while proc.poll() is None:
            state = read(OUT / 'STATUS.json') if (OUT / 'STATUS.json').exists() else {}
            write(R / 'HEARTBEAT.json', {'utc': now(), 'worker': 1127648, 'supervisor_pid': os.getpid(),
                  'train_pid': proc.pid, 'train_alive': True, 'attempt': attempt,
                  'elapsed_seconds': time.monotonic() - started,
                  'last_complete_epoch': state.get('last_complete_epoch', 0),
                  'new_patient_training': True, 'new_test_model_forward': False,
                  'epoch': state.get('epoch'), 'step': state.get('step'),
                  'best_epoch': state.get('best_epoch'), 'best_scan': state.get('best_scan'),
                  'test_pixels_access': False})
            if time.monotonic() - started > MAX_WALL_SECONDS:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=60)
                write(R / 'PIPELINE_STATUS.json', {'utc': now(), 'state': 'stopped_wall_budget',
                      'running': False, 'last_complete_epoch': state.get('last_complete_epoch', 0),
                  'new_patient_training': True, 'new_test_model_forward': False,
                      'training_started': True, 'test_pixels_access': False})
                raise RuntimeError('14h wall budget exceeded')
            time.sleep(30)
        if (R / 'TRAIN_STOP.json').exists() and proc.returncode == 0:
            receipt = read(R / 'TRAIN_STOP.json')
            write(R / 'PIPELINE_STATUS.json', {'utc': now(), 'state': 'training_complete' if receipt['last_complete_epoch'] == 200 else 'training_stopped_at_gate',
                  'running': False, 'worker': 1127648, 'attempt': attempt,
                  'last_complete_epoch': receipt['last_complete_epoch'], 'best_epoch': receipt['best_epoch'],
                  'best_scan': receipt['best_scan'], 'test_pixels_access': False})
            final_log = R / 'finalize.log'
            with final_log.open('ab') as stream:
                final = subprocess.run([str(PY), 'scripts/finalize_valid_region_multiscale_20261010.py'],
                                       cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT)
            if final.returncode or not (R / 'COMPLETION_AUDIT.json').exists():
                write(R / 'PIPELINE_STATUS.json', {'utc': now(), 'state': 'publication_failed_after_training',
                      'running': False, 'training_complete': True, 'worker': 1127648,
                      'last_complete_epoch': receipt['last_complete_epoch'],
                      'test_pixels_access': False})
                raise RuntimeError('training completed but final publication failed; do not retrain')
            write(R / 'PIPELINE_STATUS.json', {'utc': now(), 'state': 'complete', 'running': False,
                  'worker': 1127648, 'last_complete_epoch': receipt['last_complete_epoch'],
                  'best_epoch': receipt['best_epoch'], 'best_scan': receipt['best_scan'],
                  'promotion_gate_passed': read(R / 'FINAL_RESULT.json')['promotion_gate_passed'],
                  'test_pixels_access': False, 'completion_audit_sha256': sha(R / 'COMPLETION_AUDIT.json')})
            return
        write(R / f'ENGINEERING_FAILURE_ATTEMPT{attempt}.json', {'utc': now(), 'exit_code': proc.returncode,
              'last_complete_epoch': read(OUT / 'STATUS.json').get('last_complete_epoch', 0) if (OUT / 'STATUS.json').exists() else 0,
              'train_stop_exists': (R / 'TRAIN_STOP.json').exists(),
              'latest_exists': (OUT / 'latest.pth').exists()})
    write(R / 'PIPELINE_STATUS.json', {'utc': now(), 'state': 'failed_after_engineering_retries',
          'running': False, 'worker': 1127648, 'test_pixels_access': False})
    raise RuntimeError('bounded engineering retries exhausted')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        write(R / 'SUPERVISOR_FAILURE.json', {'utc': now(), 'type': type(exc).__name__, 'reason': str(exc)})
        raise
