"""Bounded read-only utilization monitor; never starts work or extends budgets."""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import subprocess
import time

from .governance import RECORD, SCRATCH, atomic_json, ROOT

REMOTE = '''import json,subprocess,shutil,time
q=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
print(json.dumps(dict(at_unix=time.time(),gpu_csv=q,scratch_free_bytes=shutil.disk_usage('/tmp').free)))'''


def main():
    auth = json.loads((RECORD / 'AUTHORIZATION.json').read_text())
    SCRATCH.mkdir(parents=True, exist_ok=True)
    latest = {}
    last_busy = {}
    total = {}
    active = {}
    deadline = max(value['deadline_unix'] for value in auth['workers'].values())
    while time.time() < deadline:
        for worker, value in auth['workers'].items():
            if time.time() >= value['deadline_unix']:
                continue
            observed = {'worker': worker, 'at_unix': time.time()}
            try:
                result = subprocess.run(['mlx', 'worker', 'login', worker, '--', 'python3', '-c', REMOTE],
                                        text=True, capture_output=True, timeout=45)
                if result.returncode:
                    raise RuntimeError(result.stderr[-800:] or result.stdout[-800:])
                data = json.loads(next(line for line in reversed(result.stdout.splitlines()) if line.startswith('{')))
                observed.update(data)
                gpus = []
                for line in data['gpu_csv'].strip().splitlines():
                    index, uuid, name, utilization, memory, capacity = [item.strip() for item in line.split(',')]
                    key = f'{worker}:{index}'
                    busy = int(utilization) > 0
                    last_busy.setdefault(key, time.time())
                    if busy:
                        last_busy[key] = time.time()
                    total[key] = total.get(key, 0) + 1
                    active[key] = active.get(key, 0) + int(busy)
                    idle = time.time() - last_busy[key]
                    gpus.append(dict(index=int(index), uuid=uuid, name=name, utilization=int(utilization),
                                     memory_mib=int(memory), capacity_mib=int(capacity),
                                     observed_active_fraction=active[key] / total[key],
                                     seconds_since_observed_gpu_activity=idle,
                                     idle_approaching_reclamation=idle >= 3 * 3600))
                observed['gpus'] = gpus
                observed.pop('gpu_csv', None)
            except Exception as error:
                observed['error'] = str(error)
            latest[worker] = observed
            with (SCRATCH / 'resource_samples.jsonl').open('a') as stream:
                stream.write(json.dumps(observed, ensure_ascii=False) + '\n')
        atomic_json(RECORD / 'RESOURCE_STATUS.json', dict(
            at_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            pid=os.getpid(), state='monitoring', workers=latest,
            persistent_free_bytes=__import__('shutil').disk_usage(ROOT).free,
            utilization_note='60s point samples; active means observed utilization >0, not continuous GPU occupancy',
            raw_samples=str(SCRATCH / 'resource_samples.jsonl')))
        time.sleep(min(60, max(0, deadline - time.time())))
    atomic_json(RECORD / 'RESOURCE_MONITOR_COMPLETE.json', {'state': 'deadline_reached', 'at_unix': time.time()})


if __name__ == '__main__':
    main()
