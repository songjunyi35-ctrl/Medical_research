"""Observe current queues and audit complete validation outputs, without reruns."""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .governance import ROOT, RECORD, atomic_json, sha


def read_if(path):
    return json.loads(path.read_text()) if path.is_file() else None


def snapshot():
    folders = {name: ROOT / 'archive/research_records' / f'2026-10-29_race_v2_{name}'
               for name in ('training', 'inference_a', 'inference_b')}
    output = {}
    for name, folder in folders.items():
        output[name] = {}
        for filename in ('PIPELINE_STATUS.json', 'STATUS.json', 'HEARTBEAT.json', 'PREPARE_HEARTBEAT.json',
                         'CACHE_HEARTBEAT.json', 'ANALYSIS_HEARTBEAT.json', 'FORWARD_PROGRESS.json',
                         'STARTUP_VERIFIED.json', 'DONE.json'):
            value = read_if(folder / filename)
            if value is not None:
                output[name][filename] = {key: val for key, val in value.items()
                                         if key in ('at', 'state', 'phase', 'stage', 'new_epoch', 'iteration',
                                                    'new_updates', 'completed', 'count', 'done', 'total',
                                                    'case_forwards', 'best_new_epoch', 'best_scan', 'metrics')}
        # Failure receipts remain visible even if a later amendment resolved them.
        output[name]['failure_receipts'] = [str(p.relative_to(ROOT)) for p in sorted(folder.glob('*FAILURE*.json'))]
    return output


def main():
    authorization = json.loads((RECORD / 'AUTHORIZATION.json').read_text())
    end = max(worker['deadline_unix'] for worker in authorization['workers'].values())
    reports = [ROOT / 'archive/research_records/2026-10-29_race_v2_inference_a/RAW_VALIDATION108.json',
               ROOT / 'archive/research_records/2026-10-29_race_v2_inference_b/RAW_VALIDATION.json']
    while time.time() < end:
        state = snapshot()
        if all(path.exists() for path in reports) and not (RECORD / 'VALIDATION_ARITHMETIC_AUDIT.json').exists():
            marker = RECORD / 'VALIDATION_AUDIT_ATTEMPT.json'
            if not marker.exists():
                atomic_json(marker, {'at_unix': time.time(), 'inputs': {str(p): sha(p) for p in reports}})
                result = subprocess.run([sys.executable, '-m', 'scripts.race_v2_stage_20261029.audit_validation',
                                         *map(str, reports)], cwd=ROOT, capture_output=True, text=True, timeout=90)
                atomic_json(RECORD / 'VALIDATION_AUDIT_EXIT.json',
                            {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr[-4000:]})
        atomic_json(RECORD / 'STAGE_STATUS.json', dict(
            at_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            state='active_new_stage', supervisor_pid=os.getpid(), queues=state,
            validation_audit_passed=(RECORD / 'VALIDATION_ARITHMETIC_AUDIT.json').exists(),
            final_test_selection_frozen=(RECORD / 'FINAL_TEST_SELECTION.json').exists(),
            test270_scoring_automatically_started=False,
            note='Observer only; queue progress is not proof of achieved0.8; no old task restarts'))
        time.sleep(min(60, max(0, end - time.time())))
    atomic_json(RECORD / 'SUPERVISION_DEADLINE.json', {'at_unix': time.time(), 'state': 'initial_deadline_reached'})


if __name__ == '__main__':
    main()
