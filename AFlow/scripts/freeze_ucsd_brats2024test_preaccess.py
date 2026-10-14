#!/usr/bin/env python3
"""Freeze MODEL_PREACCESS for the UCSD BraTS-2024-test panel after cache+smoke, before any model on cohort pixels."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts import ucsd_brats2024test_protocol as p


def main():
    if p.MODEL_PREACCESS.exists():
        raise FileExistsError(p.MODEL_PREACCESS)
    p.docs()
    cache = json.loads(p.CACHE_RECEIPT.read_text())
    smoke = json.loads((p.RECORD / 'SMOKE_RECEIPT.json').read_text())
    if cache.get('all_gates_passed') is not True or smoke.get('status') != 'passed':
        raise ValueError('cache/smoke not passed')
    evaluator = p.ROOT / 'scripts/evaluate_ucsd_brats2024test_panel.py'
    doc = {'schema': 'aflow.ucsd_brats2024test.model_preaccess', 'schema_version': 1,
           'status': 'frozen_after_cache_before_model',
           'created_at_utc': datetime.now(timezone.utc).isoformat(),
           'access': {'model_inference': False, 'cohort_metrics_revealed': False},
           'cohort_sha256': p.COHORT_SHA,
           'cache_receipt_sha256': p.sha(p.CACHE_RECEIPT),
           'smoke_receipt_sha256': p.sha(p.RECORD / 'SMOKE_RECEIPT.json'),
           'source_receipt_sha256': p.sha(p.SOURCE_RECEIPT),
           'label_scan_sha256': p.sha(p.RECORD / 'RAW_LABEL_SCAN.json'),
           'first_probe_sha256': p.sha(p.FIRST_PROBE),
           'checkpoints': {'full972_kd168': p.FULL_SHA, 'gcsam184': p.GCSAM_SHA},
           'views': {'full972': ['identity', 'fixed_flip8'], 'gcsam184': ['identity']},
           'threshold': 0.0, 'bootstrap_samples': p.BOOTSTRAP_SAMPLES,
           'seeds': {'abs_full_identity': p.SEED_ABS_FULL_IDENTITY, 'abs_full_flip8': p.SEED_ABS_FULL_FLIP8,
                     'abs_gcsam': p.SEED_ABS_GCSAM, 'paired_flip8': p.SEED_PAIRED_FLIP8,
                     'paired_gcsam': p.SEED_PAIRED_GCSAM},
           'code': {'evaluator_sha256': p.sha(evaluator), 'protocol_sha256': p.sha(p.ROOT / 'scripts/ucsd_brats2024test_protocol.py')},
           'one_time_only': True, 'result_must_not_trigger_tuning_or_repeat': True}
    p.write_new(p.MODEL_PREACCESS, doc)
    print(json.dumps(doc, indent=2))


if __name__ == '__main__':
    main()
