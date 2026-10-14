"""Fixed one-shot, native-grid test of the validation-selected multiscale model."""
import json
from pathlib import Path

import numpy as np
import torch

from scripts.followup_test270_common import (
    ROOT, MANIFEST, CANONICAL_TEST, TEST_CANVAS, SCHEMA, NativeTestDataset,
    case_metrics, pad_to_canvas, model_for, seed_all, numeric_mode,
    summary_ci, paired, read_rows, validate_rows, sha, write, write_new, now,
)
from scripts.train_t1ce_wt_capg3d import summarize_cases

R = ROOT / 'archive/research_records/2026-09-29_segresnet_multiscale_e166_test270'
TRAIN_RECORD = ROOT / 'archive/research_records/2026-09-29_segresnet_native_multiscale_resume50'
SOURCE = ROOT / 'workspace/BraTS/segresnet_t1ce_wt/native_multiscale_resume50_20260929/best.pth'
CHECKPOINT = R / 'checkpoint_epoch166.pth'
CHECKPOINT_SHA = '0645c5e781571ceff2e6d26c62ca52e12e985cf2c1b097f07d7b8a0bc7b48ce2'
CACHE = Path('/tmp/aflow_multiscale_e166_test270_20260929/native_test')
OLD_NATIVE = ROOT / 'archive/research_records/2026-09-20_segresnet_native_res_e173_test270'
OLD_COMBO = ROOT / 'archive/research_records/2026-09-21_segresnet_pretrain_native_e166_test270'
ID = 'segresnet_multiscale_native_epoch166_test270_20260929'


def read(path):
    return json.loads(Path(path).read_text())


def verify():
    frozen = read(R / 'FREEZE.json')
    for name, digest in frozen['files'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('frozen file changed: ' + name)
    if sha(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError('selected checkpoint changed')
    return frozen


def status(stage, **details):
    write(R / 'PIPELINE_STATUS.json', {'utc': now(), 'stage': stage, **details})
    print(json.dumps({'utc': now(), 'stage': stage, **details}, ensure_ascii=False), flush=True)


def load_model(device):
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    if state['epoch'] != 166:
        raise RuntimeError('selected epoch mismatch')
    model = model_for(device)
    model.load_state_dict(state['model_state_dict'], strict=True)
    model.eval().requires_grad_(False)
    return model


def references():
    combo = read(OLD_COMBO / 'RAW_TEST270.json')['summary']['per_case']
    native = read(OLD_NATIVE / 'RAW_TEST270.json')
    refs = {'old_combo166_native': combo,
            'native173': native['candidate']['summary']['per_case'],
            'fixed_pure193_native': native['baseline_e193']['native_trilinear_logits']['per_case']}
    for rows in refs.values():
        validate_rows(rows)
    return refs
