"""Isolated inference-only search. Never imports a historical launcher or optimizer."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / 'archive/research_records/2026-10-28_segresnet_inference_search'
TMP = Path('/tmp/aflow_inference_search_20261028')
CACHE = TMP / 'validation'
PY = ROOT.parent / 'sammed3d-venv/bin/python'
CHECKPOINT = ROOT / 'archive/research_records/2026-10-28_segresnet_w32_e171_test270/checkpoint_epoch171.pth'
CHECKPOINT_SHA = '02a60b2d37da0d8de8018c221e1c00566106bd21d0d713b95ac3d9f94d87fade'
REFERENCE = ROOT / 'workspace/BraTS/segresnet_t1ce_wt/width_resume_20261027/w32/validation_e171.json'
CANVAS = (160, 208, 160)
MANIFEST = ROOT / 'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/validation_t1ce_wt.jsonl'


def now():
    return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def status(state, **kwargs):
    write(R / 'PIPELINE_STATUS.json', dict(at=now(), state=state, worker=1152682,
          training=False, test_access=False, **kwargs))


def guard():
    if time.time() >= read(R / 'AUTHORIZATION.json')['deadline_unix']:
        raise RuntimeError('8-hour inference search budget reached')
    if shutil.disk_usage(ROOT).free < 20 * 2**30:
        raise RuntimeError('persistent reserve below20GiB')
    used = sum(p.stat().st_size for p in R.rglob('*') if p.is_file())
    if used > 2 * 2**30:
        raise RuntimeError('new persistent data exceeds2GiB')
    if TMP.exists() and shutil.disk_usage(TMP).free < 30 * 2**30:
        raise RuntimeError('scratch reserve below30GiB')


def verify():
    guard()
    frozen = read(R / 'EXECUTION_FREEZE.json')
    for path, digest in frozen['source_sha256'].items():
        if sha(ROOT / path) != digest:
            raise RuntimeError('frozen source changed: ' + path)
    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    return read(R / 'PLAN.json')


def load_model(device):
    import torch
    from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == 171
    model = AutomaticPromptLocator(center_aware=False, init_filters=32, dropout=.1).to(device)
    assert sum(v.numel() for v in model.parameters()) == 18795969
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    for name, value in model.state_dict().items():
        assert torch.equal(value.cpu(), checkpoint['model_state_dict'][name])
        assert torch.isfinite(value).all()
    return model.eval().requires_grad_(False)


def numeric_mode():
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(20260819)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True


def slices(geometry):
    return tuple(slice(o, o + e) for o, e in zip(geometry['canvas_offset'], geometry['extent']))


def dataset():
    import numpy as np
    index = read(CACHE / 'index.json')
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    assert len(rows) == 108 and len({d['subject_id'] for d in rows}) == 50
    assert index['case_ids'] == [d['case_id'] for d in rows]
    assert tuple(index['canvas']) == CANVAS and index['split'] == 'validation'
    assert index['test_access'] is False and read(R / 'DATA_READY.json')['status'] == 'passed'
    images = np.memmap(CACHE / 'images.f32', np.float32, 'r', shape=(108, *CANVAS))
    labels = np.memmap(CACHE / 'labels.u8', np.uint8, 'r', shape=(108, *CANVAS))
    return rows, index['geometry'], images, labels


def build_plan():
    import itertools
    import numpy as np
    atoms = []
    for scale in [1., .85, .925, 1.075, 1.15]:
        for flip in range(8):
            name = f'scale{scale:g}_flip{flip}'
            atoms.append(dict(id=name, kind='scale_flip', scale=scale, flip=flip))
    for axis, amount in itertools.product(range(3), [-8, -4, 4, 8]):
        atoms.append(dict(id=f'shift{axis}_{amount}', kind='shift', axis=axis, amount=amount))
    for axis, degrees in itertools.product(range(3), [-5, 5]):
        atoms.append(dict(id=f'rotate{axis}_{degrees}', kind='rotate', axis=axis, degrees=degrees))
    for kind, values in [('gain', [.8, .9, 1.1, 1.2]), ('offset', [-.1, .1]), ('gamma', [.8, 1.2])]:
        for value in values:
            atoms.append(dict(id=f'{kind}{value:g}', kind=kind, value=value))
    for overlap, blend in itertools.product([.5, .75], ['gaussian', 'constant']):
        atoms.append(dict(id=f'sliding128_o{overlap}_{blend}', kind='sliding', overlap=overlap, blend=blend))
    identity = 'scale1_flip0'
    recipes = []
    for atom in atoms:
        recipes.append(dict(id=atom['id'], atoms=[atom['id']], weights=[1.], fusion='logit'))
    # All nonempty subsets; singletons are already included. Both fusion spaces are frozen.
    for bits in range(1, 256):
        selected = [f'scale1_flip{i}' for i in range(8) if bits & (1 << i)]
        if len(selected) == 1:
            continue
        for fusion in ['logit', 'probability']:
            recipes.append(dict(id=f'mirror_subset{bits:03}_{fusion}', atoms=selected,
                                weights=[1 / len(selected)] * len(selected), fusion=fusion))
    groups = {}
    for scale in [.85, .925, 1.075, 1.15]:
        groups[f'scale{scale:g}_all8'] = [a['id'] for a in atoms if a.get('scale') == scale]
        groups[f'scale{scale:g}_identity_pair'] = [identity, f'scale{scale:g}_flip0']
    groups['multiscale_identity'] = [a['id'] for a in atoms if a['kind'] == 'scale_flip' and a['flip'] == 0]
    groups['multiscale_all40'] = [a['id'] for a in atoms if a['kind'] == 'scale_flip']
    for kind in ['shift', 'rotate', 'gain', 'offset', 'gamma', 'sliding']:
        groups[kind + '_mean'] = [identity] + [a['id'] for a in atoms if a['kind'] == kind]
    for name, group in groups.items():
        for fusion in ['logit', 'probability']:
            recipes.append(dict(id=name + '_' + fusion, atoms=group,
                                weights=[1 / len(group)] * len(group), fusion=fusion))
    # Conservative identity blends of every nonidentity atom, explicit weights, no fitting.
    for atom in atoms:
        if atom['id'] == identity:
            continue
        for weight in [.25, .5, .75]:
            recipes.append(dict(id=f"blend_{atom['id']}_w{weight}", atoms=[identity, atom['id']],
                                weights=[1 - weight, weight], fusion='logit'))
    thresholds = np.round(np.arange(.1, .9001, .025), 6).tolist()
    rules = [dict(id='none', kind='none'), dict(id='largest', kind='largest')]
    rules += [dict(id=f'minvol{v}', kind='volume', value=v) for v in [10, 25, 50, 100, 250, 500, 1000]]
    rules += [dict(id=f'relvol{v}', kind='relative', value=v) for v in [.001, .005, .01, .025, .05]]
    rules += [dict(id=f'seed{v}', kind='seed', value=v) for v in [.6, .7, .8, .9]]
    return dict(at=now(), atoms=atoms, recipes=recipes, thresholds=thresholds,
                stage2_top_recipes=6, stage2_sigmas=[0., .5, 1.],
                stage2_thresholds=np.round(np.arange(.1, .9001, .05), 6).tolist(),
                component_rules=rules, connectivity=26,
                volume_decoder_temperatures=[.5, .75, 1., 1.5, 2.],
                stage1_configurations=len(recipes) * len(thresholds),
                stage2_configurations=6 * (3 * 17 * len(rules) + 5),
                test_access=False, training=False)
