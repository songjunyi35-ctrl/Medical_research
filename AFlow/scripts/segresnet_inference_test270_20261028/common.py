"""One fixed, validation-selected candidate; no training or test parameter search."""
import json
import os
import shutil
import time
from pathlib import Path
from scripts.segresnet_inference_search_20261028.common import ROOT, PY, CHECKPOINT, CHECKPOINT_SHA, now, sha, read, write
from scripts.followup_test270_common import MANIFEST, validate_rows, summary_ci, paired
from scripts.segresnet_native_res_test_common import NativeTestDataset

R = ROOT / 'archive/research_records/2026-10-28_segresnet_inference_selected_test270'
SEARCH = ROOT / 'archive/research_records/2026-10-28_segresnet_inference_search'
OLD = ROOT / 'archive/research_records/2026-10-28_segresnet_w32_e171_test270'
TMP = Path('/tmp/aflow_inference_selected_test270_20261028')
CACHE = TMP / 'native'
CANVAS = (160, 208, 168)
ID = 'segresnet_w32_e171_mirror5_volumeT2_test270_20261028'


def guard():
    assert time.time() < read(R / 'AUTHORIZATION.json')['deadline_unix'], '4-hour evaluation budget reached'
    assert shutil.disk_usage(ROOT).free >= 20 * 2**30, 'persistent reserve below20GiB'
    assert sum(p.stat().st_size for p in R.rglob('*') if p.is_file()) < 2 * 2**30, 'persistent above2GiB'


def verify():
    guard()
    for path, digest in read(R / 'EXECUTION_FREEZE.json')['source_sha256'].items():
        assert sha(ROOT / path) == digest, path
    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    assert sha(R / 'SELECTED_RECIPE.json') == sha(SEARCH / 'DEPLOYMENT_RECIPE.json')


def rows():
    return [json.loads(s) for s in MANIFEST['test'].read_text().splitlines() if s.strip()]


def status(state, **kwargs):
    write(R / 'PIPELINE_STATUS.json', dict(at=now(), state=state, worker=1152682,
          training=False, current_test_retuning=False, **kwargs))


def write_new(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


def env():
    result = os.environ.copy()
    result.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLCONFIGDIR=str(TMP / 'mpl'))
    return result


def predictor():
    # Preserve the entire frozen prediction method; adapt only its shape assertion
    # to this model's historical270 canvas, chosen from existing image geometry.
    from scripts.segresnet_inference_search_20261028 import deploy
    deploy.CANVAS = CANVAS
    model = deploy.SelectedPredictor()
    assert model.selection == read(R / 'SELECTED_RECIPE.json')
    assert model.batch == 8
    return model


def dataset():
    return NativeTestDataset(CACHE, [d['case_id'] for d in rows()])
