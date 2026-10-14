"""CPU-capacity amendment: isolated 24-process rebuild after V1 partial stop."""
import datetime
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_native_cache_multiscale_20260924 as base

ROOT_R = ROOT / 'archive/research_records/2026-09-24_segresnet_native_multiscale_supervision'
REPAIR = ROOT_R / 'REPAIR_V2'
base.R = REPAIR
base.CACHE = Path('/tmp/aflow_multiscale_native_20260924_v2')
base.builder.CACHE_ROOT = base.CACHE
original_build = base.builder.build


def build_24(split, processes):
    if processes != 12:
        raise RuntimeError('unexpected V1 builder request')
    return original_build(split, processes=24)


base.builder.build = build_24


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    base.write(ROOT_R / 'PIPELINE_STATUS.json', {'utc': now(), 'state': 'building_native_cache_v2',
          'split': 'train', 'running': True, 'worker': 1109780,
          'processes': 24, 'training_started': False, 'test_pixels_access': False})
    try:
        base.main()
        shutil.copy2(REPAIR / 'DATA_AUDIT.json', ROOT_R / 'DATA_AUDIT.json')
        base.write(ROOT_R / 'PIPELINE_STATUS.json', {'utc': now(), 'state': 'cache_complete',
              'running': False, 'worker': 1109780,
              'processes': 24, 'data_audit_sha256': base.sha(ROOT_R / 'DATA_AUDIT.json'),
              'training_started': False, 'test_pixels_access': False})
    except BaseException as exc:
        base.write(ROOT_R / 'CACHE_V2_FAILURE.json', {'utc': now(), 'error': repr(exc),
              'patient_training_started': False, 'test_pixels_access': False})
        base.write(ROOT_R / 'PIPELINE_STATUS.json', {'utc': now(), 'state': 'cache_v2_failed',
              'running': False, 'worker': 1109780, 'training_started': False,
              'test_pixels_access': False})
        raise


if __name__ == '__main__':
    main()
