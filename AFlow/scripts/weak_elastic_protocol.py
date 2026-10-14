"""Sept10 prospective two fixed sampling strata, one new augmentation factor."""
import json
from scripts import structured_background_protocol as source

ROOT=source.ROOT
RECORD=ROOT/'archive/research_records/2026-09-10_weak_elastic'
WORK=ROOT/'workspace/BraTS/sammed3d_t1ce_wt/weak_elastic_20260910'
ROLES=source.ROLES
CACHE=source.CACHE
NATIVE=source.NATIVE
FIT=source.FIT
DEV=source.DEV
FIT_SHA=source.FIT_SHA
DEV_SHA=source.DEV_SHA
INDEX_SHA=source.INDEX_SHA
NATIVE_SHA=source.NATIVE_SHA
SEEDS=source.SEEDS
ARMS=('native_elastic','subject_elastic')
sha=source.sha
checked=source.checked
rows=source.rows
write_new=source.write_new
check_roles=source.check_roles


def control_path(seed, arm):
    if arm == 'native_elastic':
        return source.WORK/f'seed{seed}_control/TRAIN_DONE.json'
    if arm == 'subject_elastic':
        return ROOT/f'workspace/BraTS/sammed3d_t1ce_wt/subject_balanced_sampling_20260907/seed{seed}_subject_balanced/TRAIN_DONE.json'
    raise ValueError(arm)


def check_freeze():
    source.check_freeze()
    frozen=json.loads((RECORD/'IMPLEMENTATION_FREEZE.json').read_text())
    for name,digest in frozen['files'].items():checked(ROOT/name,digest)
    return frozen
