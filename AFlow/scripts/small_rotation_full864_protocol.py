"""Conditional full864 refit protocol; no implicit admission from manifest existence."""
import json
from scripts import small_rotation_protocol as source

ROOT=source.ROOT
RECORD=ROOT/'archive/research_records/2026-09-10_small_rotation_full864'
WORK=ROOT/'workspace/BraTS/sammed3d_t1ce_wt/small_rotation_full864_20260910'
CACHE=source.CACHE
NATIVE=source.NATIVE
INDEX_SHA=source.INDEX_SHA
NATIVE_SHA=source.NATIVE_SHA
FIT=RECORD/'FIT864.jsonl'
FIT_SHA='d92619562fb301509f9668873684a3a7f57b0edee1df1fc35e65cea43bdbba52'
DEV=source.DEV
DEV_SHA=source.DEV_SHA  # provenance only: these 180 are now training, never selection.
SEEDS=(2026091071,)
ARMS=('full_control','full_rotation')
sha=source.sha
checked=source.checked
rows=source.rows
write_new=source.write_new


def check_roles():
    fit,dev=source.check_roles();checked(FIT,FIT_SHA)
    full=rows(FIT)
    assert full==sorted(fit+dev,key=lambda r:r['case_id'])
    assert len(full)==864 and len({r['subject_id'] for r in full})==391
    return full,[]


def check_freeze():
    source.check_freeze();check_roles()
    frozen=json.loads((RECORD/'IMPLEMENTATION_FREEZE.json').read_text())
    for name,digest in frozen['files'].items():checked(ROOT/name,digest)
    if frozen['selected_arm'] not in source.ARMS:raise ValueError('no eligible Stage A arm')
    return frozen
