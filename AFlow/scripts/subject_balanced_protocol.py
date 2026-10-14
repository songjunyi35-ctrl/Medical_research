"""New experiment paths, reusing the explicitly frozen fit/development roles."""
import json
from scripts import structured_background_protocol as source

ROOT=source.ROOT
RECORD=ROOT/'archive/research_records/2026-09-07_subject_balanced_sampling'
WORK=ROOT/'workspace/BraTS/sammed3d_t1ce_wt/subject_balanced_sampling_20260907'
SOURCE_RECORD=source.RECORD
SOURCE_WORK=source.WORK
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
ARMS=('subject_balanced',)
sha=source.sha
checked=source.checked
rows=source.rows
write_new=source.write_new
check_roles=source.check_roles


def check_freeze():
    source.check_freeze()
    frozen=json.loads((RECORD/'IMPLEMENTATION_FREEZE.json').read_text())
    for name,digest in frozen['files'].items():checked(ROOT/name,digest)
    return frozen
