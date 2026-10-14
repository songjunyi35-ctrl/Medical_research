"""Independent conditional target-representation study, fixed fit/dev roles."""
import json
from pathlib import Path
from scripts import structured_background_protocol as source

ROOT=source.ROOT
RECORD=ROOT/'archive/research_records/2026-09-10_fractional_occupancy'
WORK=ROOT/'workspace/BraTS/sammed3d_t1ce_wt/fractional_occupancy_20260910'
SOFT_CACHE=Path('/tmp/aflow-fractional-occupancy-20260910')
CACHE=source.CACHE
NATIVE=source.NATIVE
FIT=source.FIT
DEV=source.DEV
FIT_SHA=source.FIT_SHA
DEV_SHA=source.DEV_SHA
INDEX_SHA=source.INDEX_SHA
NATIVE_SHA=source.NATIVE_SHA
SEEDS=source.SEEDS
ARMS=('fractional_occupancy',)
sha=source.sha
checked=source.checked
rows=source.rows
write_new=source.write_new
check_roles=source.check_roles


def check_freeze():
    source.check_freeze();check_roles()
    frozen=json.loads((RECORD/'IMPLEMENTATION_FREEZE.json').read_text())
    for name,digest in frozen['files'].items():checked(ROOT/name,digest)
    return frozen
