"""Independent postprocessing study of already-revealed fixed native controls."""
import json
from scripts import structured_background_protocol as source

ROOT=source.ROOT
RECORD=ROOT/'archive/research_records/2026-09-10_peak_protected_cleanup'
CACHE=source.CACHE
FIT=source.FIT
DEV=source.DEV
FIT_SHA=source.FIT_SHA
DEV_SHA=source.DEV_SHA
INDEX_SHA=source.INDEX_SHA
SEEDS=source.SEEDS
sha=source.sha
checked=source.checked
rows=source.rows
write_new=source.write_new
check_roles=source.check_roles


def source_controls():
    source.check_freeze();check_roles()
    result={}
    expected={2026090711:'fdad97d9abc736638bcd9911fc82ddca1a434fd0c9a46c3e9306d608276b3d6e',2026090712:'5b014fb256c272b828e01322b7985f10d8ac5c53f19fa475761c25537f697e43'}
    for seed in SEEDS:
        path=source.WORK/f'seed{seed}_control/TRAIN_DONE.json';d=json.loads(path.read_text())
        assert d['checkpoint']['sha256']==expected[seed] and d['fit_sha256']==FIT_SHA and not d['teacher_checkpoints']
        checked(d['checkpoint']['path'],expected[seed])
        result[str(seed)]={'train_report_path':str(path),'train_report_sha256':sha(path),'checkpoint':d['checkpoint']}
    return result
