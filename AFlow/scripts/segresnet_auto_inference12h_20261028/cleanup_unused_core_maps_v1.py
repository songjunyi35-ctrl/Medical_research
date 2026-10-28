"""Release only three unused core aggregate maps; retain every individual logit."""
import os
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_v2 import guard
from scripts.segresnet_auto_inference12h_20261028.rolling_cleanup_v1 import no_readers


def main():
    guard();identifier='r1_0000_core_mirror8';source=R/'jobs'/identifier/'DONE.json';done=read(source)
    folder=R/'cleanup_records/002_unused_core_aggregates';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'DONE.json').exists():return
    assert (R/'analyses/r1_0000_core_fusions/DONE.json').exists() and (R/'analyses/r1_1000_calibration/DONE.json').exists()
    for queue in ['analysis_queue_v2','analysis_queue_aux_v1']:
        for p in (R/queue).glob('*.json'):
            assert read(p).get('group_source')!=identifier,'aggregate consumer registered'
    names=['mean_probability','mean_logit','std_probability'];prefix='groups/'+identifier+'/'
    files=[x for x in done['files'] if x['path'].startswith(prefix) and x['path'][len(prefix):].split('/')[0] in names]
    assert len(files)==810
    for name in names:
        target=TMP/'groups'/identifier/name;assert not no_readers(target)
    for entry in files:
        p=TMP/entry['path'];assert not p.is_symlink() and p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
    write(folder/'FILES.json',files)
    write(folder/'PLAN.json',dict(at=now(),source_sha256=sha(Path(__file__)),source_done_sha256=sha(source),
          exact_paths=names,preserve='all2160 individual core logits, all original inputs/models, all score/cut tables and masks',
          proof='all active/queued core_source consumers reconstruct from atoms; no registered group_source consumer of core aggregate maps',
          regeneration='frozen render_v1 pointwise sum of sigmoid/square/raw logits in ascending flip0..7;0 neural forwards',directory_skeleton_preserved=True))
    for entry in files:(TMP/entry['path']).unlink()
    record=dict(at=now(),state='complete',bytes_freed=sum(x['bytes'] for x in files),gib_freed=sum(x['bytes'] for x in files)/2**30,files=len(files),all_core_atoms_preserved=True,old_cache_or_model_deleted=False,manifest_sha256=sha(folder/'FILES.json'))
    write(folder/'DONE.json',record);write(source.parent/'UNUSED_AGGREGATES_DELETED.json',record);print(record)
if __name__=='__main__':main()
