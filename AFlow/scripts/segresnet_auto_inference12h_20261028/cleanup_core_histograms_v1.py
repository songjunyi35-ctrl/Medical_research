"""Authorized rolling cleanup of completed new-study derived histograms only."""
import os,shutil
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_v2 import guard


def main():
    guard();identifier='r1_0000_core_fusions';target=TMP/'analyses'/identifier/'histograms'
    expected=(TMP/'analyses'/'r1_0000_core_fusions'/'histograms').resolve()
    assert target.resolve()==expected and expected.is_relative_to(TMP.resolve())
    for name in ['r1_1000_calibration','r1_1000a_fine_calibration','r1_1002_adaptive_calibration']:
        assert (R/'analyses'/name/'DONE.json').is_file(),name
    for queue in ['cpu_queue_v2','analysis_queue_v2']:
        for path in (R/queue).glob('*.json'):
            job=read(path)
            if job.get('source_analysis')!=identifier:continue
            assert (R/'analyses'/job['id']/'DONE.json').is_file(),job['id']
    # A completed dependency cannot launch a future reader because all registered
    # jobs referencing this histogram source are already complete.
    prefix=str(expected)+'/'
    readers=[]
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():continue
        try:
            if process.stat().st_uid!=os.getuid():continue
            for fd in (process/'fd').iterdir():
                try:
                    if os.readlink(fd).startswith(prefix):readers.append(dict(pid=int(process.name),kind='fd'))
                except (FileNotFoundError,PermissionError):pass
            if prefix in (process/'maps').read_text():readers.append(dict(pid=int(process.name),kind='mmap'))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    assert not readers,readers
    folder=R/'cleanup_records/001_original_core_histograms';folder.mkdir(parents=True,exist_ok=True)
    if not target.exists():assert (folder/'DONE.json').exists();return
    files=[dict(path=str(p.relative_to(TMP)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(target.rglob('*')) if p.is_file()]
    assert len(files)==28*270 and all(x['path'].endswith('.npy') for x in files)
    write(folder/'FILES.json',files)
    write(folder/'PLAN.json',dict(at=now(),target=str(expected),source_sha256=sha(Path(__file__)),files=len(files),bytes=sum(x['bytes'] for x in files),readers=readers,
          authorization='AUTHORIZATION rolling cleanup of only new reproducible intermediates after score/SHA archival and no readers',
          preserved=['all score/cut matrices','all ranking/configuration tables','all audited masks','core8 logits','native images/labels/index','all old caches/models/reports'],
          regeneration='core8 saved logits + original core_fusions_v3 probability function + frozen selected recipe indices + native labels;0 new model forwards'))
    shutil.rmtree(target)
    record=dict(at=now(),state='complete',bytes_freed=sum(x['bytes'] for x in files),gib_freed=sum(x['bytes'] for x in files)/2**30,files=len(files),manifest_sha256=sha(folder/'FILES.json'),old_cache_or_model_deleted=False)
    write(folder/'DONE.json',record);write(R/'analyses'/identifier/'DERIVED_HISTOGRAMS_DELETED.json',record);print(record)
if __name__=='__main__':main()
