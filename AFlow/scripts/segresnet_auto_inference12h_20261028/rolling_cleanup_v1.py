"""Release only completed, clearly suboptimal new-study renderer payloads."""
import os
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_v2 import guard


def no_readers(target):
    prefix=str(target.resolve())+'/';readers=[]
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():continue
        try:
            if process.stat().st_uid!=os.getuid():continue
            for fd in (process/'fd').iterdir():
                try:
                    if os.readlink(fd).startswith(prefix):readers.append(int(process.name))
                except (FileNotFoundError,PermissionError):pass
            if prefix in (process/'maps').read_text():readers.append(int(process.name))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return sorted(set(readers))


def protected():
    keep={'r1_0000_core_mirror8'}
    for p in (R/'candidates').glob('*/RECIPE.json'):
        recipe=read(p)
        for value in [recipe.get('source_job',{}),recipe.get('aggregation',{})]:
            for key in ['group_source','group','core_source']:
                if value.get(key):keep.add(value[key])
    return keep


def main():
    import fcntl
    lock=(R/'ROLLING_CLEANUP.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    resource=read(R/'RESOURCE_CURRENT.json');token=resource['token']
    write(R/'ROLLING_CLEANUP_PID.json',dict(at=now(),pid=os.getpid(),worker=resource['worker'],resource_token=token))
    while time.time()<read(R/'AUTHORIZATION.json')['deadline_unix']:
        guard();assert read(R/'RESOURCE_CURRENT.json')['token']==token
        incumbent=read(R/'INCUMBENT.json')['scan_dice'];keep=protected()
        analyses=[read(p) for p in (R/'analysis_queue_v2').glob('*.json')]
        calibrations=[read(p) for p in (R/'cpu_queue_v2').glob('*.json')]
        for source in sorted((R/'jobs').glob('*/DONE.json')):
            identifier=source.parent.name
            if not identifier.startswith(('r1_','a1_')) or identifier in keep:continue
            target=TMP/'groups'/identifier;receipt=R/'cleanup_records'/('group_'+identifier)
            if (receipt/'DONE.json').exists() or not target.exists():continue
            assert target.resolve().is_relative_to((TMP/'groups').resolve())
            related=[j for j in analyses if j.get('group_source')==identifier]
            if not related:continue
            scores=[];ready=True
            for analysis in related:
                aid=analysis['id']
                if not (R/'analyses'/aid/'DONE.json').exists():ready=False;break
                consumers=[j for j in calibrations if j.get('source_analysis')==aid]
                if not consumers:ready=False;break
                for consumer in consumers:
                    p=R/'analyses'/consumer['id']/'DONE.json'
                    if not p.exists():ready=False;break
                    scores.append(read(p)['best']['scan_dice'])
                if not ready:break
            if not ready or not scores or max(scores)>=incumbent-.002:continue
            readers=no_readers(target)
            if readers:continue
            done=read(source);files=done['files'];actual=sorted(target.rglob('*.npy'))
            assert len(actual)==len(files) and not list(target.rglob('*.tmp'))
            for entry in files:
                p=TMP/entry['path'];assert p.resolve().is_relative_to(target.resolve()) and not p.is_symlink()
                assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
            receipt.mkdir(parents=True,exist_ok=True)
            write(receipt/'FILES.json',files)
            write(receipt/'PLAN.json',dict(at=now(),identifier=identifier,target=str(target),source_done_sha256=sha(source),source_sha256=sha(Path(__file__)),
                  fully_scored=True,best_calibrated_scan=max(scores),incumbent_scan=incumbent,eligibility='all consumers complete; all registered candidates/core protected; global score gap>0.002',readers=readers,
                  preservation='all results, matrices/cuts, histograms, audited masks, native inputs and checkpoint remain; only this new derived renderer npy payload',directory_skeleton_preserved_for_concurrent_walkers=True))
            for p in actual:p.unlink()
            record=dict(at=now(),state='complete',identifier=identifier,bytes_freed=sum(x['bytes'] for x in files),files=len(files),manifest_sha256=sha(receipt/'FILES.json'),old_cache_or_model_deleted=False)
            write(receipt/'DONE.json',record);write(source.parent/'DERIVED_PAYLOADS_DELETED.json',record)
        freed=sum(read(p).get('bytes_freed',0) for p in (R/'cleanup_records').glob('*/DONE.json'))
        write(R/'ROLLING_CLEANUP_STATUS.json',dict(at=now(),state='monitoring',bytes_freed_total=freed,gib_freed_total=freed/2**30,protected=sorted(keep),global_gap=.002))
        time.sleep(30)
    write(R/'ROLLING_CLEANUP_STATUS.json',dict(at=now(),state='window_closed'))
if __name__=='__main__':main()
