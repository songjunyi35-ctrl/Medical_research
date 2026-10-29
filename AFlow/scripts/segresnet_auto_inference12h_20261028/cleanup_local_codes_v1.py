"""Release completed local-grid code fields; retain original logits and all scores."""
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_v2 import guard
from scripts.segresnet_auto_inference12h_20261028.rolling_cleanup_v1 import no_readers


def main():
    guard();source=R/'analyses/h1_0000_local_brain_joint';target=TMP/'analyses/h1_0000_local_brain_joint/codes'
    assert (source/'DONE.json').exists() and (R/'candidates/0036_local_brain_joint/RESULT.json').exists() and (R/'live_verification/0036_local_brain_joint/DONE.json').exists()
    assert not no_readers(target);files=read(source/'CODE_FILES.json');assert len(files)==270
    actual=sorted(target.glob('*.npy'));assert len(actual)==270;entries=[]
    for entry in files:
        p=Path(entry['path']);assert p.resolve().is_relative_to(target.resolve()) and not p.is_symlink();assert sha(p)==entry['sha256'];entries.append(dict(path=str(p),bytes=p.stat().st_size,sha256=entry['sha256']))
    folder=R/'cleanup_records/h1_local_codes';folder.mkdir(parents=True,exist_ok=True);assert not (folder/'DONE.json').exists()
    write(folder/'FILES.json',entries);write(folder/'PLAN.json',dict(at=now(),source_sha256=sha(Path(__file__)),source_manifest_sha256=sha(source/'CODE_FILES.json'),target=str(target),all_consumers_done=True,readers=[],preserve='all nativeinputs/checkpoint/core8individual logits/allscore-cut matrices/auditedmasks',regeneration='same fixedcase24 aggregation from originalcore8 logits,zeroCNN',keep_directory_skeleton=True))
    for p in actual:p.unlink()
    record=dict(at=now(),state='complete',bytes_freed=sum(e['bytes'] for e in entries),files=270,old_cache_or_model_deleted=False,manifest_sha256=sha(folder/'FILES.json'))
    write(folder/'DONE.json',record);write(source/'DERIVED_CODES_DELETED.json',record);print(record)

if __name__=='__main__':main()
