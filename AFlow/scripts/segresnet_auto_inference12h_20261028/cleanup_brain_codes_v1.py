"""Release four derived code fields only after both brain beams finish."""
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_v2 import guard
from scripts.segresnet_auto_inference12h_20261028.rolling_cleanup_v1 import no_readers

def main():
    guard();source=R/'analyses/f2_0000_brain_component_beam';target=TMP/'analyses/f2_0000_brain_component_beam/codes'
    for identifier in ['f2_0000_brain_component_beam','f3_0000_brain_decoder_component_beam']:
        assert (R/'analyses'/identifier/'DONE.json').exists()
    assert not no_readers(target)
    files=read(source/'CODE_FILES.json');assert len(files)==1080
    actual=sorted(target.rglob('*.npy'));assert len(actual)==len(files)
    entries=[]
    for entry in files:
        p=Path(entry['path']);assert p.resolve().is_relative_to(target.resolve()) and not p.is_symlink()
        assert sha(p)==entry['sha256'];entries.append(dict(path=str(p),bytes=p.stat().st_size,sha256=entry['sha256']))
    folder=R/'cleanup_records/brain_component_codes';folder.mkdir(parents=True,exist_ok=True);assert not (folder/'DONE.json').exists()
    write(folder/'FILES.json',entries);write(folder/'PLAN.json',dict(at=now(),target=str(target),source_sha256=sha(Path(__file__)),source_code_manifest_sha256=sha(source/'CODE_FILES.json'),all_consumers_done=True,readers=[],preservation='native inputs, weights, core8 individual logits, every score/cut matrix, all audited masks',regeneration='same4 bases from retained core8 logits, zero CNN',directory_skeleton_preserved=True))
    for p in actual:p.unlink()
    record=dict(at=now(),state='complete',bytes_freed=sum(x['bytes'] for x in entries),files=1080,old_cache_or_model_deleted=False,manifest_sha256=sha(folder/'FILES.json'))
    write(folder/'DONE.json',record);write(source/'DERIVED_CODES_DELETED.json',record);print(record)
if __name__=='__main__':main()
