"""Metadata-only normalization of audited historical270 atoms. No array access."""
from scripts.race_v2_inference_a_20261029.common import *
def main():
    audit=read(R/'EXISTING270_CACHE_AUDIT.json')
    assert audit['all2160_exact'] and audit['new270_scoring'] is False
    ready=read(OLD/'CACHE_READY.json');atoms=[];sources={}
    for rank in (0,1):
        p=OLD/f'FORWARD_COMPLETE_{rank}.json';sources[str(p)]=sha(p)
        atoms.extend(read(p)['atoms'])
    atoms.sort(key=lambda a:(a['slot'],a['view']))
    assert len(atoms)==2160 and {(a['slot'],a['view']) for a in atoms}=={(s,v) for s in range(270) for v in range(8)}
    manifest=ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/sealed_test/test_cases_t1ce.jsonl'
    path=R/'final270/reuse_race164/ATOMS_READY.json';assert not path.exists()
    write(path,dict(at=now(),status='audited_historical_reuse',model_id='race164',run_id='race164_e164_test270_20261029_historical',checkpoint_path=str(CHECKPOINT),checkpoint_sha256=CHECKPOINT_SHA,architecture='sammed3d.race_seg.RaceSeg',construction=dict(width=32,dropout=.1,dim=16,enabled=True),manifest_sha256=sha(manifest),index_sha256=ready['index_sha256'],cache_path='/tmp/aflow_race_e164_test270_20261029/native',canvas=[160,208,168],case_forwards=2160,historical_case_forwards=2160,new_model_forwards=0,atoms=atoms,source_rank_manifests_sha256=sources,existing_integrity_audit_path=str(R/'EXISTING270_CACHE_AUDIT.json'),existing_integrity_audit_sha256=sha(R/'EXISTING270_CACHE_AUDIT.json'),no_array_loaded_by_this_entry=True,new270_scoring=False))
    print(str(path))
if __name__=='__main__':main()
