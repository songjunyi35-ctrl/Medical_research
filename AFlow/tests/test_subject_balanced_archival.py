"""Synthetic archival contracts for reused controls and new sampler candidates."""
import json
from pathlib import Path
import pytest
from scripts import subject_balanced_protocol as p
from scripts import finalize_subject_balanced_sampling as finalizer
from scripts.evaluate_structured_background_aux import summarize,compare


def fixture(tmp_path,monkeypatch,gain,tamper_control=False):
    record=tmp_path/'record';source=tmp_path/'source';work=tmp_path/'work'
    for d in (record,source,work):d.mkdir()
    for key,value in [('ROOT',tmp_path),('RECORD',record),('SOURCE_RECORD',source),('WORK',work)]:monkeypatch.setattr(p,key,value)
    monkeypatch.setattr(p,'check_freeze',lambda:None)
    ids=[(f'c{s}_{i}',f'p{s}') for s in range(78) for i in range(3 if s<24 else 2)]
    raw_source={'results':{}};results={};comps={};gates={}
    for i,seed in enumerate(p.SEEDS):
        entries=[]
        for arm,inter in [('control',50),('candidate',50+gain)]:
            rows=[{'case_id':c,'subject_id':s,'intersection':inter,'predicted_voxels':100,'target_voxels':100,'dice':inter/100,'point_hit':True} for c,s in ids]
            entries.append(summarize(rows,2026091830+i if arm=='candidate' else 1+i))
        control,candidate=entries;raw_source['results'][f'seed{seed}_control']=control
        results[str(seed)]={'reused_control':json.loads(json.dumps(control)),'candidate':candidate}
        c=compare(control,candidate,2026090830+i*100);comps[str(seed)]=c
        gates[str(seed)]={'subject_gain':c['subject_delta']>=.003,'scan_positive':c['scan_delta']>0,'point_safety':True,'subject_ci_safety':c['subject_95ci'][0]>=-.003,'scan_cluster_ci_safety':c['scan_cluster_95ci'][0]>=-.003}
        d=work/f'seed{seed}_subject_balanced';d.mkdir();ck=d/'fixed.pth';ck.write_bytes(b'no model, hash-only fixture')
        (d/'TRAIN_DONE.json').write_text(json.dumps({'checkpoint':{'path':str(ck),'sha256':p.sha(ck)},'elapsed_seconds':1,'peak_gpu_memory_bytes':1}))
    (source/'RAW_DEVELOPMENT.json').write_text(json.dumps(raw_source))
    pre=record/'DEV_PREACCESS.json';pre.write_text('{}')
    if tamper_control:results[str(p.SEEDS[0])]['reused_control']['per_case'][0]['dice']=.9
    raw={'results':results,'comparisons':comps,'gates':gates,'average_scan_delta':gain/100,'all_gates_passed':gain>0,'new_forwards':360,'source_raw_sha256':p.sha(source/'RAW_DEVELOPMENT.json'),'dev_preaccess_sha256':p.sha(pre),'access':{'reused_development180':True,'test270':False,'external':False,'new_control_inference':False}}
    (record/'RAW_DEVELOPMENT.json').write_text(json.dumps(raw,sort_keys=True))
    return record


@pytest.mark.parametrize('gain',[1,-1])
def test_supported_and_failed_result_archival(tmp_path,monkeypatch,gain):
    record=fixture(tmp_path,monkeypatch,gain)
    finalizer.main()
    result=json.loads((record/'RESULT.json').read_text())
    assert result['all_gates_passed']==(gain>0)
    assert json.loads((record/'VERIFICATION_RECEIPT.json').read_text())['status']=='passed'


def test_reused_control_cannot_be_silently_changed(tmp_path,monkeypatch):
    record=fixture(tmp_path,monkeypatch,1,tamper_control=True)
    with pytest.raises(AssertionError):finalizer.main()
    assert not (record/'RESULT.json').exists()
