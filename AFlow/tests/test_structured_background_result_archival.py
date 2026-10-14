"""Exercise archival with synthetic reports, including shuffled JSON and tampering."""
import json
from pathlib import Path
import pytest
from scripts import structured_background_protocol as p
from scripts import finalize_structured_background_aux as finalizer
from scripts.evaluate_structured_background_aux import summarize,compare


def synthetic_run(tmp_path,monkeypatch,gain,tamper=False):
    record=tmp_path/'record';work=tmp_path/'work'
    record.mkdir();work.mkdir()
    ids=[]
    for subject in range(78):
        for scan in range(3 if subject<24 else 2):
            ids.append((f'case{subject:03}_{scan}',f'patient{subject:03}'))
    dev=[{'case_id':c,'subject_id':s} for c,s in ids]
    monkeypatch.setattr(p,'ROOT',tmp_path)
    monkeypatch.setattr(p,'RECORD',record)
    monkeypatch.setattr(p,'WORK',work)
    monkeypatch.setattr(p,'check_freeze',lambda:None)
    monkeypatch.setattr(p,'check_roles',lambda:([],dev))
    results={}
    for i,seed in enumerate(p.SEEDS):
        for j,arm in enumerate(p.ARMS):
            key=f'seed{seed}_{arm}'
            intersection=50+(gain if arm=='candidate' else 0)
            rows=[{'case_id':c,'subject_id':s,'dice':intersection/100,'point_hit':True,'intersection':intersection,'predicted_voxels':100,'target_voxels':100} for c,s in ids]
            results[key]=summarize(rows,2026090730+1000+i*2+j)
            d=work/key;d.mkdir();ckpt=d/'fixed.pth';ckpt.write_bytes(b'synthetic hash-only placeholder, never loaded')
            report={'checkpoint':{'path':str(ckpt),'sha256':p.sha(ckpt)},'elapsed_seconds':1,'peak_gpu_memory_bytes':100}
            (d/'TRAIN_DONE.json').write_text(json.dumps(report))
    comps={};gates={}
    for i,seed in enumerate(p.SEEDS):
        c=compare(results[f'seed{seed}_control'],results[f'seed{seed}_candidate'],2026090730+i*100)
        comps[str(seed)]=c
        gates[str(seed)]={'subject_gain':c['subject_delta']>=.003,'scan_positive':c['scan_delta']>0,'point_safety':True,'subject_ci_safety':c['subject_95ci'][0]>=-.003,'scan_cluster_ci_safety':c['scan_cluster_95ci'][0]>=-.003}
    pre=record/'DEV_PREACCESS.json';pre.write_text('{}\n')
    raw={'results':results,'comparisons':comps,'gates':gates,'average_scan_delta':gain/100,'all_gates_passed':gain>0,'forwards':720,'dev_preaccess_sha256':p.sha(pre),'access':{'reused_development180':True,'test270':False,'external':False}}
    if tamper:
        results['seed2026090711_candidate']['per_case'][0]['intersection']+=1
    # Saved JSON alphabetizes candidate before control; seed offsets must not follow that order.
    (record/'RAW_DEVELOPMENT.json').write_text(json.dumps(raw,sort_keys=True))
    return record


@pytest.mark.parametrize('gain,expected',[(1,'completed_reused_development_supported'),(-1,'completed_reused_development_gate_failed')])
def test_archival_handles_sorted_keys_and_positive_negative_gates(tmp_path,monkeypatch,gain,expected):
    record=synthetic_run(tmp_path,monkeypatch,gain)
    finalizer.main()
    result=json.loads((record/'RESULT.json').read_text())
    receipt=json.loads((record/'VERIFICATION_RECEIPT.json').read_text())
    assert result['status']==expected
    assert receipt['status']=='passed'
    assert receipt['result_sha256']==p.sha(record/'RESULT.json')


def test_archival_rejects_tampered_saved_overlap_counts(tmp_path,monkeypatch):
    record=synthetic_run(tmp_path,monkeypatch,1,tamper=True)
    with pytest.raises(AssertionError):
        finalizer.main()
    assert not (record/'RESULT.json').exists()
