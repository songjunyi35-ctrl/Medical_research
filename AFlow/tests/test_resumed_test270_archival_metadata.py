"""Synthetic-only end-to-end provenance rendering; no real test cohort access."""
import json
import pytest
from scripts import archive_resumed_test270_panel as archive


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize('study',['small_rotation_full864','image_blur'])
def test_full_and_fit_roles_survive_archive(tmp_path,monkeypatch,study):
    monkeypatch.setattr(archive,'ROOT',tmp_path)
    record=tmp_path/f'archive/research_records/2026-09-10_{study}'
    destination=tmp_path/f'archive/research_records/2026-09-10_{study}_test270'
    full=study=='small_rotation_full864';arms=['full_control','full_rotation'] if full else ['control','candidate']
    reports={}
    for arm in arms:
        checkpoint=tmp_path/f'work/{arm}.pth';checkpoint.parent.mkdir(parents=True,exist_ok=True);checkpoint.write_bytes(arm.encode())
        train=tmp_path/f'work/{arm}.json';dump(train,{'elapsed_seconds':3600})
        reports[arm]={'checkpoint':{'path':str(checkpoint),'sha256':archive.sha(checkpoint)},'train_report':{'path':str(train),'sha256':archive.sha(train)}}
    pre={'reports':reports,'bootstrap':{'seed':2026091032}}
    if full:
        dump(record/'TRAIN_PREACCESS.json',{'selected_sampling_policy':'subject_rotation'})
        pre['training_preaccess_sha256']=archive.sha(record/'TRAIN_PREACCESS.json')
    else:dump(record/'RESULT.json',{'checkpoints':{'seed11':{'seconds':3600},'seed12':{'seconds':3600}}})
    dump(record/'TEST_PREACCESS.json',pre)
    metrics={f'{arm}_{mode}':{'scan_dice':.77,'subject_dice':.75,'point_hits':265,'scan_cluster_95ci':[.7,.8],'subject_95ci':[.68,.81],'per_case':[]} for arm in arms for mode in ['identity','flip8']}
    raw={'results':metrics,'new_forwards':4320,'test_preaccess_sha256':archive.sha(record/'TEST_PREACCESS.json'),'test_tuning_scope':'synthetic fixture only','comparisons':{},'gates':{'candidate_overall_scan_lead':True,'candidate_identity_1x_scan_lead':True},'candidate_overall_scientific_gate':True,'seconds':40,'peak_gpu_allocated_bytes':5*1024**3,'prediction_masks':{'bytes':4*270*128**3//8},'access':{'synthetic_only':True}}
    dump(record/'RAW_TEST270.json',raw);dump(record/'TEST_VERIFICATION.json',{'status':'passed','raw_sha256':archive.sha(record/'RAW_TEST270.json')});dump(record/'TEST_MASK_INDEX.json',{'synthetic_fixture':True})
    result=archive.archive(study,record,destination)
    rows=result['leaderboard_rows'];assert len(rows)==4
    assert {r['locator_forwards'] for r in rows}=={1,8}
    assert all(r['training_scope']['fit_scans']==(864 if full else 684) for r in rows)
    assert all(r['training_scope']['fit_subjects']==(391 if full else 313) for r in rows)
    assert all(r['training_scope']['seed']==(2026091071 if full else 2026090711) for r in rows)
    assert all(r['training_scope']['former_development180_consumed_as_training']==full for r in rows)
    assert all(r['training_scope']['sampling_policy']=='subject_equal_within_native' for r in rows)
    assert result['new_training_stage_gpu_hours']==2
    if not full:assert json.loads((record/'RESULT.json').read_text())=={'checkpoints':{'seed11':{'seconds':3600},'seed12':{'seconds':3600}}}
    assert not (tmp_path/'AFLOW_BRATS_LEADERBOARD.json').exists(),'archive stage must not implicitly mutate canonical board'
