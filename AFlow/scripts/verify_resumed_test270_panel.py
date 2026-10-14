"""Independent saved-mask/GT/count/cluster-CI audit, with zero model forwards."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import numpy as np
from sammed3d.packed_prediction_masks import read_packed_case,file_sha
from scripts.structured_background_protocol import ROOT,checked,write_new,rows
from scripts.finalize_structured_background_aux import independent_ci
from scripts import weak_elastic_test270_helpers as constants


def verify_case(row,prediction,target):
    if prediction.shape!=target.shape or prediction.dtype!=np.bool_ or target.dtype!=np.bool_:raise ValueError('same-grid boolean masks required')
    target_count=int(target.sum());pred_count=int(prediction.sum());intersection=int((prediction&target).sum())
    if target_count<=0:raise ValueError('frozen test GT must be nonempty')
    assert (row['intersection'],row['predicted_voxels'],row['target_voxels'])==(intersection,pred_count,target_count)
    exact=2*intersection/(pred_count+target_count)
    assert abs(row['exact_count_dice']-exact)<1e-12
    # Metrics deliberately use the historical GPU float32 arithmetic, not new float64 values.
    assert abs(row['dice']-exact)<=1e-7
    point=row['point_flat_index'];assert isinstance(point,int) and 0<=point<target.size
    assert row['point_hit']==bool(target.reshape(-1)[point])


def verify_summary(summary,seed):
    grouped=defaultdict(list)
    for row in summary['per_case']:grouped[row['subject_id']].append(row['dice'])
    assert len(summary['per_case'])==summary['scans']==270 and len(grouped)==summary['subjects']==119
    assert abs(summary['scan_dice']-statistics.fmean(r['dice'] for r in summary['per_case']))<1e-12
    assert abs(summary['subject_dice']-statistics.fmean(statistics.fmean(v) for v in grouped.values()))<1e-12
    assert summary['point_hits']==sum(r['point_hit'] for r in summary['per_case'])
    ci=independent_ci(summary['per_case'],seed)
    assert all(np.allclose(v,summary[k],rtol=0,atol=1e-12) for k,v in ci.items())


def verify_contrast(saved,a,b,seed):
    delta=[]
    for left,right in zip(a['per_case'],b['per_case'],strict=True):
        assert (left['case_id'],left['subject_id'])==(right['case_id'],right['subject_id'])
        delta.append({**left,'dice':right['dice']-left['dice']})
    assert abs(saved['scan_delta']-(b['scan_dice']-a['scan_dice']))<1e-12
    assert abs(saved['subject_delta']-(b['subject_dice']-a['subject_dice']))<1e-12
    assert saved['point_delta']==b['point_hits']-a['point_hits']
    ci=independent_ci(delta,seed)
    assert all(np.allclose(v,saved[k],rtol=0,atol=1e-12) for k,v in ci.items())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',choices=['small_rotation_full864','image_blur'],required=True);args=parser.parse_args()
    record=ROOT/f'archive/research_records/2026-09-10_{args.study}';output=record/'TEST_VERIFICATION.json'
    if output.exists():raise FileExistsError(output)
    raw_path=record/'RAW_TEST270.json';raw=json.loads(raw_path.read_text());assert raw['status']=='completed'
    pre=record/'TEST_PREACCESS.json';checked(pre,raw['test_preaccess_sha256']);receipt=json.loads(pre.read_text())
    for name,digest in receipt['files'].items():checked(ROOT/name,digest)
    assert raw['new_forwards']==receipt['inference']['expected_forwards']==4320
    assert receipt['test']['scans']==270 and receipt['test']['subjects']==119
    checked(constants.TEST,constants.TEST_SHA);checked(constants.CACHE/'index.json',constants.INDEX_SHA);checked(constants.CACHE/'labels.u8',constants.LABELS_SHA)
    expected=rows(constants.TEST);ids=[r['case_id'] for r in expected]
    assert json.loads((constants.CACHE/'index.json').read_text())['case_ids']==ids
    mask_path=Path(raw['prediction_masks']['index_path']);assert mask_path==record/'TEST_MASK_INDEX.json'
    checked(mask_path,raw['prediction_masks']['index_sha256']);mask_index=json.loads(mask_path.read_text())
    assert mask_index['case_ids']==ids and mask_index['shape']==[128,128,128] and mask_index['cases']==270
    assert set(mask_index['files'])==set(raw['results']) and len(raw['results'])==4
    for name,info in mask_index['files'].items():
        path=Path(info['path']);assert path==record/'test_prediction_masks'/f'{name}.bits'
        checked(path,info['sha256']);assert info['cases']==270 and info['bytes']==270*128**3//8
    labels=np.memmap(constants.CACHE/'labels.u8',mode='r',dtype=np.uint8,shape=(270,128,128,128))
    seed=receipt['bootstrap']['seed'];assert receipt['bootstrap']['samples']==10000
    checks={}
    for name,summary in raw['results'].items():
        assert [r['case_id'] for r in summary['per_case']]==ids
        for i,row in enumerate(summary['per_case']):
            assert row['subject_id']=='-'.join(row['case_id'].split('-')[:3])
            target=np.array(labels[i],dtype=bool);prediction=read_packed_case(mask_index,name,i)
            verify_case(row,prediction,target)
        verify_summary(summary,seed)
        checks[name]={'scans':270,'mask_overlap_counts_recomputed':True,'saved_point_gt_hit_verified':True,'raw_logit_argmax_not_independently_recomputed':True,'historical_float32_dice_tolerance':1e-7,'absolute_patient_cluster_ci_recomputed':True}
    checked(constants.LEADER,constants.LEADER_SHA)
    leader=constants.leader_rows(json.loads(constants.LEADER.read_text()))
    all_summaries={**raw['results'],'old_overall':leader}
    assert set(raw['comparison_pairs'])==set(raw['comparisons'])
    for name,(left,right) in raw['comparison_pairs'].items():verify_contrast(raw['comparisons'][name],all_summaries[left],all_summaries[right],seed)
    role=raw['gate_role_outputs'];candidate=raw['results'][role['candidate_flip8']];control=raw['results'][role['control_flip8']];contrast=raw['comparisons']['candidate_flip8_vs_old_overall']
    gates={'candidate_overall_scan_lead':candidate['scan_dice']>constants.OVERALL_SCAN,
           'candidate_identity_1x_scan_lead':raw['results'][role['candidate_identity']]['scan_dice']>constants.SINGLE_SCAN,
           'candidate_flip8_above_matched_control':candidate['scan_dice']>control['scan_dice'],
           'candidate_subject_above_matched_control':candidate['subject_dice']>control['subject_dice'],
           'subject_above_old_overall':candidate['subject_dice']>constants.OVERALL_SUBJECT,
           'point_at_least265':candidate['point_hits']>=265,
           'point_at_least_matched_control':candidate['point_hits']>=control['point_hits'],
           'scan_cluster_ci_positive':contrast['scan_cluster_95ci'][0]>0,
           'subject_ci_positive':contrast['subject_95ci'][0]>0}
    assert gates==raw['gates']
    assert raw['candidate_overall_scientific_gate']==all(v for k,v in gates.items() if k!='candidate_identity_1x_scan_lead')
    inputs={}
    for arm,values in receipt['reports'].items():
        inputs[arm]={}
        for kind,value in values.items():
            if isinstance(value,dict) and 'path' in value and 'sha256' in value:
                checked(value['path'],value['sha256']);inputs[arm][kind]=value
    write_new(output,{'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),'raw_sha256':file_sha(raw_path),'test_preaccess_sha256':file_sha(pre),'mask_index_sha256':file_sha(mask_path),'mask_files':mask_index['files'],'checks':checks,'comparisons_recomputed':list(raw['comparison_pairs']),'gates_recomputed':gates,'checkpoint_report_hashes':inputs,'test_manifest_sha256':constants.TEST_SHA,'cache_index_sha256':constants.INDEX_SHA,'labels_sha256':constants.LABELS_SHA,'new_model_forwards':0,'access':'same_reused_test_gt_for_saved_prediction_verification_only','verifier_sha256':file_sha(Path(__file__))})
    print(json.dumps({'status':'passed','actual_masks_verified':1080,'gates':gates,'raw_sha256':file_sha(raw_path)},indent=2),flush=True)


if __name__=='__main__':main()
