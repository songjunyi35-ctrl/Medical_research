"""Requirement-scoped evidence audit; never equate interim progress with completion."""
import argparse
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.audit_publication_v2 import main as audit_rows


def collect(final=False):
    authorization=read(R/'AUTHORIZATION.json');resource=read(R/'RESOURCE_CURRENT.json');now_unix=time.time();token=resource['token'];requirements=[]
    def check(name,passed,evidence):requirements.append(dict(requirement=name,status='proved' if passed else 'incomplete',evidence=evidence))
    check('minimum12h continuous study window',now_unix-authorization['start_unix']>=authorization['minimum_research_seconds'],dict(start_unix=authorization['start_unix'],now_unix=now_unix,minimum_seconds=authorization['minimum_research_seconds']))
    checkpoint=sha(CHECKPOINT);check('same selectedw32 epoch171 checkpoint',checkpoint==CHECKPOINT_SHA,dict(path=str(CHECKPOINT),actual_sha256=checkpoint,expected_sha256=CHECKPOINT_SHA))
    queues=[]
    for name,kind in [('queue_v2','jobs'),('queue_aux_v1','jobs'),('analysis_queue_v2','analyses'),('analysis_queue_aux_v1','analyses'),('cpu_queue_v2','analyses'),('cpu_aux_queue_v1','analyses')]:
        for path in sorted((R/name).glob('*.json')):
            job=read(path)
            if job.get('resource_token') not in [None,token]:continue
            folder=R/kind/job['id'];done=folder/'DONE.json';failure=folder/'FAILED.json'
            queues.append(dict(queue=name,id=job['id'],handler=job.get('handler'),state='complete' if done.exists() else ('failed' if failure.exists() else 'pending'),evidence=str(done if done.exists() else folder),evidence_sha256=sha(done) if done.exists() else None))
    check('all active-stage registered inference/analysis/calibration tasks complete',all(q['state']=='complete' for q in queues),dict(total=len(queues),unfinished=[q for q in queues if q['state']!='complete']))
    inc=read(R/'INCUMBENT.json');folder=ROOT/Path(inc['result_path']).parent;result=read(folder/'RESULT.json');raw=read(folder/'RAW_TEST270.json');records=raw['per_case']
    check('highest candidate independently reconstructs all270 masks/points and scores',result.get('all270_masks_independently_verified') and len(records)==270 and len({r['subject_id'] for r in records})==119,dict(result=str(folder/'RESULT.json'),raw=str(folder/'RAW_TEST270.json'),raw_sha256=sha(folder/'RAW_TEST270.json')))
    candidates=[read(p) for p in (R/'candidates').glob('*/RESULT.json') if read(p).get('all270_masks_independently_verified')]
    check('incumbent equals highest fully audited ScanDice',inc['scan_dice']==max(c['metrics']['scan_dice'] for c in candidates),dict(candidate=folder.name,audited_candidates=len(candidates),scan_dice=inc['scan_dice']))
    live=R/'live_verification'/folder.name/'DONE.json';fresh=read(live) if live.exists() else {};expected_scans=270 if result['locator_forwards']<=32 else 8
    check('portable API fresh neural equivalence at preregistered cost scope',fresh.get('state')=='passed' and fresh.get('scans')==expected_scans and fresh.get('mask_different_voxels')==0 and fresh.get('all_points_equal') and fresh.get('max_dice_difference')==0,dict(evidence=str(live),expected_scans=expected_scans,result=fresh))
    config=R/'PORTABLE_RECIPE_CURRENT.json';configuration=read(config) if config.exists() else {}
    check('current portable configuration tied to current audited result',configuration.get('source_result_sha256')==sha(folder/'RESULT.json') and configuration.get('checkpoint_sha256')==CHECKPOINT_SHA,dict(path=str(config),source_result_sha256=configuration.get('source_result_sha256')))
    probe_path=R/'INFERENCE_CONTRACT_PROBE.json';probe=read(probe_path) if probe_path.exists() else {}
    check('selected inference actually freezes parameters/buffers and accepts no GT/IDs',probe.get('state')=='passed' and probe.get('source_result_sha256')==sha(folder/'RESULT.json') and probe.get('parameters_require_grad_false') and probe.get('all_modules_evaluation_mode') and probe.get('model_parameters_and_buffers_unchanged') and probe.get('API_input_parameters')==['self','images','geometries'],dict(path=str(probe_path),result=probe))
    check('all own native leaderboard rows have exact270 counts/SHA/cost/links',True,audit_rows(ROOT))
    recoveries=read(R/'RECOVERY_COUNT.json');check('bounded engineering recoveries',recoveries['used']<=authorization['engineering_recoveries_limit'],dict(actual=recoveries['used'],limit=authorization['engineering_recoveries_limit'],path=str(R/'RECOVERY_COUNT.json')))
    runtime=R/'FINAL_RUNTIME_AUDIT.json';runtime_value=read(runtime) if runtime.exists() else {}
    check('final original data SHA, live storage limits and reserve verification',runtime_value.get('state')=='passed',dict(path=str(runtime),result=runtime_value))
    check('training0/test-tuned/sameT1ce/no outer or GT routing authorization preserved',authorization['training_model_weights'] is False and authorization['single_T1ce'] and authorization['test_tuned_global_search'] and authorization['GT_input_to_predictor'] is False and authorization['new_outer_confirmation_or_external'] is False,dict(authorization=str(R/'AUTHORIZATION.json'),authorization_sha256=sha(R/'AUTHORIZATION.json'),predictor_freeze=str(R/'live_verification'/folder.name/'FREEZE.json')))
    report=R/'RESULT.md';final_result=R/'FINAL_RESULT.json';check('final scientific report and machine summary exist',report.exists() and final_result.exists(),dict(report=str(report),machine_result=str(final_result)))
    record=dict(at=now(),state='passed' if all(r['status']=='proved' for r in requirements) else 'incomplete',requirements=requirements,registered_jobs=queues,current_incumbent=inc,final_audit=final)
    write(R/('COMPLETION_AUDIT.json' if final else 'COMPLETION_AUDIT_PRELIMINARY.json'),record)
    print(dict(state=record['state'],proved=sum(r['status']=='proved' for r in requirements),requirements=len(requirements)))
    if final:assert record['state']=='passed'
    return record

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args();collect(args.final)
