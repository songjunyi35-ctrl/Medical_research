"""Independent full864 index/RNG replay before any conditional test admission."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import torch
from scripts import small_rotation_full864_protocol as p
from scripts.audit_subject_sampler_replay import independent_weights,replay


def rotation_stream(seed,steps):
    gen=torch.Generator().manual_seed(seed+200000);digest=hashlib.sha256();selected=0
    for _ in range(steps):
        flags=torch.rand(8,generator=gen)<.2;angles=(torch.rand(8,3,generator=gen)*2-1)*15.
        digest.update(flags.numpy().tobytes());digest.update(angles.numpy().tobytes());selected+=int(flags.sum())
    return digest.hexdigest(),selected


def main():
    frozen=p.check_freeze();fit,_=p.check_roles();torch.set_num_threads(1)
    output=p.RECORD/'TRAINING_REPLAY_FINAL.json'
    if output.exists():raise FileExistsError(output)
    assert json.loads((p.RECORD/'TRAINING_DONE.json').read_text())['status']=='both_full864_fixed168_complete'
    cases=[r['case_id'] for r in fit];subjects={r['case_id']:r['subject_id'] for r in fit};volumes=json.loads(p.NATIVE.read_text())['native_wt_voxels']
    policies=independent_weights(cases,subjects,volumes);weights=policies['subject_balanced' if frozen['selected_arm']=='subject_rotation' else 'control'];seed=p.SEEDS[0]
    expected=replay(weights,seed,168,cases,subjects,volumes);weight_sha=hashlib.sha256(weights.numpy().tobytes()).hexdigest()
    reports={};checks={};hashes={}
    for arm in p.ARMS:
        path=p.WORK/f'seed{seed}_{arm}/TRAIN_DONE.json';r=json.loads(path.read_text());h=r['history'];rolepath=path.parent/'FULL864_ROLE_RECEIPT.json';role=json.loads(rolepath.read_text())
        assert r['all_gates_passed'] and r['fixed_epochs']==168 and r['seed']==seed and r['arm']==arm
        assert r['fit_sha256']==p.FIT_SHA and not r['teacher_checkpoints'] and not r['access']['development_metrics']
        assert len(h)==168 and r['sampling_weight_sha256']==weight_sha and r['export_max_abs_error']==0
        assert all(a['sampler_sha256']==b['sampler_sha256'] and b['steps']==121 for a,b in zip(expected,h,strict=True))
        assert role['selected_sampling_policy']==frozen['selected_arm'] and role['fit864_sha256']==p.FIT_SHA
        assert role['development180_consumed_as_training'] and role['auxiliary_coefficient']==0 and not role['validation_or_test_metrics_accessed']
        e=role['deformation']
        if arm=='full_rotation':
            digest,count=rotation_stream(seed,168*121)
            assert e['steps']==168*121 and e['samples']==168*968 and e['selected']==count and e['independent_random_stream_sha256']==digest
            assert e['probability']==.2 and e['max_degrees']==15. and e['max_observed_degrees']<=15.000001
        else:assert e['steps']==e['samples']==e['selected']==0
        p.checked(Path(r['checkpoint']['path']),r['checkpoint']['sha256'])
        ck=torch.load(r['checkpoint']['path'],map_location='cpu',weights_only=False)
        assert ck['fit_sha256']==p.FIT_SHA and ck['epoch']==168 and ck['seed']==seed and ck['arm']==arm and not ck['development_metrics_accessed']
        reports[arm]=r
        checks[arm]={'independent_all168_sampler_sha':True,'full864_weights_sha':weight_sha,'role_and_checkpoint_metadata':True,'rotation_stream_replayed':arm=='full_rotation','control_extra_rotation_steps':e['steps'] if arm=='full_control' else None,'export_report_error':r['export_max_abs_error']}
        hashes[arm]={'report_path':str(path),'report_sha256':p.sha(path),'role_receipt_sha256':p.sha(rolepath),'checkpoint':r['checkpoint']}
    a,b=reports['full_control'],reports['full_rotation'];assert a['initial_state_sha256']==b['initial_state_sha256']
    assert all(x['augmentation_rng_sha256']==y['augmentation_rng_sha256'] for x,y in zip(a['history'],b['history'],strict=True))
    p.write_new(output,{'status':'passed_full864_fixed168','created_utc':datetime.now(timezone.utc).isoformat(),'checks':checks,'reports':hashes,'same_initial_and_all_base_rng':True,'new_forwards':0,'source_development180_used_as_training':True,'test_access':False,'verifier_sha256':p.sha(__file__)})
    print('full864 independent training replay passed',flush=True)


if __name__=='__main__':main()
