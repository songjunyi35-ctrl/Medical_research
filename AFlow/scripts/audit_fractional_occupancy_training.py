"""Metadata-only independent draw replay and unchanged-image/RNG control audit."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
from scripts import fractional_occupancy_protocol as p
from scripts.audit_subject_sampler_replay import independent_weights,replay


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
    p.check_freeze();fit,_=p.check_roles();cases=[r['case_id'] for r in fit];subjects={r['case_id']:r['subject_id'] for r in fit};volumes=json.loads(p.NATIVE.read_text())['native_wt_voxels']
    weights=independent_weights(cases,subjects,volumes)['control'];weight_sha=hashlib.sha256(weights.numpy().tobytes()).hexdigest()
    results={};hashes={}
    for seed in p.SEEDS:
        work=p.WORK/f'seed{seed}_fractional_occupancy';path=work/('TRAIN_DONE.json' if args.final else 'progress.json')
        body=path.read_bytes();candidate=json.loads(body);hashes[str(path.relative_to(p.ROOT))]=hashlib.sha256(body).hexdigest()
        control_path=p.source.WORK/f'seed{seed}_control/TRAIN_DONE.json';control=json.loads(control_path.read_text());hashes[str(control_path.relative_to(p.ROOT))]=p.sha(control_path)
        started=json.loads((work/'STARTED.json').read_text());assert started['initial_state_sha256']==control['initial_state_sha256']
        h=candidate['history'];epochs=len(h);assert 0<epochs<=168
        if args.final:assert epochs==168 and candidate['all_gates_passed'] and candidate['sampling_weight_sha256']==weight_sha
        expected=replay(weights,seed,epochs,cases,subjects,volumes)
        assert weight_sha==control['sampling_weight_sha256']
        assert all(a['sampler_sha256']==b['sampler_sha256']==c['sampler_sha256'] and b['augmentation_rng_sha256']==c['augmentation_rng_sha256'] and b['first_batch']['image_sha256']==c['first_batch']['image_sha256'] and b['steps']==121 for a,b,c in zip(expected,h,control['history'][:epochs],strict=True))
        assert any(a['first_batch']['labels_sha256']!=b['first_batch']['labels_sha256'] for a,b in zip(h,control['history'][:epochs],strict=True))
        results[str(seed)]={'epochs_verified':epochs,'initial_exact':True,'all_effective_draws_exact':True,'original_augmentation_rng_exact':True,'all_logged_first_batch_image_bytes_exact':True,'targets_changed':True,'weight_sha256':weight_sha,'mean_unique_subjects_per_epoch':statistics.fmean(r['unique_subjects'] for r in expected)}
    output=p.RECORD/('TRAINING_REPLAY_FINAL.json' if args.final else 'TRAINING_REPLAY_'+datetime.now(timezone.utc).strftime('%H%M%S')+'.json')
    p.write_new(output,{'status':'passed_full168' if args.final else 'passed_snapshot','created_utc':datetime.now(timezone.utc).isoformat(),'results':results,'source_snapshot_sha256':hashes,'script_sha256':p.sha(Path(__file__)),'independent_sampler_script_sha256':p.sha(p.ROOT/'scripts/audit_subject_sampler_replay.py'),'access':'fit_metadata_and_training_logs_only','new_model_forwards':0,'candidate_development_access':False})
    print(json.dumps({'path':str(output),'results':results},indent=2),flush=True)


if __name__=='__main__':main()
