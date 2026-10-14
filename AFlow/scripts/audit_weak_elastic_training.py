"""Independent sampler and new random-stream replay from metadata only."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import torch
from scripts import weak_elastic_protocol as p
from scripts.audit_subject_sampler_replay import independent_weights,replay


def deformation_stream(seed,steps):
    generator=torch.Generator().manual_seed(seed+100000)
    digest=hashlib.sha256();selected=0
    for _ in range(steps):
        flags=torch.rand(8,generator=generator)<.2
        values=torch.rand(8,3,7,7,7,generator=generator)*2-1
        selected+=int(flags.sum())
        digest.update(flags.numpy().tobytes());digest.update(values.numpy().tobytes())
    return digest.hexdigest(),selected


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
    p.check_freeze();fit,_=p.check_roles();torch.set_num_threads(1)
    cases=[r['case_id'] for r in fit];subjects={r['case_id']:r['subject_id'] for r in fit}
    volumes=json.loads(p.NATIVE.read_text())['native_wt_voxels']
    weights=independent_weights(cases,subjects,volumes)
    results={};sources={}
    for seed in p.SEEDS:
        for arm in p.ARMS:
            key=f'{seed}_{arm}';work=p.WORK/f'seed{key}'
            path=work/('TRAIN_DONE.json' if args.final else 'progress.json')
            if not path.exists():
                if args.final:raise FileNotFoundError(path)
                continue
            body=path.read_bytes();candidate=json.loads(body);sources[str(path.relative_to(p.ROOT))]=hashlib.sha256(body).hexdigest()
            h=candidate['history'];epochs=len(h);assert 0<epochs<=168
            if args.final:assert epochs==168 and candidate['all_gates_passed']
            control=json.loads(p.control_path(seed,arm).read_text())
            initial=json.loads((work/'STARTED.json').read_text())['initial_state_sha256']
            assert initial==control['initial_state_sha256']
            w=weights['subject_balanced' if arm=='subject_elastic' else 'control']
            replayed=replay(w,seed,epochs,cases,subjects,volumes)
            assert all(x['sampler_sha256']==y['sampler_sha256']==z['sampler_sha256'] and y['augmentation_rng_sha256']==z['augmentation_rng_sha256'] and y['steps']==121 for x,y,z in zip(replayed,h,control['history'][:epochs],strict=True))
            digest=hashlib.sha256(w.numpy().tobytes()).hexdigest()
            assert digest==control['sampling_weight_sha256']
            if args.final:assert digest==candidate['sampling_weight_sha256']
            ep=work/('ELASTIC_FINAL.json' if args.final else 'ELASTIC_PROGRESS.json')
            eb=ep.read_bytes();e=json.loads(eb);sources[str(ep.relative_to(p.ROOT))]=hashlib.sha256(eb).hexdigest()
            expected,count=deformation_stream(seed,e['steps'])
            assert expected==e['independent_random_stream_sha256'] and count==e['selected']
            assert e['samples']==e['steps']*8
            results[key]={'epochs_verified':epochs,'initial_exact':True,'all_sampler_indices_sha_exact':True,'base_augmentation_rng_exact':True,'weight_sha256':digest,'deformation_steps_verified':e['steps'],'deformation_stream_exact':True,'selected_samples':count,'augmentation_fraction':count/e['samples'],'mean_unique_scans_per_epoch':statistics.fmean(x['unique_scans'] for x in replayed),'mean_unique_subjects_per_epoch':statistics.fmean(x['unique_subjects'] for x in replayed)}
    if args.final:assert len(results)==4
    output=p.RECORD/('TRAINING_REPLAY_FINAL.json' if args.final else 'TRAINING_REPLAY_'+datetime.now(timezone.utc).strftime('%H%M%S')+'.json')
    p.write_new(output,{'status':'passed_full168' if args.final else 'passed_snapshot','created_utc':datetime.now(timezone.utc).isoformat(),'results':results,'source_snapshot_sha256':sources,'script_sha256':p.sha(Path(__file__)),'independent_weights_script_sha256':p.sha(p.ROOT/'scripts/audit_subject_sampler_replay.py'),'access':{'fit_metadata_training_logs':True,'patient_pixels':False,'checkpoint_load':False,'development_metrics':False,'test':False},'new_model_forwards':0})
    print(json.dumps({'path':str(output),'results':results},indent=2),flush=True)


if __name__=='__main__':main()
