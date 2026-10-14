"""Freeze code/roles before new access, then audit canonical inputs and execution gates."""
import sys,os,json,subprocess,ast,inspect
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.segresnet_pretrained_native_common import *
PY=str(ROOT.parent/'sammed3d-venv/bin/python')
FILES=['scripts/segresnet_pretrained_native_common.py','scripts/train_segresnet_pretrained_native.py',
       'scripts/build_segresnet_pretrained_native_cache.py','scripts/evaluate_pretrained_native_references.py',
       'scripts/prepare_pretrained_native.py','scripts/run_pretrained_native.py','scripts/finalize_pretrained_native.py',
       'tests/test_segresnet_pretrained_native.py','scripts/train_t1ce_wt_capg3d.py','scripts/train_t1ce_wt_plain_long_schedule.py',
       'scripts/build_t1ce_wt_cache.py','sammed3d/t1ce_binary.py','sammed3d/t1ce_wt_training.py',
       'sammed3d/t1ce_wt_cached.py','sammed3d/t1ce_wt_capg3d.py']


def freeze():
    assert not (R/'FREEZE.json').exists() and not OUT.exists();R.mkdir(parents=True,exist_ok=True)
    for f in FILES:ast.parse((ROOT/f).read_text(),filename=f)
    test=subprocess.run([PY,'-m','pytest','tests/test_segresnet_pretrained_native.py','-q'],cwd=ROOT,capture_output=True,text=True,
                        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    write(R/'UNIT_TESTS.json',{'utc':now(),'returncode':test.returncode,'stdout':test.stdout,'stderr':test.stderr})
    if test.returncode:raise RuntimeError(test.stdout+test.stderr)
    old=json.loads((ROOT/'archive/research_records/2026-09-17_segresnet_plain972_baseline/DATA_AUDIT.json').read_text());sets={}
    for role,path in MANIFEST.items():
        assert sha(path)==old['manifest_sha256'][str(path.relative_to(ROOT))]
        rows=read_rows(path);sets[role]={x['subject_id'] for x in rows}
        assert len(rows)==len({x['case_id'] for x in rows})=={'train':972,'validation':108,'test':270}[role]
        assert len(sets[role])=={'train':444,'validation':50,'test':119}[role]
    assert not (sets['train']&sets['validation'] or sets['train']&sets['test'] or sets['validation']&sets['test'])
    assert sha(PRETRAIN_CKPT)==PRETRAIN_SHA
    for _,(path,digest,_,_) in REFERENCES.items():assert sha(path)==digest
    import monai
    from monai.networks.nets import segresnet
    from monai.losses import dice
    write(R/'ENVIRONMENT.json',{'utc':now(),'torch':torch.__version__,'monai':monai.__version__,
          'segresnet_sha256':sha(inspect.getsourcefile(segresnet)),'dice_sha256':sha(inspect.getsourcefile(dice))})
    paths=[ROOT/f for f in FILES]+list(MANIFEST.values())+[PRETRAIN_CKPT,R/'IDEA.md',R/'UNIT_TESTS.json',R/'ENVIRONMENT.json']
    paths += [p/'index.json' for p in CANONICAL.values()]
    for _,(p,_,report,_) in REFERENCES.items():paths.extend([p,report])
    write(R/'FREEZE.json',{'utc':now(),'status':'frozen_before_new_patient_or_gpu_smoke_access','files':{str(p.relative_to(ROOT)):sha(p) for p in paths},
          'candidate':'public_pretrain150_then_native_resolution_full972','initialization_sha256':PRETRAIN_SHA,
          'train_scans':972,'validation_scans':108,'test_metadata_only':270,'epochs':200,'per_gpu_batch':4,'global_batch':8,'updates_per_epoch':122,
          'canvas':list(CANVAS),'training_numeric_mode':'native legacy fused AdamW, cudnn benchmark and TF32',
          'evaluation_numeric_mode':'all three references and all candidate validation deterministic, benchmark off, matmul TF32 off, bf16 batch2',
          'score_grid':'native crop, padding excluded','threshold':0,'tta':False,'postprocessing':False,
          'selection':'maximum validation Scan Dice; earliest tie','primary_comparisons':['native173','pretrained179'],
          'promotion_gate':'scan>=stronger_single+0.01 and both paired97.5% lower bounds>0; descriptive reused validation, not independent confirmation',
          'bootstrap':'10000 patient cluster absolute seed20260917; paired seed20260918; 95% and 97.5%',
          'test_authorized':False,'test_tuning_used':False,'new_outer_confirmation_authorized':False,
          'worker':1104883,'max_engineering_resumes':2,'wall_budget_seconds':WALL_BUDGET_SECONDS})
    print(json.dumps({'status':'frozen','files':len(paths),'unit_tests':'passed'}))


def audit():
    verify();assert not (R/'CANONICAL_AUDIT.json').exists()
    write(R/'CANONICAL_PREACCESS.json',{'utc':now(),'roles':['train','validation'],'purpose':'byte integrity only','test_pixels_access':False})
    old=json.loads((ROOT/'archive/research_records/2026-09-17_segresnet_plain972_baseline/DATA_AUDIT.json').read_text())
    for c in old['caches']:
        p=ROOT/c['path'];assert sha(p/'index.json')==c['index_sha256']
        for f,h in c['payload_sha256'].items():assert sha(p/f)==h
    write(R/'CANONICAL_AUDIT.json',{'utc':now(),'status':'passed','caches':old['caches'],'test_pixels_access':False})


def gate():
    verify();paths=[R/'GPU_SMOKE.json',R/'BATCH_DECISION.json',R/'DATA_AUDIT.json',R/'CANONICAL_AUDIT.json',R/'REFERENCE_NATIVE_VALIDATION.json']
    for p in paths:assert json.loads(p.read_text()).get('status','passed')=='passed'
    assert batch_decision()==4
    data=json.loads((R/'DATA_AUDIT.json').read_text())
    for c in data['caches']:
        p=Path(c['path']);assert sha(p/'index.json')==c['index_sha256']
        assert sha(p/'images.f32')==c['images_sha256'] and sha(p/'labels.u8')==c['labels_sha256']
        paths.append(p/'index.json')
    write(R/'EXECUTION_GATE.json',{'utc':now(),'files':{str(p):sha(p) for p in paths},'freeze_sha256':sha(R/'FREEZE.json')})


if __name__=='__main__':{'freeze':freeze,'audit':audit,'gate':gate}[sys.argv[1]]()
