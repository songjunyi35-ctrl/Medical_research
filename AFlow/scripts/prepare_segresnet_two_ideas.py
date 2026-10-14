"""Freeze new entries, then separately audit train/validation cache bytes on worker."""
import sys, json, subprocess, ast, inspect
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.segresnet_two_ideas_common import *
PY=str(ROOT.parent/'sammed3d-venv/bin/python')
FILES=['scripts/segresnet_two_ideas_common.py','scripts/train_segresnet_two_ideas.py',
       'scripts/build_segresnet_two_ideas_classes.py','scripts/download_segresnet_two_ideas_pretrain.py',
       'scripts/build_segresnet_two_ideas_pretrain.py','scripts/prepare_segresnet_two_ideas.py',
       'scripts/run_segresnet_two_ideas.py','scripts/finalize_segresnet_two_ideas.py',
       'sammed3d/segresnet_semantic_wt.py','tests/test_segresnet_two_ideas.py',
       'scripts/segresnet_plain972_common.py','scripts/train_t1ce_wt_capg3d.py','scripts/train_t1ce_wt_plain_long_schedule.py',
       'scripts/build_t1ce_wt_cache.py','scripts/evaluate_t1ce_wt_plain_locator_test270.py',
       'sammed3d/subregion_labels.py','sammed3d/spatial_roundtrip_audit.py','sammed3d/t1ce_wt_capg3d.py',
       'sammed3d/t1ce_binary.py','sammed3d/t1ce_wt_training.py','sammed3d/t1ce_wt_cached.py']


def freeze():
    if (R/'FREEZE.json').exists() or (R/'PIPELINE_START.json').exists() or OUT.exists():raise RuntimeError('already frozen/started')
    R.mkdir(parents=True,exist_ok=True)
    for name in FILES:
        ast.parse((ROOT/name).read_text(),filename=name)
    result=subprocess.run([PY,'-m','pytest','tests/test_segresnet_two_ideas.py','-q'],cwd=ROOT,capture_output=True,text=True,
                          env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    write(R/'UNIT_TESTS.json',{'utc':now(),'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
                              'patient_pixels_access':False})
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    import monai
    from monai.networks.nets import segresnet
    from monai.losses import dice
    write(R/'ENVIRONMENT.json',{'utc':now(),'torch':torch.__version__,'monai':monai.__version__,
          'segresnet_source_sha256':sha(inspect.getsourcefile(segresnet)),'dice_source_sha256':sha(inspect.getsourcefile(dice))})
    old=json.loads((ROOT/'archive/research_records/2026-09-17_segresnet_plain972_baseline/DATA_AUDIT.json').read_text())
    roles={}
    for role,path in MANIFEST.items():
        assert sha(path)==old['manifest_sha256'][str(path.relative_to(ROOT))]
        rows=read_rows(path);roles[role]={r['subject_id'] for r in rows}
        assert (len(rows),len(roles[role]))=={'train':(972,444),'validation':(108,50),'test':(270,119)}[role]
    assert not (roles['train']&roles['validation'] or roles['train']&roles['test'] or roles['validation']&roles['test'])
    extras=[*MANIFEST.values(),SOURCE_INVENTORY,TRAIN_CACHE/'index.json',VAL_CACHE/'index.json',
            ROOT/'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/test270/index.json',
            BASELINE_RUN/'initial.pth',BASELINE_RUN/'validation_e193.json',
            ROOT/'archive/research_records/2026-09-17_segresnet_plain972_e193_test270/RAW_TEST270.json',
            R/'UNIT_TESTS.json',R/'ENVIRONMENT.json',R/'IDEA.md']
    extras += [p/'IDEA.md' for p in RECORDS.values()]
    assert sha(BASELINE_RUN/'initial.pth')=='0c232550b9ce17b16fc5ce0ec751a83e2fc597d3a5ffc7ca94ab9dbab1e67dbd'
    files={name:sha(ROOT/name) for name in FILES}
    files.update({str(p.relative_to(ROOT)):sha(p) for p in extras})
    write(R/'FREEZE.json',{'utc':now(),'status':'frozen_before_gpu_smoke_and_patient_access','files':files,
          'candidates':list(RECORDS),'execution_order':['semantic_regions','pretrain','finetune_pretrained'],
          'world_size':2,'per_gpu_batch':4,'seed':BASELINE_SEED,'downstream_epochs':200,'pretrain_epochs':150,
          'no_test_tuned_training':True,'test_authorized_one_validation_selected_checkpoint_per_candidate':True,
          'test_grid':128,'threshold_raw_wt_logit':0,'precision':'bfloat16','selection':'validation108 scan Dice, earliest tie',
          'point_is_secondary_not_gate':True,'new_outer_confirmation_access':False,'worker':1104883,
          'maximum_engineering_resumes':2,'disk_persistent_reserve_bytes':2*(1<<30)})
    print(json.dumps({'frozen_files':len(files),'unit_tests':'passed'}))


def audit():
    verify(); old=json.loads((ROOT/'archive/research_records/2026-09-17_segresnet_plain972_baseline/DATA_AUDIT.json').read_text())
    if (R/'DATA_AUDIT.json').exists():raise RuntimeError('audit already done')
    write(R/'DATA_PREACCESS.json',{'utc':now(),'role':'train and validation cache integrity only','test_pixels_access':False,
                                 'freeze_sha256':sha(R/'FREEZE.json')})
    caches=[]
    for item in old['caches']:
        path=ROOT/item['path']; assert sha(path/'index.json')==item['index_sha256']
        for name,digest in item['payload_sha256'].items():
            actual=sha(path/name);assert actual==digest, str(path/name)
        caches.append(item)
    inv,cases=selected_cases()
    result={'utc':now(),'status':'passed','train_case_ids':[r['case_id'] for r in read_rows(MANIFEST['train'])],
            'validation_case_ids':[r['case_id'] for r in read_rows(MANIFEST['validation'])],
            'pretrain_case_ids':[cache_case_id(c['subject_id']) for c in cases],'caches':caches,'test_access':False,
            'manifest_sha256':{k:sha(p) for k,p in MANIFEST.items()},'patient_overlap':0}
    write(R/'DATA_AUDIT.json',result)


def execution_gate():
    verify()
    paths=[R/'DATA_AUDIT.json',R/'GPU_SMOKE_semantic_regions.json',R/'GPU_SMOKE_pretrain.json',R/'CLASS_CACHE_RECEIPT.json',
           CLASS_CACHE/'index.json']
    for p in paths:assert json.loads(p.read_text()).get('status','passed')=='passed'
    receipt=json.loads((R/'CLASS_CACHE_RECEIPT.json').read_text()); assert sha(CLASS_CACHE/'classes.u8')==receipt['classes_sha256']
    write(R/'EXECUTION_GATE.json',{'utc':now(),'files':{str(p):sha(p) for p in paths},'freeze_sha256':sha(R/'FREEZE.json')})


if __name__=='__main__':{'freeze':freeze,'audit':audit,'gate':execution_gate}[sys.argv[1]]()
