import sys,json,subprocess,ast,inspect
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.patient_equal_common import *
FILES=['scripts/patient_equal_common.py','scripts/train_patient_equal.py','scripts/prepare_patient_equal.py',
 'scripts/run_patient_equal.py','scripts/finalize_patient_equal.py','sammed3d/patient_equal_loss.py','tests/test_patient_equal_loss.py',
 'scripts/segresnet_plain972_common.py','scripts/segresnet_two_ideas_common.py','scripts/train_t1ce_wt_capg3d.py',
 'scripts/train_t1ce_wt_plain_long_schedule.py','sammed3d/t1ce_wt_capg3d.py','sammed3d/t1ce_wt_cached.py','sammed3d/segresnet_semantic_wt.py']


def freeze():
    assert not (R/'FREEZE.json').exists() and not OUT.exists()
    for f in FILES:ast.parse((ROOT/f).read_text())
    old=json.loads((ROOT/'archive/research_records/2026-09-17_segresnet_plain972_baseline/DATA_AUDIT.json').read_text())
    groups={};roles={}
    for name,p in MANIFEST.items():
        assert sha(p)==old['manifest_sha256'][str(p.relative_to(ROOT))]
        rows=read_rows(p);roles[name]=rows;groups[name]={x['subject_id'] for x in rows}
        assert len(rows)==len({x['case_id'] for x in rows})=={'train':972,'validation':108,'test':270}[name]
        assert len(groups[name])=={'train':444,'validation':50,'test':119}[name]
        assert all(x['case_id'].rsplit('-',1)[0]==x['subject_id'] for x in rows)
    assert not (groups['train']&groups['validation'] or groups['train']&groups['test'] or groups['validation']&groups['test'])
    weights=patient_weights(roles['train']);counts=collections.Counter(x['subject_id'] for x in roles['train'])
    assert abs(sum(weights.values())-972)<1e-9
    for s in counts:assert abs(sum(weights[x['case_id']] for x in roles['train'] if x['subject_id']==s)-972/444)<1e-12
    write(R/'WEIGHTS.json',{'utc':now(),'formula':'972/(444*n_subject)','case_weights':weights,'subject_scan_counts':dict(counts),
          'train_manifest_sha256':sha(MANIFEST['train']),'generated_from':'training subject metadata only; no validation/test features or labels'})
    test=subprocess.run([sys.executable,'-m','pytest','tests/test_patient_equal_loss.py','-q'],cwd=ROOT,capture_output=True,text=True)
    write(R/'UNIT_TESTS.json',{'utc':now(),'returncode':test.returncode,'stdout':test.stdout,'stderr':test.stderr})
    if test.returncode:raise RuntimeError(test.stdout+test.stderr)
    import monai
    from monai.losses import dice
    write(R/'ENVIRONMENT.json',{'utc':now(),'torch':torch.__version__,'monai':monai.__version__,'dice_source_sha256':sha(inspect.getsourcefile(dice))})
    assert sha(SOURCE)==SOURCE_SHA
    paths=[ROOT/f for f in FILES]+list(MANIFEST.values())+[SOURCE,BASELINE_REPORT,PRETRAINED_REPORT,TRAIN_CACHE/'index.json',VAL_CACHE/'index.json',R/'IDEA.md',R/'WEIGHTS.json',R/'UNIT_TESTS.json',R/'ENVIRONMENT.json']
    write(R/'FREEZE.json',{'utc':now(),'status':'frozen_before_gpu_smoke_and_patient_access','files':{str(p.relative_to(ROOT)):sha(p) for p in paths},
          'epochs':200,'seed':SEED,'world_size':1,'worker':1105022,'global_batch':8,'tail_batch':4,'steps_per_epoch':122,
          'every_scan_once':True,'objective':'patient weighted per-image Dice+.1 BCE; fixed global-mean-one weights, no batch renormalization',
          'primary_metric':'ordinary unweighted validation Scan Dice','selection':'maximum, earliest tie','threshold':0,'tta':False,
          'postprocessing':False,'primary_reference':'existing pretrained179 validation','secondary_reference':'fixed pure193 validation',
          'test_authorized':False,'test_tuning_used':False,'outer_confirmation_authorized':False,'single_vs_dual_rng_difference_disclosed':True})
    print(json.dumps({'status':'frozen','weights':len(weights),'subjects':len(counts),'min_weight':min(weights.values()),'max_weight':max(weights.values())}))


def audit():
    verify();assert not (R/'DATA_AUDIT.json').exists()
    write(R/'DATA_PREACCESS.json',{'utc':now(),'roles':['train','validation'],'purpose':'cache hashes only','test_access':False})
    old=json.loads((ROOT/'archive/research_records/2026-09-17_segresnet_plain972_baseline/DATA_AUDIT.json').read_text())
    for c in old['caches']:
        p=ROOT/c['path'];assert sha(p/'index.json')==c['index_sha256']
        for name,h in c['payload_sha256'].items():assert sha(p/name)==h
    write(R/'DATA_AUDIT.json',{'utc':now(),'status':'passed','train_case_ids':[x['case_id'] for x in read_rows(MANIFEST['train'])],
          'validation_case_ids':[x['case_id'] for x in read_rows(MANIFEST['validation'])],'caches':old['caches'],'test_access':False})


if __name__=='__main__':{'freeze':freeze,'audit':audit}[sys.argv[1]]()
