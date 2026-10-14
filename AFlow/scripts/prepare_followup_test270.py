import sys,subprocess,inspect,ast,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.followup_test270_common import *
FILES=['scripts/followup_test270_common.py','scripts/build_followup_test270_cache.py','scripts/evaluate_followup_test270.py',
       'scripts/prepare_followup_test270.py','scripts/publish_followup_test270.py','scripts/run_followup_test270.py','tests/test_followup_test270.py',
       'scripts/build_segresnet_native_res_test_cache.py','scripts/segresnet_native_res_test_common.py','scripts/segresnet_native_res_common.py',
       'scripts/segresnet_plain972_common.py','scripts/segresnet_two_ideas_common.py','scripts/train_t1ce_wt_capg3d.py',
       'scripts/train_t1ce_wt_plain_long_schedule.py','scripts/evaluate_t1ce_wt_plain_locator_test270.py','scripts/build_t1ce_wt_cache.py',
       'sammed3d/t1ce_wt_capg3d.py','sammed3d/t1ce_binary.py','sammed3d/t1ce_wt_training.py','sammed3d/t1ce_wt_cached.py']


def freeze():
    assert not (S/'FREEZE.json').exists()
    for n in FILES:ast.parse((ROOT/n).read_text())
    roles={};expected={'train':('6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65',972,444),
      'validation':('f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb',108,50),
      'test':('a5ddc0ed2c0b269431758b14061da3c6163dd0c48133b60a2ef8efd7f23aa8f9',270,119)}
    for role,p in MANIFEST.items():
        h,n,m=expected[role];rows=read_rows(p);assert sha(p)==h and len(rows)==len({x['case_id'] for x in rows})==n
        roles[role]={x['subject_id'] for x in rows};assert len(roles[role])==m
    assert not (roles['train']&roles['validation'] or roles['train']&roles['test'] or roles['validation']&roles['test'])
    for key,r in RECORDS.items():
        assert sha(r/'checkpoint_selected.pth')==HASHES[key]
        assert json.loads((TRAIN_RECORDS[key]/'DONE.json').read_text())['test_access'] is False
    assert sha(PRETRAINED/'checkpoint_selected.pth')==REF_SHA
    unit=subprocess.run([sys.executable,'-m','pytest','tests/test_followup_test270.py','-q'],cwd=ROOT,capture_output=True,text=True)
    write(S/'UNIT_TESTS.json',{'utc':now(),'returncode':unit.returncode,'stdout':unit.stdout,'stderr':unit.stderr})
    if unit.returncode:raise RuntimeError(unit.stdout+unit.stderr)
    import monai
    from monai.networks.nets import segresnet
    write(S/'ENVIRONMENT.json',{'utc':now(),'torch':torch.__version__,'monai':monai.__version__,'segresnet_source_sha256':sha(inspect.getsourcefile(segresnet))})
    paths=[ROOT/n for n in FILES]+list(MANIFEST.values())+[S/'IDEA.md',S/'UNIT_TESTS.json',S/'ENVIRONMENT.json',CANONICAL_TEST/'index.json',
      OLD_NATIVE/'CACHE_RECEIPT.json',OLD_NATIVE/'RAW_TEST270.json',PRETRAINED/'checkpoint_selected.pth',PRETRAINED/'RAW_TEST270.json',PURE/'RAW_TEST270.json']
    for key,r in RECORDS.items():paths.extend([r/'checkpoint_selected.pth',r/'SELECTION_RECEIPT.json',r/'IDEA.md',TRAIN_RECORDS[key]/'RESULT.json'])
    freeze={'utc':now(),'status':'frozen_before_new_test_pixels_and_model_access','files':{str(p.relative_to(ROOT)):sha(p) for p in paths},
            'candidates':{'native':{'epoch':166,'checkpoint_sha256':HASHES['native'],'canvas':list(TEST_CANVAS)},
                          'patient':{'epoch':172,'checkpoint_sha256':HASHES['patient'],'grid':128}},
            'reference179_sha256':REF_SHA,'reference179_role':'one already-tested-model native-grid projection, no new selection',
            'cohort':{'scans':270,'subjects':119},'precision':'bf16','batch':2,'threshold':0,'tta':False,'postprocessing':False,
            'bootstrap':'10000 patient clusters; absolute20260917, paired20260918; 95% and97.5% descriptive',
            'test_authorized':True,'test_tuning_used':False,'new_outer_confirmation_authorized':False,'automatic_repeat':False}
    write(S/'FREEZE.json',freeze)
    for r in RECORDS.values():write(r/'FREEZE.json',{'utc':now(),'shared_freeze':str((S/'FREEZE.json').relative_to(ROOT)),
              'shared_freeze_sha256':sha(S/'FREEZE.json'),'selection_sha256':sha(r/'SELECTION_RECEIPT.json')})
    print('freeze passed')


def smoke():
    verify();assert torch.cuda.is_available();torch.set_num_threads(2);seed_all(20260819);numeric_mode();results=[]
    for key,shape in [('native',TEST_CANVAS),('patient',(128,128,128))]:
        model=model_for(torch.device('cuda:0'));state=torch.load(RECORDS[key]/'checkpoint_selected.pth',map_location='cpu',weights_only=False)
        assert state['epoch']==EPOCHS[key];model.load_state_dict(state['model_state_dict'],strict=True);del state;model.eval()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):y=model(torch.randn(2,1,*shape,device='cuda')).segmentation_logits
        assert y.shape==(2,1,*shape) and bool(torch.isfinite(y).all())
        results.append({'candidate':key,'shape':list(y.shape),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'passed':True})
        del y,model;torch.cuda.empty_cache()
    write(S/'GPU_SMOKE.json',{'utc':now(),'status':'passed','results':results,'patient_pixels_access':False})


def audit():
    verify();assert json.loads((S/'GPU_SMOKE.json').read_text())['status']=='passed'
    for key,r in RECORDS.items():write_new(r/'PREACCESS.json',{'utc':now(),'authorization':'user explicitly authorized both fixed candidates test270',
              'test_pixels_access':True,'candidate_model_forward_yet':False,'checkpoint_sha256':HASHES[key],
              'selection_receipt_sha256':sha(r/'SELECTION_RECEIPT.json'),'freeze_sha256':sha(S/'FREEZE.json'),
              'cohort':{'scans':270,'subjects':119},'current_test_tuning':False,'new_outer_confirmation_access':False})
    write_new(S/'CACHE_PREACCESS.json',{'utc':now(),'roles':['test270'],'test_pixels_access':True,'model_forward_yet':False})
    index=json.loads((CANONICAL_TEST/'index.json').read_text());hashes={}
    assert index['manifest_sha256']==sha(MANIFEST['test']) and index['case_ids']==[r['case_id'] for r in read_rows(MANIFEST['test'])]
    for filename,key,width in [('images.f32','images_sha256',4),('labels.u8','labels_sha256',1)]:
        p=CANONICAL_TEST/filename;assert p.stat().st_size==270*128**3*width;hashes[filename]=sha(p);assert hashes[filename]==index[key]
    receipt={'utc':now(),'status':'passed','count':270,'subjects':119,'cache_index_sha256':sha(CANONICAL_TEST/'index.json'),
             'payload_sha256':hashes,'source':str(CANONICAL_TEST),'test_access':True}
    write(S/'CANONICAL_RECEIPT.json',receipt);write(RECORDS['patient']/'CACHE_RECEIPT.json',receipt)


if __name__=='__main__':{'freeze':freeze,'smoke':smoke,'audit':audit}[sys.argv[1]]()
