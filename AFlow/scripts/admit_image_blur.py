"""Conditional engineering admission only after the prioritized rotation closes."""
from datetime import datetime,timezone
import json
import os
import subprocess
import sys
from scripts import image_blur_protocol as p
from scripts.freeze_image_blur import main as freeze_training


def main():
    parent=p.ROOT/'archive/research_records/2026-09-10_small_rotation'
    assert (parent/'TRAINING_FAILURE.json').exists() or ((parent/'RESULT.json').exists() and json.loads((parent/'RESULT.json').read_text())['selected_arm'] is None)
    assert datetime.now(timezone.utc)<datetime(2026,9,10,19,0,tzinfo=timezone.utc)
    if (p.RECORD/'ENGINEERING_ADMISSION.json').exists():raise FileExistsError('already admitted')
    smoke=json.loads((p.RECORD/'SMOKE_RECEIPT.json').read_text());assert smoke['status']=='passed'
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    run=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_image_blur.py','tests/test_image_blur_evaluation.py'],cwd=p.ROOT,env=env,capture_output=True,text=True)
    p.write_new(p.RECORD/'SYNTHETIC_TESTS.json',{'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr,'created_utc':datetime.now(timezone.utc).isoformat()})
    assert run.returncode==0
    files=['scripts/evaluate_image_blur.py','scripts/finalize_image_blur.py','scripts/audit_image_blur_training.py','tests/test_image_blur_evaluation.py','scripts/evaluate_structured_background_aux.py','scripts/finalize_structured_background_aux.py','scripts/audit_subject_sampler_replay.py','scripts/train_t1ce_wt_capg3d.py','sammed3d/t1ce_wt_capg3d.py']
    p.write_new(p.RECORD/'DEVELOPMENT_IMPLEMENTATION_FREEZE.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'before_formal_training':True,'files':{n:p.sha(p.ROOT/n) for n in files}})
    freeze_training()
    p.write_new(p.RECORD/'ENGINEERING_ADMISSION.json',{'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),'code_sha256':p.sha(__file__),'files':{n:p.sha(p.RECORD/n) for n in ['SYNTHETIC_TESTS.json','SMOKE_RECEIPT.json','TRAIN_PREACCESS.json','IMPLEMENTATION_FREEZE.json','DEVELOPMENT_IMPLEMENTATION_FREEZE.json']},'prioritized_rotation_closed':True,'candidate_dev_access':False,'test_access':False})
    print('conditional image-blur pair admitted',flush=True)


if __name__=='__main__':main()
