"""Engineering admission binds fit stress and evaluation BEFORE formal training."""
from datetime import datetime,timezone
import json
import os
import subprocess
import sys
from scripts import small_rotation_protocol as p
from scripts.freeze_small_rotation import main as freeze_training


def main():
    if (p.RECORD/'ENGINEERING_ADMISSION.json').exists():raise FileExistsError('already admitted')
    stress=json.loads((p.RECORD/'FIT_ROTATION_STRESS.json').read_text())
    assert stress['status']=='passed' and stress['scans']==684 and not stress['erasures']
    p.checked(p.ROOT/'sammed3d/small_rotation.py',stress['rotation_sha256'])
    p.checked(p.ROOT/'scripts/audit_small_rotation_fit.py',stress['code_sha256'])
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'}
    run=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_small_rotation.py','tests/test_small_rotation_evaluation.py'],cwd=p.ROOT,env=env,capture_output=True,text=True)
    p.write_new(p.RECORD/'SYNTHETIC_TESTS.json',{'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr,'created_utc':datetime.now(timezone.utc).isoformat()})
    assert run.returncode==0
    eval_files=['scripts/evaluate_small_rotation.py','scripts/evaluate_structured_background_aux.py','sammed3d/t1ce_wt_capg3d.py','scripts/train_t1ce_wt_capg3d.py','tests/test_small_rotation_evaluation.py','scripts/finalize_small_rotation.py','scripts/finalize_structured_background_aux.py','scripts/audit_small_rotation_training.py','scripts/audit_subject_sampler_replay.py']
    p.write_new(p.RECORD/'DEVELOPMENT_IMPLEMENTATION_FREEZE.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'before_formal_training':True,'files':{n:p.sha(p.ROOT/n) for n in eval_files}})
    freeze_training()
    p.write_new(p.RECORD/'ENGINEERING_ADMISSION.json',{'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),'files':{n:p.sha(p.RECORD/n) for n in ['FIT_ROTATION_STRESS.json','SYNTHETIC_TESTS.json','SMOKE_RECEIPT.json','TRAIN_PREACCESS.json','IMPLEMENTATION_FREEZE.json','DEVELOPMENT_IMPLEMENTATION_FREEZE.json']},'admission_code_sha256':p.sha(__file__),'new_candidate_dev_access':False,'new_test_access':False})
    print('all training/engineering/development implementation gates frozen before formal training',flush=True)


if __name__=='__main__':main()
