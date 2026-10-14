"""Bounded single-thread CPU regression/validator report, not model evaluation."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'archive/research_records/2026-09-10_resumed_eight_hour'


def main():
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'}
    stamp=datetime.now(timezone.utc).strftime('%H%M%S');output=REC/f'QUALITY_{stamp}.json'
    commands=[[sys.executable,'-m','pytest','-q','tests'],*[ [sys.executable,'scripts/'+name] for name in ['verify_aflow_brats_method_catalog.py','verify_aflow_brats_doc_layout.py','verify_aflow_brats_leaderboard.py']]]
    results=[]
    for command in commands:
        started=time.monotonic()
        try:
            run=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=600)
            results.append({'command':command,'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr,'seconds':time.monotonic()-started})
        except subprocess.TimeoutExpired as error:
            results.append({'command':command,'returncode':None,'timeout':True,'seconds':time.monotonic()-started,'error':str(error)})
        print(command[-1]+': '+str(results[-1]['returncode']),flush=True)
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'passed' if all(r['returncode']==0 for r in results) else 'failed','results':results,'gpu_disabled':True,'threads_per_math_library':1,'formal_model_evaluation':False}
    with output.open('x') as f:json.dump(report,f,indent=2)
    print(str(output),flush=True)
    if report['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
