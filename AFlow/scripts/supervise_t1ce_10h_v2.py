"""Engineering amendment001: wait for separately authorized concurrent evaluation."""
import sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import supervise_t1ce_10h as original
from scripts.t1ce_10h_common import *

old_wait=original.wait_for
def coordinated_wait(which,path):
 old_wait(which,path)
 foreign=ROOT/'archive/research_records/2026-09-21_segresnet_followup_test270/PIPELINE_STATUS.json'
 while foreign.exists() and read(foreign).get('running'):
  original.heartbeat(which,'waiting_for_independent_followup_evaluation',foreign_status=read(foreign))
  if time.time()>timestamp(P['new_training_stop_utc']):raise RuntimeError('independent task did not release resources by deadline')
  time.sleep(15)
original.wait_for=coordinated_wait

if __name__=='__main__':
 try:original.main(sys.argv[1])
 except BaseException:
  write(R/f'SUPERVISOR_V2_FAILURE_{sys.argv[1]}_{int(time.time())}.json',{'utc':now(),'traceback':traceback.format_exc()});raise
