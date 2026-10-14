"""Resume after saved panel; no repeated selection, panel scoring or model training."""
import sys,time,fcntl,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import scripts.complete_t1ce_10h_v3
from scripts import complete_t1ce_10h as original
from scripts.export_t1ce_10h_v2 import export
from scripts.t1ce_10h_common import *

def main():
 lock=(R/'completion_owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert (R/'FINAL_SELECTION.json').exists()
 export();original.remote_stage('deploy_t1ce_10h.py','check');original.access_summary();original.publish();original.checks();original.evidence()
 write(R/'COMPLETION_RECOVERED.json',{'utc':now(),'status':'passed','resumed_after':'saved final panel; export metadata only','no_panel_rerun':True,'new_training':False})
 fcntl.flock(lock,fcntl.LOCK_UN);lock.close();original.main()

if __name__=='__main__':
 try:main()
 except BaseException:write(R/f'COMPLETION_RESUME_FATAL_{int(time.time())}.json',{'utc':now(),'traceback':traceback.format_exc()});raise
