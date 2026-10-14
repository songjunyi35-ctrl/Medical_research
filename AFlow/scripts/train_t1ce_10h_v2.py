"""Engineering-only NCCL cleanup exit after synchronized committed output."""
import sys,os,threading,traceback,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import train_t1ce_10h as original
from scripts.t1ce_10h_common import R,read,write,now

destroy=original.dist.destroy_process_group
def committed_cleanup():
 arm=sys.argv[2];mode=sys.argv[1]
 marker=R/arm/('GPU_SMOKE.json' if mode=='smoke' else 'STOP_RESULT.json')
 assert marker.exists(),'refuse success exit before committed result'
 timer=threading.Timer(20,lambda:os._exit(0));timer.daemon=True;timer.start()
 destroy();sys.stdout.flush();sys.stderr.flush();os._exit(0)
original.dist.destroy_process_group=committed_cleanup

if __name__=='__main__':
 try:{'smoke':original.smoke,'train':original.train}[sys.argv[1]](sys.argv[2])
 except BaseException:
  write(R/sys.argv[2]/f'FAILURE_v2_{sys.argv[1]}_{os.environ.get("RANK","0")}_{int(time.time())}.json',{'utc':now(),'traceback':traceback.format_exc()});raise
