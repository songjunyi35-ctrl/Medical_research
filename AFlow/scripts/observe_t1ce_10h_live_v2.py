"""Transport failure is unknown process state, not evidence of process exit."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import observe_t1ce_10h_live as original

remote=original.remote
def observe_remote(port,pid):
 result=remote(port,pid);code=result['returncode']
 if code==255:
  result.update(alive=None,observation_status='ssh_transport_unavailable',process_exit_not_inferred=True)
 elif code==0:
  expected='supervise_t1ce_10h_v3.py dual' if port==11323 else 'supervise_t1ce_10h_v2.py single'
  if expected not in result['ps']:result.update(alive=None,observation_status='pid_identity_mismatch')
  else:result['observation_status']='process_observed'
 elif code==1:result['observation_status']='pid_missing_on_reachable_endpoint'
 else:result.update(alive=None,observation_status='unknown_observation_error')
 return result
original.remote=observe_remote

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seconds',type=int,default=1800);original.main(p.parse_args().seconds)
