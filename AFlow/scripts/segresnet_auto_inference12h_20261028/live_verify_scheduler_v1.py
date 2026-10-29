"""Run fresh270 for audited core incumbents once phase inference releases memory."""
import fcntl,os,subprocess,sys
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_cached_v1 import guard
from scripts.segresnet_auto_inference12h_20261028.deploy_core_v3 import export_recipe


def main():
    lock=(R/'LIVE_VERIFY_SCHEDULER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    token=read(R/'RESOURCE_CURRENT.json')['token'];deadline=read(R/'AUTHORIZATION.json')['deadline_unix']
    write(R/'LIVE_VERIFY_SCHEDULER_PID.json',dict(at=now(),pid=os.getpid(),worker=1153287,resource_token=token,source_sha256=sha(Path(__file__))))
    failed=set();unsupported=set()
    while time.time()<deadline:
        guard();assert read(R/'RESOURCE_CURRENT.json')['token']==token
        inc=read(R/'INCUMBENT.json');source=ROOT/Path(inc['result_path']).parent;candidate=source.name
        folder=R/'live_verification'/candidate
        if (folder/'DONE.json').exists() or candidate in failed:
            write(R/'LIVE_VERIFY_SCHEDULER_STATUS.json',dict(at=now(),state='watching_new_incumbents',candidate=candidate,already_passed=(folder/'DONE.json').exists(),failed_once=candidate in failed));time.sleep(15);continue
        if not (R/'jobs/a2_0000_polyphase512/DONE.json').exists():
            write(R/'LIVE_VERIFY_SCHEDULER_STATUS.json',dict(at=now(),state='waiting_for_phase_parent',candidate=candidate));time.sleep(15);continue
        if deadline-time.time()<900:
            write(R/'LIVE_VERIFY_SCHEDULER_STATUS.json',dict(at=now(),state='not_starting_in_final15minutes',candidate=candidate));time.sleep(15);continue
        query=subprocess.run(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
        if query.returncode!=0:
            write(R/'LIVE_VERIFY_SCHEDULER_STATUS.json',dict(at=now(),state='memory_observation_unavailable',candidate=candidate,error=query.stderr[:500]));time.sleep(15);continue
        free_mib=int(query.stdout.strip())
        if free_mib<35*1024:
            write(R/'LIVE_VERIFY_SCHEDULER_STATUS.json',dict(at=now(),state='waiting_for_memory',candidate=candidate,physical_gpu=1,free_mib=free_mib,required_mib=35*1024));time.sleep(15);continue
        recipe=R/'portable_verification_recipes'/(candidate+'.json')
        try:export_recipe(source,recipe)
        except AssertionError as error:
            if candidate not in unsupported:
                write(R/'LIVE_VERIFY_UNSUPPORTED.json',dict(at=now(),candidate=candidate,error=str(error),requires='corresponding group predictor; core API must not pretend support'));unsupported.add(candidate)
            time.sleep(15);continue
        folder.mkdir(parents=True,exist_ok=True)
        write(folder/'SCHEDULER_FREEZE.json',dict(at=now(),candidate=candidate,physical_gpu=1,observed_free_mib=free_mib,phase_parent_done_sha256=sha(R/'jobs/a2_0000_polyphase512/DONE.json'),source_sha256=sha(Path(__file__)),recipe_sha256=sha(recipe),reference_raw_sha256=sha(source/'RAW_TEST270.json'),GT_input_to_predictor=False,test_tuned=True,training=False))
        env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']='1';env['MPLCONFIGDIR']=str(TMP/'matplotlib')
        logfile=TMP/('live_verify_'+candidate+'.log')
        with logfile.open('a') as output:
            child=subprocess.Popen([sys.executable,'-m','scripts.segresnet_auto_inference12h_20261028.verify_portable_live_v3','--candidate',candidate,'--recipe',str(recipe)],cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
            while child.poll() is None:
                write(R/'LIVE_VERIFY_SCHEDULER_STATUS.json',dict(at=now(),state='fresh270_running',candidate=candidate,pid=child.pid,physical_gpu=1,log=str(logfile)));time.sleep(10)
        if child.returncode!=0 or not (folder/'DONE.json').exists():
            failed.add(candidate)
            write(folder/'SCHEDULER_FAILED.json',dict(at=now(),candidate=candidate,exit_code=child.returncode,no_automatic_retry=True,log=str(logfile)))
        else:write(R/'LIVE_VERIFY_SCHEDULER_LAST_PASS.json',read(folder/'DONE.json'))
    write(R/'LIVE_VERIFY_SCHEDULER_STATUS.json',dict(at=now(),state='window_closed'))

if __name__=='__main__':main()
