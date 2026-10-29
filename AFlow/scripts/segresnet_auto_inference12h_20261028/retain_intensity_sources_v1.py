"""End completed rolling-pruning supervision; retain28 inputs for group fusion."""
import datetime,os,signal,time
from scripts.segresnet_auto_inference12h_20261028.common import *


def main():
    baseline=read(R/'INTENSITY_BUDGET_GATE_POSTPUBLICATION.json')['worst_case_total_bytes']
    f4=read(R/'analyses/f4_0000_brain_phase/PLAN.json')['extra_scratch_upper_bound_bytes']
    h1=sum(v['bytes'] for v in read(R/'analyses/h1_0000_local_brain_joint/CODE_FILES.json'))+625*4*270*4+64*2**20
    reserve=1536*2**20;worst=baseline+f4+h1+reserve
    assert worst<300*2**30
    registration=read(R/'INTENSITY_REGISTRATION.json')
    for item in registration['groups']:
        done=R/'jobs'/item['render']/'DONE.json'
        if done.exists():assert len(list((TMP/'groups'/item['render']/'mean_probability').glob('*.npy')))==270,'a source was already pruned'
    pid=read(R/'ROLLING_CLEANUP_PID.json')['pid'];assert pid==8077
    proc=Path('/proc')/str(pid);assert b'rolling_cleanup_v1' in (proc/'cmdline').read_bytes() and proc.stat().st_uid==os.getuid()
    folder=R/'amendments/007_retain_intensity_sources';folder.mkdir(parents=True,exist_ok=True)
    while True:
        status=read(R/'ROLLING_CLEANUP_STATUS.json');age=time.time()-datetime.datetime.fromisoformat(status['at']).timestamp();wait_channel=(proc/'wchan').read_text()
        if 'nanosleep' in wait_channel and 0<=age<20:break
        time.sleep(1)
    write(folder/'STOP_PLAN.json',dict(at=now(),pid=pid,cmdline='rolling_cleanup_v1',wait_channel=wait_channel,status_age_seconds=age,reason='Prior complete sources pruned; retain future28 intensity means for planned joint fusion',source_sha256=sha(Path(__file__)),worst_scratch_bytes=worst,extra_ensemble_reserve_bytes=reserve,inference_or_model_restart=False,recovery_count_changed=False))
    os.kill(pid,signal.SIGTERM)
    for _ in range(50):
        if not proc.exists() or (proc/'stat').read_text().split()[2]=='Z':break
        time.sleep(.1)
    else:raise AssertionError('cleanup process did not terminate')
    write(folder/'STOPPED.json',dict(at=now(),state='complete',pid=pid,all28_sources_retained_for_fusion=True,existing_model_processes_untouched=True,recovery_count_changed=False,bytes_pruned_after_this_stop=0))
    print(dict(state='complete',stopped_only_cleanup_pid=pid,worst_scratch_gib=worst/2**30,model_restarts=0))

if __name__=='__main__':main()
