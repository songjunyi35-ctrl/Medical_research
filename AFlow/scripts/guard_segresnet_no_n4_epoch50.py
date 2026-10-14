"""User-directed epoch50 decision guard; no change to frozen training source."""
import sys,os,json,time,signal,traceback,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.segresnet_plain972_no_n4_common import ROOT,R,OUT,sha,write,now,summary_ci
from scripts.evaluate_plain972_e193_flip8_validation import paired
BASE=ROOT/'workspace/BraTS/segresnet_t1ce_wt/plain972_baseline_20260917'
A=R/'EPOCH50_REVIEW'


def decide(delta,ci,recent_mean):
    clear=abs(delta)>=.01 and (ci[0]>0 or ci[1]<0) and delta*recent_mean>0
    return {'clear_difference':bool(clear),'continue_training':bool(clear),'absolute_scan_delta_at_least_0p01':bool(abs(delta)>=.01),'paired_ci_excludes_zero':bool(ci[0]>0 or ci[1]<0),'recent_mean_same_direction':bool(delta*recent_mean>0),'reason':'clear_same_epoch_difference_continue_original_schedule' if clear else 'no_clear_difference_stop_at50_not_proven_equivalence'}


def read_logs(directory):
    rows=[]
    for line in (directory/'TRAIN.jsonl').read_text().splitlines():
        if line.strip():rows.append(json.loads(line))
    return {r['epoch']:r for r in rows}


def process_identity(pid,needle):
    p=Path(f'/proc/{pid}/cmdline')
    value=p.read_bytes().replace(b'\0',b' ').decode()
    if needle not in value:raise RuntimeError(f'PID {pid} no longer belongs to {needle}')
    return {'pid':pid,'cmdline':value,'pgid':os.getpgid(pid),'stat_starttime':Path(f'/proc/{pid}/stat').read_text().split()[21]}


def inspect_processes():
    launch=json.loads((R/'LAUNCH.json').read_text());pipeline=json.loads((R/'PIPELINE_START.json').read_text())
    train=process_identity(launch['pid'],'scripts/train_segresnet_plain972_no_n4.py')
    supervisor=process_identity(launch['supervisor_pid'],'scripts/watch_segresnet_plain972_no_n4.py')
    parent=process_identity(pipeline['pid'],'scripts/run_segresnet_plain972_no_n4.py')
    assert train['pgid']==train['pid'] and os.getpgid(0)!=train['pgid']
    return {'train':train,'supervisor':supervisor,'pipeline':parent}


def pause(processes):
    # Freeze orchestration first so a deliberate stop cannot be rewritten as failure.
    for key in ['pipeline','supervisor']:os.kill(processes[key]['pid'],signal.SIGSTOP)
    os.killpg(processes['train']['pgid'],signal.SIGSTOP)
    write(A/'PAUSED.json',{'utc':now(),'processes':processes,'durable_epoch_target':50,'possible_unsaved_next_epoch_work':'guard observes committed epoch; a fraction of epoch51 may start before SIGSTOP; never reported as epoch51 complete'})


def resume(processes):
    os.killpg(processes['train']['pgid'],signal.SIGCONT)
    for key in ['supervisor','pipeline']:os.kill(processes[key]['pid'],signal.SIGCONT)


def terminate(processes):
    # Parent/supervisor have default SIGTERM handlers; do not let them publish failure.
    for key in ['pipeline','supervisor']:
        try:os.kill(processes[key]['pid'],signal.SIGTERM);os.kill(processes[key]['pid'],signal.SIGCONT)
        except ProcessLookupError:pass
    try:os.killpg(processes['train']['pgid'],signal.SIGTERM);os.killpg(processes['train']['pgid'],signal.SIGCONT)
    except ProcessLookupError:pass
    time.sleep(4)
    try:os.killpg(processes['train']['pgid'],signal.SIGKILL)
    except ProcessLookupError:pass
    for key in ['pipeline','supervisor']:
        try:os.kill(processes[key]['pid'],signal.SIGKILL)
        except ProcessLookupError:pass


def publish_stop(logs,comparison,decision):
    from scripts.watch_segresnet_plain972_no_n4 import archive,documents,checks
    import torch
    log=logs[50];selected=json.loads((OUT/'BEST.json').read_text())
    evidence={}
    for name,epoch in [('epoch050.pth',50),('latest.pth',50),('best.pth',selected['epoch'])]:
        p=OUT/name;state=torch.load(p,map_location='cpu',weights_only=False);assert state['epoch']==epoch
        assert all(bool(torch.isfinite(t).all()) for t in state['model_state_dict'].values());del state
        evidence[str(p.relative_to(ROOT))]=sha(p)
    assert evidence[str((OUT/'latest.pth').relative_to(ROOT))]==log['checkpoint_sha256']
    assert evidence[str((OUT/'best.pth').relative_to(ROOT))]==selected['checkpoint_sha256']
    archive(log)
    final=json.loads((R/'RESULT.json').read_text())
    final.update(status='stopped_by_user_epoch50_rule',completed_epochs=50,original_target_epochs=200,completed_original_200_epochs=False,optimizer_updates=6100,new_test_pixels_access=False,decision=decision,comparison= comparison,checkpoint_evidence=evidence,stop_amendment_sha256=sha(A/'AMENDMENT.json'),selected_validation_summary=summary_ci(json.loads((OUT/selected['report']).read_text())['per_case']),interpretation='No clear early same-epoch difference under amended savings rule; not equivalence or completed200 conclusion; no new test/outer/confirmation')
    write(R/'STOP_RESULT.json',final);write(R/'FINAL_RESULT.json',final);write(R/'RESULT.json',final)
    status={'status':'stopped_by_user_epoch50_rule','last_complete_epoch':50,'best_epoch':selected['epoch'],'best_scan_dice':selected['scan_dice'],'optimizer_updates':6100,'updated_utc':now(),'test_pixels_access':False,'resume_authorized':False}
    write(OUT/'STATUS.json',status);write(R/'HEARTBEAT.json',{'utc':now(),'alive':False,**status});write(R/'PIPELINE_STATUS.json',{'utc':now(),'status':'stopped_by_user_epoch50_rule','test_access':False})
    text='# 无额外N4纯SegResNet：用户规则下完成50轮后停止\n\n'
    text+=f"实际完成50/200轮、6100更新，保留第50轮快照、latest和前50轮最佳第{selected['epoch']}轮；未完成原200轮。\n\n"
    text+=f"第50轮同108扫描/50患者验证：N4={comparison['epoch50_N4_scan']:.6f}，无N4={comparison['epoch50_noN4_scan']:.6f}，无N4−N4={comparison['epoch50_paired']['scan_delta']:+.6f}，患者配对95%CI={comparison['epoch50_paired']['scan_delta_ci95']}；41–50轮平均差值={comparison['recent10_mean_scan_delta']:+.6f}。\n\n"
    text+='没有达到事先补充的明确差异门，按用户省时指令停止。属于看过早期轨迹后新增的自适应停止规则，不是原始200轮预注册完成，也不证明等效或最终无效。保持单T1ce/972-108-270；没有新test/outer/confirmation，原test揭示历史保留。不自动恢复训练。\n\n[完整比较](EPOCH50_REVIEW/COMPARISON.json) · [停止收据](STOP_RESULT.json) · [用户规则补记](EPOCH50_REVIEW/AMENDMENT.md)。\n'
    (R/'RESULT.md').write_text(text)
    documents('50轮按用户规则停止，禁止自动恢复',50,selected)
    p=ROOT/'AGENTS.md';s=p.read_text();block='## 2026-09-18 无N4组按50轮用户规则停止（最新终态）\n\n- 已完成50/200轮、6100更新；50轮与旧N4的固定108验证比较未达到新增明确差异门，按用户要求节省算力停止。保留50轮/前50最佳/latest，训练与监督退出；没有新test。不可称等效或200轮完成，不自动恢复。收据：`archive/research_records/2026-09-18_segresnet_plain972_no_n4/STOP_RESULT.json`。\n\n'
    p.write_text(s.replace('## 2026-09-18',block+'## 2026-09-18',1))
    p=ROOT/'AFLOW_BRATS_HANDOFF_CURRENT.md';s=p.read_text().replace('当前已授权一组无额外N4的SegResNet基础对照，准备/训练状态见下方实时记录；无新测试。','无额外N4组已按用户50轮比较规则停止，实际50/200；无运行中训练或新测试。');p.write_text(s)
    p=ROOT/'AFLOW_BRATS_WEEKLY_REPORT.md';s=p.read_text()+f"\n无N4基础对照按用户省时规则停止于50/200轮；同轮无N4−N4验证差{comparison['epoch50_paired']['scan_delta']:+.6f}、CI{comparison['epoch50_paired']['scan_delta_ci95']}，未达明确差异门，不是等效证明，无新test。[终态](archive/research_records/2026-09-18_segresnet_plain972_no_n4/RESULT.md)。\n";p.write_text(s)
    checks()
    write(A/'STOP_COMPLETE.json',{'utc':now(),'status':'completed','stop_result_sha256':sha(R/'STOP_RESULT.json'),'checkpoints':evidence})


def main():
    lock=(A/'GUARD.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    amendment=json.loads((A/'AMENDMENT.json').read_text())
    for name,h in amendment['files'].items():assert sha(ROOT/name)==h
    assert not (A/'GUARD_STARTED.json').exists()
    processes=inspect_processes();write(A/'GUARD_STARTED.json',{'utc':now(),'pid':os.getpid(),'processes':processes,'target_epoch':50,'amendment_sha256':sha(A/'AMENDMENT.json')})
    tick=0
    while True:
        status=json.loads((OUT/'STATUS.json').read_text())
        if int(status['last_complete_epoch'])>=50:break
        if status['status']!='running':raise RuntimeError('training stopped before target')
        if time.monotonic()-tick>15:
            inspect_processes();write(A/'HEARTBEAT.json',{'utc':now(),'status':'waiting_for_epoch50','last_complete_epoch':status['last_complete_epoch'],'pid':os.getpid()});tick=time.monotonic()
        time.sleep(.25)
    processes=inspect_processes();pause(processes)
    logs=read_logs(OUT);old=read_logs(BASE)
    if max(logs)!=50:
        terminate(processes);raise RuntimeError('guard missed exact durable50; stopped rather than silently continue')
    left=json.loads((BASE/'validation_e050.json').read_text())['per_case'];right=json.loads((OUT/'validation_e050.json').read_text())['per_case']
    rows=[{'epoch':e,'N4_scan':old[e]['validation']['locator_dsc_mean'],'noN4_scan':logs[e]['validation']['locator_dsc_mean'],'delta_noN4_minus_N4':logs[e]['validation']['locator_dsc_mean']-old[e]['validation']['locator_dsc_mean']} for e in range(1,51)]
    best_old=max(range(1,51),key=lambda e:old[e]['validation']['locator_dsc_mean']);best_new=max(range(1,51),key=lambda e:logs[e]['validation']['locator_dsc_mean'])
    comparison={'utc':now(),'role':'same reused validation108/50','rows':rows,'epoch50_N4_scan':rows[-1]['N4_scan'],'epoch50_noN4_scan':rows[-1]['noN4_scan'],'epoch50_paired':paired(left,right),'recent10_mean_scan_delta':float(np.mean([r['delta_noN4_minus_N4'] for r in rows[-10:]])),'recent10_noN4_higher_epochs':[r['epoch'] for r in rows[-10:] if r['delta_noN4_minus_N4']>0],'matched_budget_best':{'N4':{'epoch':best_old,'scan':old[best_old]['validation']['locator_dsc_mean']},'noN4':{'epoch':best_new,'scan':logs[best_new]['validation']['locator_dsc_mean']}},'new_test_access':False,'new_inference':False,'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [BASE/'validation_e050.json',OUT/'validation_e050.json',BASE/'TRAIN.jsonl',OUT/'TRAIN.jsonl']}}
    decision=decide(comparison['epoch50_paired']['scan_delta'],comparison['epoch50_paired']['scan_delta_ci95'],comparison['recent10_mean_scan_delta'])
    write(A/'COMPARISON.json',comparison);write(A/'DECISION.json',{'utc':now(),**decision})
    if decision['continue_training']:
        resume(processes)
        (A/'RESULT.md').write_text('# 第50轮检查：存在明确同轮差异，继续原计划\n\n'+json.dumps({'comparison':comparison['epoch50_paired'],'recent10_mean_delta':comparison['recent10_mean_scan_delta'],'decision':decision},ensure_ascii=False,indent=2)+'\n\n仅为复用验证的阶段判断，不自动进入test。\n')
        write(A/'HEARTBEAT.json',{'utc':now(),'status':'decision_complete_training_resumed','last_complete_epoch':50})
        from scripts.watch_segresnet_plain972_no_n4 import checks
        checks()
    else:
        terminate(processes);publish_stop(logs,comparison,decision)
        write(A/'HEARTBEAT.json',{'utc':now(),'status':'stopped_training_at50','last_complete_epoch':50})

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(A/'GUARD_FAILURE.json',{'utc':now(),'error':repr(e),'traceback':traceback.format_exc(),'instruction':'If PAUSED exists and no stop receipt, keep training paused pending repair; never silently resume'});raise
