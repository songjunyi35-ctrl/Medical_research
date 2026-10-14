"""Durable local supervisor: heartbeat, validation trend, failure record, final archival."""
import sys, os, json, time, subprocess, fcntl, signal, re, traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.sammed3d_graymean_single972 import ROOT, RECORD, WORK, write, sha, now
from scripts.finalize_sammed3d_single972_ddp import intervals
R=ROOT/'archive/research_records/2026-09-17_sammed3d_resume139'; OUT=WORK/'graymean'
PY=str(ROOT.parent/'sammed3d-venv/bin/python')
ENV={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'0,1','NCCL_DEBUG':'WARN'}

def trend():
    rows=[]
    committed=json.loads((OUT/'STATUS.json').read_text())['last_complete_epoch']
    for p in sorted(OUT.glob('development_e*.json')):
        d=json.loads(p.read_text())
        if d['epoch']>committed:continue
        rows.append({k:d[k] for k in ['epoch','scan_final5','subject_final5','scan_best5','scans','subjects']})
    best=max(rows,key=lambda d:d['scan_final5'])
    result={'utc':now(),'cohort':'fixed reused validation108/50','selection':'final5; earliest tie','best':best,'latest':rows[-1],'curve':rows,'historical_reference':{'dice':.594226,'cohort':'reused internal test270/119','metric':'scan best of5','direct_comparison_valid':False},'validation_best5_above_reference_number':max(d['scan_best5'] for d in rows)>.594226,'test_access':False}
    write(R/'DICE_MONITOR.json',result)
    return result

def docs(stage, t):
    latest=t['latest'];best=t['best']
    text=f"{stage}。固定972训练/108验证/270测试；双A100每卡4例、全局8。最近验证第{latest['epoch']}轮：第五点击Scan {latest['scan_final5']:.6f}，五点击最佳 {latest['scan_best5']:.6f}；按第五点击选优仍为第{best['epoch']}轮 {best['scan_final5']:.6f}。旧0.594226是test270五点击最佳，不能与本验证直接比较。未新增测试/外层/确认评估。"
    link='archive/research_records/2026-09-17_sammed3d_resume139/'
    p=ROOT/'AFLOW_BRATS_HANDOFF_CURRENT.md';s=p.read_text()
    s=re.sub(r'<!-- SAMMED3D_SINGLE972_STATUS -->.*?<!-- END_SAMMED3D_SINGLE972_STATUS -->',lambda m:'<!-- SAMMED3D_SINGLE972_STATUS -->\n'+text+' [实时记录]('+link+'DICE_MONITOR.json) · [恢复协议]('+link+'IDEA.md)\n<!-- END_SAMMED3D_SINGLE972_STATUS -->',s,flags=re.S)
    s=s.replace('更新时间：2026-09-16。','更新时间：2026-09-17。')
    s=re.sub(r'\n阶段验证：第5轮.*?\n','\n',s)
    if '完成200轮' in stage:
        s=s.replace('新单组972例200轮实验的完整结果尚待产生。','本单组200轮训练结果已归档，见下方最终报告。')
        s=s.replace('1. 按当前单组预注册核对972/108缓存、双卡单组训练和评估流程；冻结后执行200轮并据实际进度报告。旧6轮检查点不当作完整972例起点。','1. 本单组200轮已完成，审核最终验证报告；无自动测试准入。')
    p.write_text(s)
    p=ROOT/'AFLOW_BRATS_METHOD_CATALOG.md';s=p.read_text();start=s.index('## 师兄灰度校准与SAM-Med3D｜');end=s.find('\n## ',start+3)
    card='## 师兄灰度校准与SAM-Med3D｜'+stage+'\n\n**一句话：** 用区域平均亮度辅助轮廓学习。 **出发点：** 完成用户要求的固定972例单组200轮。\n**具体做法：** 原数据、损失、学习率、双卡batch和五点击评估固定；从139轮恢复，数值溢出有界重算同批。\n**结果：** '+text+'\n**是否支持直觉：** 单组无对照，未证明灰度项独立收益；不同队列的0.594226不能直接作为胜负结论。\n**代价/下一步：** 双A100每卡4例，全局8；灰度项不增推理前向，工程重算成本留痕。按固定协议完成和审核，不自动进入测试。[核对材料]('+link+'RESULT.md)\n'
    s=s[:start]+card+(s[end:] if end>=0 else '')
    duplicate='## SAM-Med3D固定972单组灰度项｜200轮完成'
    if duplicate in s:
        a=s.index(duplicate); b=s.find('\n## ',a+3);s=s[:a]+(s[b:] if b>=0 else '')
    p.write_text(s)
    p=ROOT/'AFLOW_BRATS_RESULTS.md';s=p.read_text();block='<!-- RESUME139_PROGRESS -->\n## 2026-09-17｜SAM-Med3D灰度组恢复与监控\n\n**问题：** 139轮后梯度异常中断。 **做法：** 同组同数据同batch，从139轮恢复，仅增加有界数值重算及持续监控。 **结果：** '+text+' **决定：** 按预注册完成200轮，无自动测试准入。[核对材料]('+link+'RESULT.md)\n<!-- END_RESUME139_PROGRESS -->'
    if '<!-- RESUME139_PROGRESS -->' in s:s=re.sub(r'<!-- RESUME139_PROGRESS -->.*?<!-- END_RESUME139_PROGRESS -->',lambda m:block,s,flags=re.S)
    else:s+='\n'+block+'\n'
    p.write_text(s)

def event(kind, **kwargs):
    with (R/'EVENTS.jsonl').open('a') as f:f.write(json.dumps({'utc':now(),'event':kind,**kwargs},ensure_ascii=False)+'\n')

def archive_validation(t):
    d=t['latest'];p=OUT/f"development_e{d['epoch']:03d}.json";raw=json.loads(p.read_text());stats=intervals(raw['rows'])
    report={**stats,'epoch':d['epoch'],'status':'intermediate_validation','selection_best':t['best'],'report_sha256':sha(p),'preaccess_sha256':sha(R/'PREACCESS.json'),'freeze_sha256':sha(R/'FREEZE.json'),'cache_index_sha256':sha(RECORD/'CACHE_INDEX.json'),'best_checkpoint_record':json.loads((OUT/'BEST.json').read_text()),'test_outer_confirmation_access':False,'no_causal_control':True}
    write(R/f"VALIDATION_E{d['epoch']:03d}.json",report)
    write(R/'RESULT.json',report)
    (R/'RESULT.md').write_text(f"# 原单组训练恢复进展\n\n最近固定108扫描/50患者验证第{d['epoch']}轮：第五点击Scan {d['scan_final5']:.6f}，Subject {d['subject_final5']:.6f}，Scan患者整簇95%区间{stats['scan_final5_ci95']}；五点击最佳Scan {d['scan_best5']:.6f}。\n\n按第五点击选择的历史最佳第{t['best']['epoch']}轮 {t['best']['scan_final5']:.6f}。这是同验证选优，非独立确认。旧0.594226是内部test270五点击最佳，不可跨队列判断超越。灰度帮助轮廓的直觉尚不能由单组因果验证。未访问test像素或新增外层/确认。\n\n双A100、每卡4/全局8、保持原损失/学习率/数据/评估，仅修复数值溢出的同批重算。从139轮恢复至200；完整轮数以[心跳](HEARTBEAT.json)为准。检查点和原始报告仍在原实验目录，139轮另有备份。\n\n[指标与哈希](RESULT.json) · [曲线](DICE_MONITOR.json) · [工程修订七问](IDEA.md)。\n")

def checks():
    rows=[]
    for name in ['verify_aflow_brats_doc_layout.py','verify_aflow_brats_method_catalog.py','verify_aflow_brats_leaderboard.py','verify_aflow_root_artifacts.py']:
        p=subprocess.run([PY,'scripts/'+name],cwd=ROOT,env=ENV,capture_output=True,text=True);rows.append({'name':name,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    write(R/'VALIDATORS.json',rows)
    if any(r['returncode'] for r in rows):raise RuntimeError('document checks failed')

def main():
    lock=(R/'SUPERVISOR.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for path,h in json.loads((R/'FREEZE.json').read_text())['files'].items():
        if sha(ROOT/path)!=h:raise RuntimeError('freeze mismatch '+path)
    assert json.loads((R/'AMP_TEST.json').read_text())['status']=='passed'
    assert json.loads((R/'DDP_SMOKE.json').read_text())['status']=='passed'
    logpath=R/'train.log';log=logpath.open('a');p=None
    try:
        p=subprocess.Popen([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','scripts/train_sammed3d_single972_resume139.py','train','--batch','4'],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(R/'LAUNCH.json',{'utc':now(),'pid':p.pid,'supervisor_pid':os.getpid(),'worker':'1098698'})
        completion_since=None;cleanup_after_commit=False
        t=trend();archive_validation(t);last=t['latest']['epoch'];docs('已启动139轮后续训，目标200轮',t);checks();event('launched',pid=p.pid)
        while p.poll() is None:
            status=json.loads((OUT/'STATUS.json').read_text());t=trend()
            write(R/'HEARTBEAT.json',{'utc':now(),'alive':True,'pid':p.pid,'last_complete_epoch':status['last_complete_epoch'],'latest_validation':t['latest'],'best_validation':t['best'],'log_age_seconds':time.time()-logpath.stat().st_mtime})
            write(RECORD/'RUN_STATUS.json',{'updated_utc':now(),'stage':'单组双卡恢复训练中，目标200轮','last_complete_epoch':status['last_complete_epoch'],'train':972,'validation':108,'test':270,'world_size':2,'global_batch':8,'test_access':False,'heartbeat':str((R/'HEARTBEAT.json').relative_to(ROOT))})
            if t['latest']['epoch']!=last:
                last=t['latest']['epoch'];archive_validation(t);docs(f"双卡续训中，已完成{status['last_complete_epoch']}/200轮",t);event('validation',**t['latest']);checks()
            if status.get('status')=='completed' and status['last_complete_epoch']==200:
                completion_since=completion_since or time.monotonic()
                if time.monotonic()-completion_since>45:
                    event('cleanup_after_durable_completion',pid=p.pid)
                    cleanup_after_commit=True;os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                    break
            if time.time()-logpath.stat().st_mtime>1200:raise RuntimeError('training log stalled >20min')
            time.sleep(15)
        if p.returncode and not cleanup_after_commit:raise RuntimeError(f'training exited {p.returncode}')
        status=json.loads((OUT/'STATUS.json').read_text())
        if status['last_complete_epoch']!=200 or status['status']!='completed':raise RuntimeError('200 full epochs not complete')
        t=trend();archive_validation(t)
        final=subprocess.run([PY,'scripts/finalize_sammed3d_single972_ddp.py'],cwd=ROOT,env=ENV,capture_output=True,text=True)
        (R/'finalize.log').write_text(final.stdout+final.stderr)
        if final.returncode:raise RuntimeError('final archival failed')
        result=json.loads((RECORD/'RESULT.json').read_text());result['recovery_evidence']={str(q.relative_to(ROOT)):sha(q) for q in [R/'PREACCESS.json',R/'FREEZE.json',R/'IDEA.md',R/'AMP_TEST.json',R/'DDP_SMOKE.json']};result['amp_replay_events']=sum('amp_same_batch_replay' in line for line in logpath.read_text().splitlines());write(R/'RESULT.json',result)
        write(RECORD/'RESULT.json',result)
        (R/'RESULT.md').write_text('# 原单组139轮恢复至200轮：已完成\n\n固定972扫描完整200轮，单组灰度校准。选中第'+str(result['selected_epoch'])+'轮；复用108/50验证第五点击Scan '+str(result['scan_final5'])+'，Subject '+str(result['subject_final5'])+'，Scan患者整簇95%区间'+str(result['scan_final5_ci95'])+'。\n\n无新增测试/外层/确认，无对照不能证明灰度因果收益。旧0.594226属于test270五点击最佳，本轮不作跨队列胜出结论。\n\n[完整结果与原始报告入口](../2026-09-16_sammed3d_graymean_single972/RESULT.md) · [含恢复证据与溢出次数的机器报告](RESULT.json) · [监控曲线](DICE_MONITOR.json)。\n')
        docs('已完成200轮',t);checks()
        write(R/'HEARTBEAT.json',{'utc':now(),'alive':False,'status':'completed','last_complete_epoch':200,'best_validation':t['best']})
        write(RECORD/'RUN_STATUS.json',{'updated_utc':now(),'stage':'200轮完成，报告和校验已归档','last_complete_epoch':200,'test_access':False})
        event('completed',best=t['best'])
    except BaseException as exc:
        if p is not None and p.poll() is None:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        status=json.loads((OUT/'STATUS.json').read_text());failure={'utc':now(),'alive':False,'status':'failed','error':repr(exc),'traceback':traceback.format_exc(),'last_complete_epoch':status['last_complete_epoch']}
        write(R/'HEARTBEAT.json',failure);write(R/'FAILURE.json',failure)
        write(RECORD/'RUN_STATUS.json',{'updated_utc':now(),'stage':'恢复执行失败，未完成200轮',**failure})
        docs('续训失败，需核对原因',trend());event('failed',error=repr(exc));raise
    finally:log.close()

if __name__=='__main__':main()
