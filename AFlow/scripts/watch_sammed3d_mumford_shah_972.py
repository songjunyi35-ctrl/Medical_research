"""Durable monitoring and per-validation archival for the one approved MS run."""
import sys,os,json,time,subprocess,signal,fcntl,re,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.train_sammed3d_mumford_shah_972 import ROOT,R,OUT,write,sha,now,verify
PY=str(ROOT.parent/'sammed3d-venv/bin/python')
ENV={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','NCCL_DEBUG':'WARN'}
REL=str(R.relative_to(ROOT));TITLE='## 三维Mumford–Shah区域灰度能量｜'


def stats(rows):
    subjects=sorted({r['subject_id'] for r in rows});n=np.array([sum(r['subject_id']==s for r in rows) for s in subjects])
    ix=np.random.default_rng(20260917).integers(0,len(subjects),(10000,len(subjects)))
    result={'scans':len(rows),'subjects':len(subjects),'bootstrap':'10000 patient cluster percentile95 seed20260917'}
    for metric in ['best5','final5']:
        sums=np.array([sum(r[metric] for r in rows if r['subject_id']==s) for s in subjects])
        result['scan_'+metric]=float(sums.sum()/n.sum());result['subject_'+metric]=float((sums/n).mean())
        result['scan_'+metric+'_ci95']=np.quantile(sums[ix].sum(1)/n[ix].sum(1),[.025,.975]).tolist()
        result['subject_'+metric+'_ci95']=np.quantile((sums/n)[ix].mean(1),[.025,.975]).tolist()
    return result


def checks():
    results=[]
    for name in ['verify_aflow_brats_doc_layout.py','verify_aflow_brats_method_catalog.py','verify_aflow_brats_leaderboard.py','verify_aflow_root_artifacts.py']:
        p=subprocess.run([PY,'scripts/'+name],cwd=ROOT,env=ENV,capture_output=True,text=True);results.append({'name':name,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    write(R/'VALIDATORS.json',results)
    if any(r['returncode'] for r in results):raise RuntimeError('documentation validation failed')


def documents(stage,epoch=0,best=None):
    brief=f'新单组三维Mumford–Shah灰度能量实验：{stage}，实际完成{epoch}/200轮、{epoch*2}次参数更新。'
    if best:brief+=f" 固定复用108/50验证最优第{best['epoch']}轮，best5 Scan {best['scan_best5']:.6f}。"
    brief+=' 原972/108/270固定；双A100每卡4例微批，每轮累计2次更新；无本实验新增test/outer/confirmation。方法与更新量为用户新授权，受既往test揭示影响，非盲测或单因素对照。'
    p=ROOT/'AFLOW_BRATS_HANDOFF_CURRENT.md';s=p.read_text();block='<!-- MUMFORD_SHAH_SINGLE972_STATUS -->\n'+brief+' [运行/结果]('+REL+'/RESULT.md) · [心跳]('+REL+'/HEARTBEAT.json)\n<!-- END_MUMFORD_SHAH_SINGLE972_STATUS -->'
    if '<!-- MUMFORD_SHAH_SINGLE972_STATUS -->' in s:s=re.sub(r'<!-- MUMFORD_SHAH_SINGLE972_STATUS -->.*?<!-- END_MUMFORD_SHAH_SINGLE972_STATUS -->',lambda m:block,s,flags=re.S)
    else:s=s.replace('## 当前状态','## 当前状态\n\n'+block,1)
    p.write_text(s)
    p=ROOT/'AFLOW_BRATS_METHOD_CATALOG.md';s=p.read_text();a=s.find(TITLE);b=s.find('\n## ',a+3) if a>=0 else -1
    card=TITLE+stage+'\n\n**一句话：** 逐体素检查前后景亮度一致性，再限制破碎边界。 **出发点：** 旧灰度均值项未超过基线且训练更新量未对齐。\n**具体做法：** 新Turbo单组，替换区域/边界能量并按用户选择恢复每轮2更新；固定972/108/270和原SAM输入，每轮验证best5选优。\n**结果：** '+brief+'\n**是否支持直觉：** '+('仅复用验证选优，需按完整结果判断；无单因素对照。' if epoch else '尚无新精度结果。')+'\n**代价/下一步：** 双A100单组，复用缓存；目标200轮，无灰度额外推理。作者二维能量已作三维和体积归一适配，不称原样复现。[核对材料]('+REL+'/RESULT.md)\n'
    s=s[:a]+card+(s[b:] if b>=0 else '') if a>=0 else s+'\n'+card;p.write_text(s)
    p=ROOT/'AFLOW_BRATS_RESULTS.md';s=p.read_text();entry='<!-- MUMFORD_SHAH_SINGLE972_LOG -->\n## 2026-09-17｜新Mumford–Shah单组训练\n\n**问题：** 旧前景均值匹配与更新量存在复现缺口。 **做法：** 用户指定作者区域能量三维适配及每轮2更新，Turbo新训练。 **结果：** '+brief+' **决定：** 按固定200轮与逐轮验证执行，不自动测test。[核对材料]('+REL+'/RESULT.md)。\n<!-- END_MUMFORD_SHAH_SINGLE972_LOG -->'
    if '<!-- MUMFORD_SHAH_SINGLE972_LOG -->' in s:s=re.sub(r'<!-- MUMFORD_SHAH_SINGLE972_LOG -->.*?<!-- END_MUMFORD_SHAH_SINGLE972_LOG -->',lambda m:entry,s,flags=re.S)
    else:s+='\n'+entry+'\n'
    p.write_text(s)
    return brief


def archive(log,terminal=False):
    report=OUT/f"validation_e{log['epoch']:03d}.json";raw=json.loads(report.read_text())
    if sha(report)!=log['validation_report_sha256']:raise RuntimeError('validation report SHA changed')
    result=stats(raw['rows']);result.update(epoch=log['epoch'],status='completed200' if terminal else 'running',selected_checkpoint=log['selection'],completed_epochs=log['epoch'],cumulative_optimizer_updates=log['cumulative_optimizer_updates'],prior_test_informed_method_choice=True,test_pixels_access=False,outer_confirmation_access=False,training=log['train'],training_seconds_this_epoch=log['training_seconds'],elapsed_seconds=log['elapsed_seconds'],peak_allocated_bytes_per_gpu=log['peak_allocated_bytes'],checkpoint_sha256=log['checkpoint_sha256'],raw_report=str(report.relative_to(ROOT)),raw_report_sha256=sha(report),freeze_sha256=sha(R/'FREEZE.json'),preaccess_sha256=sha(R/'TRAIN_PREACCESS.json'),validation_preaccess_sha256=sha(R/'VALIDATION_PREACCESS.json'),cache_index_sha256=json.loads((R/'DATA_AUDIT.json').read_text())['cache_index_sha256'])
    write(R/f"VALIDATION_E{log['epoch']:03d}.json",result);write(R/'RESULT.json',result)
    (R/'RESULT.md').write_text('# 三维Mumford–Shah单组训练状态\n\n'+f"实际完成{log['epoch']}/200轮，共{log['cumulative_optimizer_updates']}次参数更新；每轮完整972扫描。固定复用108扫描/50患者验证，本轮best5 Scan {result['scan_best5']:.6f}、Subject {result['subject_best5']:.6f}，Scan患者整簇95%区间{result['scan_best5_ci95']}；final5 Scan {result['scan_final5']:.6f}。选优最佳第{log['selection']['epoch']}轮best5 Scan {log['selection']['scan_best5']:.6f}。\n\n"+'目标是逐体素区域亮度一致性和边界平滑；当前为单组、复用验证选优，不证明独立因果或外部泛化。与旧灰度相比同时更换loss和恢复累积；二维作者实现已作三维、强度及归约适配。方法选择发生在旧test揭示之后，本轮无新增测试/外层/确认。\n\n'+f"双A100每卡微批4，全局8，每轮504+468两个样本加权更新。累计执行约{log['elapsed_seconds']/3600:.3f}小时，每卡峰值{log['peak_allocated_bytes']/2**30:.3f}GiB。复用原缓存；新best/latest约1.5GiB，仅保留滚动latest与best，部署仍一次编码+五次交互解码，无灰度额外前向。\n\n"+'[机器指标与哈希](RESULT.json) · [固定七问](IDEA.md) · [运行心跳](HEARTBEAT.json) · [数据审计](DATA_AUDIT.json)。\n')
    return result


def finalize(logs):
    assert [r['epoch'] for r in logs]==list(range(1,201))
    assert all(r['samples']==972 and r['optimizer_updates']==2 for r in logs)
    saved=json.loads((OUT/'STATUS.json').read_text());assert saved['status']=='completed' and saved['cumulative_optimizer_updates']==400
    best=json.loads((OUT/'BEST.json').read_text());assert sha(OUT/'best.pth')==best['checkpoint_sha256']
    raw=json.loads((OUT/best['report']).read_text());assert sha(OUT/best['report'])==best['report_sha256']
    result=stats(raw['rows']);result.update(status='completed200',completed_epochs=200,cumulative_optimizer_updates=400,selected_epoch=best['epoch'],old_validation_best5_reference=.5713061351995669,validation_point_gate_passed=result['scan_best5']>.5713061351995669,test_pixels_access=False,prior_test_informed_method_choice=True,outer_confirmation_access=False,training_seconds=sum(r['training_seconds'] for r in logs),elapsed_seconds=saved['elapsed_seconds'],effective_allocated_gpu_hours=2*saved['elapsed_seconds']/3600,peak_allocated_bytes_per_gpu=max(r['peak_allocated_bytes'] for r in logs),extra_cost_note='smoke excluded; replay included in training wall time; allocation-hours include waiting during rank0 validation',evidence={str(p.relative_to(ROOT)):sha(p) for p in [R/'FREEZE.json',R/'TRAIN_PREACCESS.json',R/'VALIDATION_PREACCESS.json',R/'DATA_AUDIT.json',OUT/'best.pth',OUT/'latest.pth',OUT/best['report'],OUT/'TRAIN.jsonl']})
    write(R/'FINAL_RESULT.json',result);write(R/'RAW_SELECTED_VALIDATION.json',raw)
    with (R/'RESULT.md').open('a') as f:f.write(f"\n200轮已完成并通过全972/400更新核对；最终选第{best['epoch']}轮，best5 Scan {result['scan_best5']:.6f}，95%CI {result['scan_best5_ci95']}，验证点估计门{'通过' if result['validation_point_gate_passed'] else '未通过'}。无自动test准入。[最终结果](FINAL_RESULT.json)。\n")
    return best


def main():
    lock=(R/'SUPERVISOR.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
    assert json.loads((R/'GPU_SMOKE.json').read_text())['status']=='passed'
    logpath=R/'train.log';handle=logpath.open('a');proc=None;last=0
    try:
        proc=subprocess.Popen([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','scripts/train_sammed3d_mumford_shah_972.py','train'],cwd=ROOT,env=ENV,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        write(R/'LAUNCH.json',{'utc':now(),'pid':proc.pid,'supervisor_pid':os.getpid(),'worker':'1098698'})
        write(R/'HEARTBEAT.json',{'utc':now(),'alive':True,'status':'starting','last_complete_epoch':0})
        documents('已启动，等待首轮');checks()
        while proc.poll() is None:
            rows=[json.loads(x) for x in (OUT/'TRAIN.jsonl').read_text().splitlines()] if (OUT/'TRAIN.jsonl').exists() else []
            if rows and rows[-1]['epoch']!=last:
                for row in rows:
                    if row['epoch']>last:archive(row);last=row['epoch']
                documents('训练中',last,rows[-1]['selection']);checks()
            write(R/'HEARTBEAT.json',{'utc':now(),'alive':True,'status':'running','pid':proc.pid,'last_complete_epoch':last,'optimizer_updates':last*2,'latest_validation':rows[-1].get('validation') if rows else None,'best':rows[-1].get('selection') if rows else None,'test_pixels_access':False})
            if time.time()-logpath.stat().st_mtime>1200:raise RuntimeError('training log stalled >20min')
            time.sleep(15)
        if proc.returncode:raise RuntimeError(f'training exited {proc.returncode}')
        rows=[json.loads(x) for x in (OUT/'TRAIN.jsonl').read_text().splitlines()]
        archive(rows[-1],terminal=True);best=finalize(rows);documents('200轮完成',200,best);checks()
        write(R/'HEARTBEAT.json',{'utc':now(),'alive':False,'status':'completed','last_complete_epoch':200,'optimizer_updates':400,'best':best,'test_pixels_access':False})
        p=ROOT/'AFLOW_BRATS_WEEKLY_REPORT.md';p.write_text(p.read_text()+'\nMumford–Shah新单组200轮完成，最优验证best5='+str(best['scan_best5'])+'，无新增test。[最终结果]('+REL+'/FINAL_RESULT.json)。\n');checks()
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        if (OUT/'STATUS.json').exists():last=json.loads((OUT/'STATUS.json').read_text())['last_complete_epoch']
        failure={'utc':now(),'alive':False,'status':'failed','last_complete_epoch':last,'error':repr(exc),'traceback':traceback.format_exc()}
        write(R/'FAILURE.json',failure);write(R/'HEARTBEAT.json',failure);documents('执行中断，未完成200轮',last)
        with (R/'RESULT.md').open('a') as f:f.write('\n执行中断：'+repr(exc)+'；实际完成'+str(last)+'轮，不称200轮完成。\n')
        raise
    finally:handle.close()

if __name__=='__main__':main()
