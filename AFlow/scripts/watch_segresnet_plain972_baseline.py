"""Persistent supervision, validation receipts, final pure-baseline report."""
import sys,os,json,time,subprocess,fcntl,signal,re,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.segresnet_plain972_common import *
PY=str(ROOT.parent/'sammed3d-venv/bin/python');REL=str(R.relative_to(ROOT));TITLE='## SegResNet固定972纯基线｜'
ENV={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','NCCL_DEBUG':'WARN'}

def checks():
    rows=[]
    for name in ['verify_aflow_brats_doc_layout.py','verify_aflow_brats_method_catalog.py','verify_aflow_brats_leaderboard.py','verify_aflow_root_artifacts.py']:
        p=subprocess.run([PY,'scripts/'+name],cwd=ROOT,env=ENV,capture_output=True,text=True);rows.append({'name':name,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    write(R/'VALIDATORS.json',rows)
    if any(r['returncode'] for r in rows):raise RuntimeError('documentation checks failed')

def documents(stage,epoch=0,best=None):
    brief=f'固定972训练/108验证/270测试的纯SegResNet新组：{stage}，实际完成{epoch}/200轮。每轮972各一次、122次更新，双A100每卡4例。'
    if best:brief+=f" 最佳验证第{best['epoch']}轮Scan Dice {best['scan_dice']:.6f}、Subject {best['subject_dice']:.6f}。"
    brief+=' 无蒸馏/灰度/融合/镜像后处理；全自动一次前向，无GT定位/点击；本组未访问test/outer/confirmation像素，旧测试揭示历史保留。'
    p=ROOT/'AFLOW_BRATS_HANDOFF_CURRENT.md';s=p.read_text();block='<!-- SEGRESNET_PLAIN972_STATUS -->\n'+brief+' [结果/进度]('+REL+'/RESULT.md) · [心跳]('+REL+'/HEARTBEAT.json)\n<!-- END_SEGRESNET_PLAIN972_STATUS -->'
    if '<!-- SEGRESNET_PLAIN972_STATUS -->' in s:s=re.sub(r'<!-- SEGRESNET_PLAIN972_STATUS -->.*?<!-- END_SEGRESNET_PLAIN972_STATUS -->',lambda m:block,s,flags=re.S)
    else:s=s.replace('## 当前状态','## 当前状态\n\n'+block,1)
    p.write_text(s)
    p=ROOT/'AFLOW_BRATS_METHOD_CATALOG.md';s=p.read_text();a=s.find(TITLE);b=s.find('\n## ',a+3) if a>=0 else -1
    card=TITLE+stage+'\n\n**一句话：** 用小型三维残差网络直接给出肿瘤分割，补齐当前972训练的纯baseline。 **出发点：** 旧0.725820/0.741204纯模型用了756训练，972成绩含蒸馏。\n**具体做法：** 单T1ce、图像自身裁脑128³，随机初始化纯SegResNet；原Dice+0.1交叉熵及200轮时程，972无放回且不丢尾，每轮108验证选优。\n**结果：** '+brief+'\n**是否支持直觉：** '+('已有阶段验证，是否适合作自动基线以完整结果和使用权限判断，不作架构因果结论。' if epoch else '尚无新精度结果。')+'\n**代价/下一步：** 两卡单组，470万参数、部署1次前向；复用缓存，不自动进test。按固定200轮完成并归档。[核对材料]('+REL+'/RESULT.md)\n'
    p.write_text(s[:a]+card+(s[b:] if b>=0 else '') if a>=0 else s+'\n'+card)
    p=ROOT/'AFLOW_BRATS_RESULTS.md';s=p.read_text();block='<!-- SEGRESNET_PLAIN972_LOG -->\n## 2026-09-17｜固定972纯SegResNet基线\n\n**问题：** 用户需要选择SAM或SegResNet，但纯Seg历史成绩来自756训练。 **做法：** 新单组、原972/108/270、纯网络完整200轮，逐轮验证。 **结果：** '+brief+' **决定：** 如实记录真实轮数和权限/栅格差异，不自动测test。[材料]('+REL+'/RESULT.md)。\n<!-- END_SEGRESNET_PLAIN972_LOG -->'
    if '<!-- SEGRESNET_PLAIN972_LOG -->' in s:s=re.sub(r'<!-- SEGRESNET_PLAIN972_LOG -->.*?<!-- END_SEGRESNET_PLAIN972_LOG -->',lambda m:block,s,flags=re.S)
    else:s+='\n'+block+'\n'
    p.write_text(s)


def archive(log):
    report=OUT/f"validation_e{log['epoch']:03d}.json";raw=json.loads(report.read_text());assert sha(report)==log['validation_report_sha256']
    result=summary_ci(raw['per_case']);result.update(status='running',epoch=log['epoch'],completed_epochs=log['epoch'],optimizer_updates=log['cumulative_optimizer_updates'],selected_checkpoint=log['selection'],training=log['train'],augmentation_fallbacks_this_epoch=log['augmentation_fallbacks'],elapsed_seconds=log['elapsed_seconds'],training_seconds_this_epoch=log['training_seconds'],peak_allocated_bytes_per_gpu=log['peak_allocated_bytes_per_gpu'],new_test_pixels_access=False,new_outer_confirmation_access=False,prior_test_informed_development=True,raw_report=str(report.relative_to(ROOT)),raw_report_sha256=sha(report),checkpoint_sha256=log['checkpoint_sha256'],freeze_sha256=sha(R/'FREEZE.json'),preaccess_sha256=sha(R/'TRAIN_PREACCESS.json'),validation_preaccess_sha256=sha(R/'VALIDATION_PREACCESS.json'),data_audit_sha256=sha(R/'DATA_AUDIT.json'),cache_index_sha256={x['role']:sha(ROOT/x['path']/'index.json') for x in json.loads((R/'DATA_AUDIT.json').read_text())['caches']})
    write(R/f"VALIDATION_E{log['epoch']:03d}.json",result);write(R/'RESULT.json',result)
    (R/'RESULT.md').write_text('# 原972纯SegResNet基线进度\n\n'+f"已完成{log['epoch']}/200轮、{log['cumulative_optimizer_updates']}次更新，每轮完整972扫描无重复/丢尾。固定复用108扫描/50患者验证，本轮Scan Dice {result['scan_dice']:.6f}、Subject {result['subject_dice']:.6f}、Point hit {result['point_hit']:.6f}；Scan患者整簇95%区间{result['scan_ci95']}。最佳验证第{log['selection']['epoch']}轮Scan {log['selection']['scan_dice']:.6f}。\n\n"+'模型是470万参数纯SegResNet，随机初始化，图像自身脑区裁剪128³，Dice+0.1交叉熵。无蒸馏、灰度项、额外中心头、融合、镜像增强或后处理。全自动一次前向，指标叫scan_dice，不使用SAM五点击指标名。固定病例划分相同，但与SAM的GT中心裁剪/交互权限和栅格不同，不能当纯架构消融。\n\n'+f"双A100每卡4例/global8，无累积，每轮122更新；前100轮lr2e-4，后100轮余弦降到1e-5。本轮训练{log['training_seconds']:.2f}秒，每卡峰值{log['peak_allocated_bytes_per_gpu']/2**30:.3f}GiB。复用缓存，best约18MiB/latest约54MiB，另存每10轮轻量快照。\n\n"+'本组没有test/outer/confirmation像素评估，研究方向已受旧测试揭示影响，不称盲测。仅复用验证选优，CI是描述性区间。基线允许负结果，不根据测试调参或自动测试。\n\n[机器指标与哈希](RESULT.json) · [固定七问](IDEA.md) · [心跳](HEARTBEAT.json) · [数据审计](DATA_AUDIT.json)。\n')
    return result


def finalize(logs):
    assert [r['epoch'] for r in logs]==list(range(1,201)) and all(r['samples']==972 and r['unique_scans']==972 and r['optimizer_updates']==122 for r in logs)
    state=json.loads((OUT/'STATUS.json').read_text());assert state['status']=='completed' and state['optimizer_updates']==24400
    assert json.loads((R/'RELOAD_CHECK.json').read_text())['status']=='passed'
    best=json.loads((OUT/'BEST.json').read_text());report=OUT/best['report'];assert sha(report)==best['report_sha256'] and sha(OUT/'best.pth')==best['checkpoint_sha256']
    raw=json.loads(report.read_text());d=summary_ci(raw['per_case'])
    d.update(status='completed200',completed_epochs=200,cumulative_optimizer_updates=24400,selected_epoch=best['epoch'],new_test_pixels_access=False,new_outer_confirmation_access=False,prior_test_informed_development=True,training_seconds=sum(r['training_seconds'] for r in logs),elapsed_seconds=state['elapsed_seconds'],allocated_gpu_hours=2*state['elapsed_seconds']/3600,peak_allocated_bytes_per_gpu=max(r['peak_allocated_bytes_per_gpu'] for r in logs),augmentation_fallbacks=sum(r['augmentation_fallbacks'] for r in logs),checkpoint_storage_bytes=sum(p.stat().st_size for p in OUT.glob('*.pth')),evidence={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'best.pth',OUT/'latest.pth',OUT/'initial.pth',report,OUT/'TRAIN.jsonl',R/'FREEZE.json',R/'TRAIN_PREACCESS.json',R/'VALIDATION_PREACCESS.json',R/'DATA_AUDIT.json',R/'RELOAD_CHECK.json']})
    write(R/'FINAL_RESULT.json',d);write(R/'RAW_SELECTED_VALIDATION.json',raw)
    with (R/'RESULT.md').open('a') as f:f.write(f"\n200轮完成，24400更新，best/latest严格重载一致性通过。选第{best['epoch']}轮，Scan/Subject={d['scan_dice']:.6f}/{d['subject_dice']:.6f}，Scan95%CI={d['scan_ci95']}；未运行test。[最终报告](FINAL_RESULT.json)。\n")
    return best


def main():
    lock=(R/'SUPERVISOR.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
    assert json.loads((R/'GPU_SMOKE.json').read_text())['status']=='passed'
    logpath=R/'train.log';handle=logpath.open('a');proc=None;last=0
    try:
        proc=subprocess.Popen([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','scripts/train_segresnet_plain972_baseline.py','train'],cwd=ROOT,env=ENV,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        write(R/'LAUNCH.json',{'utc':now(),'pid':proc.pid,'supervisor_pid':os.getpid(),'worker':'1098698'});write(R/'HEARTBEAT.json',{'utc':now(),'alive':True,'status':'starting','last_complete_epoch':0})
        documents('已启动，等待首轮');checks()
        while proc.poll() is None:
            rows=[json.loads(x) for x in (OUT/'TRAIN.jsonl').read_text().splitlines()] if (OUT/'TRAIN.jsonl').exists() else []
            if rows and rows[-1]['epoch']!=last:
                for row in rows:
                    if row['epoch']>last:archive(row);last=row['epoch']
                documents('训练中',last,rows[-1]['selection']);checks()
            write(R/'HEARTBEAT.json',{'utc':now(),'alive':True,'status':'running','pid':proc.pid,'last_complete_epoch':last,'optimizer_updates':last*122,'latest_validation':rows[-1].get('validation') if rows else None,'best':rows[-1].get('selection') if rows else None,'test_pixels_access':False})
            if time.time()-logpath.stat().st_mtime>1200:raise RuntimeError('training log stalled >20min')
            time.sleep(15)
        if proc.returncode:raise RuntimeError(f'training exited {proc.returncode}')
        rows=[json.loads(x) for x in (OUT/'TRAIN.jsonl').read_text().splitlines()];archive(rows[-1]);best=finalize(rows);documents('200轮完成',200,best);checks()
        write(R/'HEARTBEAT.json',{'utc':now(),'alive':False,'status':'completed','last_complete_epoch':200,'optimizer_updates':24400,'best':best,'test_pixels_access':False})
        p=ROOT/'AFLOW_BRATS_WEEKLY_REPORT.md';p.write_text(p.read_text()+'\n原972纯SegResNet基线200轮完成，最佳验证Scan='+str(best['scan_dice'])+'，无新增test。[最终结果]('+REL+'/FINAL_RESULT.json)。\n');checks()
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        if (OUT/'STATUS.json').exists():last=json.loads((OUT/'STATUS.json').read_text())['last_complete_epoch']
        failure={'utc':now(),'alive':False,'status':'failed','last_complete_epoch':last,'error':repr(exc),'traceback':traceback.format_exc()};write(R/'FAILURE.json',failure);write(R/'HEARTBEAT.json',failure);documents('执行中断，未完成200轮',last)
        with (R/'RESULT.md').open('a') as f:f.write('\n执行中断：'+repr(exc)+'，实际完成'+str(last)+'轮，不能称200轮完成。\n')
        raise
    finally:handle.close()

if __name__=='__main__':main()
