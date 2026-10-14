"""Persistent supervision, validation receipts, final pure-baseline report."""
import sys,os,json,time,subprocess,fcntl,signal,re,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.segresnet_plain972_no_n4_common import *
PY=str(ROOT.parent/'sammed3d-venv/bin/python');REL=str(R.relative_to(ROOT));TITLE='## SegResNet固定972纯基线｜'
ENV={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','NCCL_DEBUG':'WARN'}

def checks():
    rows=[]
    for name in ['verify_aflow_brats_doc_layout.py','verify_aflow_brats_method_catalog.py','verify_aflow_brats_leaderboard.py','verify_aflow_root_artifacts.py']:
        p=subprocess.run([PY,'scripts/'+name],cwd=ROOT,env=ENV,capture_output=True,text=True);rows.append({'name':name,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    write(R/'VALIDATORS.json',rows)
    if any(r['returncode'] for r in rows):raise RuntimeError('documentation checks failed')

def documents(stage,epoch=0,best=None):
    brief=f'无额外N4单T1ce纯SegResNet：{stage}，完成{epoch}/200轮；每轮原972各一次、122次更新。'
    if best:brief+=f" 最佳验证第{best['epoch']}轮Scan {best['scan_dice']:.6f}。"
    brief+=' 原108验证选优，270测试不访问；唯一研究变化为取消额外N4，不预设当前0.759179为创新收益。'
    blocks={
      'AFLOW_BRATS_HANDOFF_CURRENT.md':brief+' [进度]('+REL+'/RESULT.md) · [实时JSON]('+REL+'/RESULT.json)',
      'AFLOW_BRATS_METHOD_CATALOG.md':'## 无额外N4纯SegResNet基础对照｜'+stage+'\n\n**一句话：** 去掉额外强度校正，检查现有方案是否真的从中获益。 **出发点：** 缺少相同972配方的无N4对照。\n**具体做法：** 只取消N4，保留单T1ce、脑区裁剪/标准化128³、同初始化和200轮训练，每轮972各一次。\n**结果：** '+brief+'\n**是否支持直觉：** '+('以完整200轮和配对区间判断，阶段分数不是最终贡献。' if epoch<200 else '见结果中的配对差值；开发验证不等同独立测试，N4是已有方法。')+'\n**代价/下一步：** 双A100，部署1次前向；缓存放临时盘、模型报告持久化，不自动测试。[核对材料]('+REL+'/RESULT.md)',
      'AFLOW_BRATS_RESULTS.md':'## 2026-09-18｜用户授权无额外N4基础对照\n\n问题：需要更基础的单T1ce对照。做法：取消N4，其余训练和数据固定。结果：'+brief+' 决定：如实接受任何结果，不弱化对照、不按test调参。[材料]('+REL+'/RESULT.md)。'
    }
    for name,body in blocks.items():
        p=ROOT/name;s=p.read_text();block='<!-- NON4_972_STATUS -->\n'+body+'\n<!-- END_NON4_972_STATUS -->'
        if '<!-- NON4_972_STATUS -->' in s:s=re.sub(r'<!-- NON4_972_STATUS -->.*?<!-- END_NON4_972_STATUS -->',lambda m:block,s,flags=re.S)
        else:s+='\n'+block+'\n'
        p.write_text(s)


def archive(log):
    report=OUT/f"validation_e{log['epoch']:03d}.json";raw=json.loads(report.read_text());assert sha(report)==log['validation_report_sha256']
    result=summary_ci(raw['per_case']);result.update(status='running',epoch=log['epoch'],completed_epochs=log['epoch'],optimizer_updates=log['cumulative_optimizer_updates'],selected_checkpoint=log['selection'],training=log['train'],augmentation_fallbacks_this_epoch=log['augmentation_fallbacks'],elapsed_seconds=log['elapsed_seconds'],training_seconds_this_epoch=log['training_seconds'],peak_allocated_bytes_per_gpu=log['peak_allocated_bytes_per_gpu'],new_test_pixels_access=False,new_outer_confirmation_access=False,prior_test_informed_development=True,raw_report=str(report.relative_to(ROOT)),raw_report_sha256=sha(report),checkpoint_sha256=log['checkpoint_sha256'],freeze_sha256=sha(R/'FREEZE.json'),preaccess_sha256=sha(R/'TRAIN_PREACCESS.json'),validation_preaccess_sha256=sha(R/'VALIDATION_PREACCESS.json'),data_audit_sha256=sha(R/'DATA_AUDIT.json'),cache_index_sha256={x['role']:sha(ROOT/x['path']/'index.json') for x in json.loads((R/'DATA_AUDIT.json').read_text())['caches']})
    write(R/f"VALIDATION_E{log['epoch']:03d}.json",result);write(R/'RESULT.json',result)
    (R/'RESULT.md').write_text('# 无额外N4原972纯SegResNet进度\n\n'+f"已完成{log['epoch']}/200轮、{log['cumulative_optimizer_updates']}次更新，每轮完整972扫描无重复/丢尾。固定复用108扫描/50患者验证，本轮Scan Dice {result['scan_dice']:.6f}、Subject {result['subject_dice']:.6f}、Point hit {result['point_hit']:.6f}；Scan患者整簇95%区间{result['scan_ci95']}。最佳验证第{log['selection']['epoch']}轮Scan {log['selection']['scan_dice']:.6f}。\n\n"+'模型是470万参数纯SegResNet，随机初始化，无额外N4，保留图像自身脑区裁剪和标准化128³，Dice+0.1交叉熵。无蒸馏、灰度项、额外中心头、融合、镜像增强或后处理。全自动一次前向，指标叫scan_dice，不使用SAM五点击指标名。固定病例划分相同，但与SAM的GT中心裁剪/交互权限和栅格不同，不能当纯架构消融。\n\n'+f"双A100每卡4例/global8，无累积，每轮122更新；前100轮lr2e-4，后100轮余弦降到1e-5。本轮训练{log['training_seconds']:.2f}秒，每卡峰值{log['peak_allocated_bytes_per_gpu']/2**30:.3f}GiB。重新从原始T1ce构建无额外N4的train/validation缓存于临时盘，best约18MiB/latest约54MiB，另存每10轮轻量快照。\n\n"+'唯一研究变化为取消额外N4；对照为原N4纯193的同108验证0.740032，不能与test0.759179跨队列相减。本组没有test/outer/confirmation像素评估，研究方向已受旧测试揭示影响，不称盲测。仅复用验证选优，CI是描述性区间。基线允许负结果，不根据测试调参或自动测试。\n\n[机器指标与哈希](RESULT.json) · [固定七问](IDEA.md) · [心跳](HEARTBEAT.json) · [数据审计](DATA_AUDIT.json)。\n')
    return result


def finalize(logs):
    assert [r['epoch'] for r in logs]==list(range(1,201)) and all(r['samples']==972 and r['unique_scans']==972 and r['optimizer_updates']==122 for r in logs)
    state=json.loads((OUT/'STATUS.json').read_text());assert state['status']=='completed' and state['optimizer_updates']==24400
    assert json.loads((R/'RELOAD_CHECK.json').read_text())['status']=='passed'
    best=json.loads((OUT/'BEST.json').read_text());report=OUT/best['report'];assert sha(report)==best['report_sha256'] and sha(OUT/'best.pth')==best['checkpoint_sha256']
    raw=json.loads(report.read_text());d=summary_ci(raw['per_case'])
    d.update(status='completed200',completed_epochs=200,cumulative_optimizer_updates=24400,selected_epoch=best['epoch'],new_test_pixels_access=False,new_outer_confirmation_access=False,prior_test_informed_development=True,training_seconds=sum(r['training_seconds'] for r in logs),elapsed_seconds=state['elapsed_seconds'],allocated_gpu_hours=2*state['elapsed_seconds']/3600,peak_allocated_bytes_per_gpu=max(r['peak_allocated_bytes_per_gpu'] for r in logs),augmentation_fallbacks=sum(r['augmentation_fallbacks'] for r in logs),checkpoint_storage_bytes=sum(p.stat().st_size for p in OUT.glob('*.pth')),evidence={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'best.pth',OUT/'latest.pth',OUT/'initial.pth',report,OUT/'TRAIN.jsonl',R/'FREEZE.json',R/'TRAIN_PREACCESS.json',R/'VALIDATION_PREACCESS.json',R/'DATA_AUDIT.json',R/'RELOAD_CHECK.json']})
    from scripts.evaluate_plain972_e193_flip8_validation import paired
    baseline=json.loads((ROOT/'workspace/BraTS/segresnet_t1ce_wt/plain972_baseline_20260917/validation_e193.json').read_text())
    d['comparison_noN4_minus_N4']=paired(baseline['per_case'],raw['per_case'])
    d['comparison_scope']='same reused validation108; each model selected on this validation; not independent confirmation'
    d['extra_N4']=False
    write(R/'FINAL_RESULT.json',d);write(R/'RESULT.json',d);write(R/'RAW_SELECTED_VALIDATION.json',raw)
    with (R/'RESULT.md').open('a') as f:f.write(f"\n200轮完成，24400更新，best/latest严格重载一致性通过。选第{best['epoch']}轮，Scan/Subject={d['scan_dice']:.6f}/{d['subject_dice']:.6f}，Scan95%CI={d['scan_ci95']}；未运行test。[最终报告](FINAL_RESULT.json)。\n")
    with (R/'RESULT.md').open('a') as f:f.write('\n无N4−N4的验证配对差值：'+json.dumps(d['comparison_noN4_minus_N4'],ensure_ascii=False)+'。相同复用验证上的选优比较，非独立确认。\n')
    return best


def main():
    lock=(R/'SUPERVISOR.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
    assert json.loads((R/'GPU_SMOKE.json').read_text())['status']=='passed'
    logpath=R/'train.log';handle=logpath.open('a');proc=None;last=0
    try:
        proc=subprocess.Popen([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','scripts/train_segresnet_plain972_no_n4.py','train'],cwd=ROOT,env=ENV,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        write(R/'LAUNCH.json',{'utc':now(),'pid':proc.pid,'supervisor_pid':os.getpid(),'worker':'1100631'});write(R/'HEARTBEAT.json',{'utc':now(),'alive':True,'status':'starting','last_complete_epoch':0})
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
        p=ROOT/'AFLOW_BRATS_WEEKLY_REPORT.md';p.write_text(p.read_text()+'\n无额外N4原972纯SegResNet对照200轮完成，最佳验证Scan='+str(best['scan_dice'])+'，无新增test。[最终结果]('+REL+'/FINAL_RESULT.json)。\n');checks()
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
