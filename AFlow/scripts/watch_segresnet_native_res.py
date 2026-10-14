"""Persistent supervision for the native-resolution run: launch/resume torchrun, per-epoch receipts, docs, final report."""
import sys,os,json,time,subprocess,fcntl,signal,re,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.segresnet_native_res_common import *
PY=str(ROOT.parent/'sammed3d-venv/bin/python');REL=str(R.relative_to(ROOT));MARK='NATIVE_RES_972_STATUS';MAX_ATTEMPTS=4
ENV={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','NCCL_DEBUG':'WARN'}


def checks():
    rows=[]
    for name in ['verify_aflow_brats_doc_layout.py','verify_aflow_brats_method_catalog.py','verify_aflow_brats_leaderboard.py','verify_aflow_root_artifacts.py']:
        p=subprocess.run([PY,'scripts/'+name],cwd=ROOT,env=ENV,capture_output=True,text=True);rows.append({'name':name,'returncode':p.returncode,'stdout':p.stdout[-2000:],'stderr':p.stderr[-2000:]})
    write(R/'VALIDATORS.json',rows)
    if any(r['returncode'] for r in rows):raise RuntimeError('documentation checks failed')


def baseline_row():
    b=json.loads((R/'BASELINE_E193_NATIVE_VALIDATION.json').read_text())['native_trilinear_logits'];return b


def documents(stage,epoch=0,best=None,final=None):
    per_gpu=batch_decision();steps=steps_per_epoch(per_gpu);b=baseline_row()
    brief=f'原始分辨率整脑纯SegResNet（不缩放到128³）：{stage}，完成{epoch}/200轮；每轮原972各一次，每卡{per_gpu}例、全局{2*per_gpu}、每轮{steps}次更新。'
    if best:brief+=f" 最佳验证第{best['epoch']}轮原始栅格Scan {best['scan_dice']:.6f}。"
    brief+=f" 同栅格对照：原N4纯193重采样到原始裁剪栅格Scan {b['scan_dice']:.6f}、Subject {b['subject_dice']:.6f}、Point {b['point_hit']:.6f}（其128³栅格自评0.740032不可直接相减）。"
    if final:brief+=f" 最终配对差值（新−基线，同108原始栅格）Scan {final['scan_delta']:+.6f}、95%CI {final['scan_delta_ci95']}、Point {final['point_hits_delta']:+d}。"
    brief+=' 原108验证选优，270测试不访问；唯一研究变化为取消128³缩放、保持1mm并零填充到160×208×160画布。'
    blocks={
      'AFLOW_BRATS_HANDOFF_CURRENT.md':brief+' [进度]('+REL+'/RESULT.md) · [实时JSON]('+REL+'/RESULT.json)',
      'AFLOW_BRATS_METHOD_CATALOG.md':'## 原始分辨率整脑SegResNet（不缩放到128³）｜'+stage+'\n\n**一句话：** 不再把脑区压缩到128³，让网络在1mm原始体素上看图。 **出发点：** 脑区裁剪框中位数136×175×142，缩到128³是各轴不等比缩放，小病灶和边界细节被抹平；历史160³失败仍是缩放，不是原始分辨率。\n**具体做法：** 同N4、同图像自身脑区裁剪、同标准化，但不缩放，零填充到固定160×208×160画布；同470万参数SegResNet、同初始化seed、Dice+0.1交叉熵、同200轮学习率计划；验证/部署仍1次前向，指标在原始裁剪栅格上算、排除填充。每卡batch由合成显存探测在8/6/4中取最大可装下者。\n**结果：** '+brief+'\n**是否支持直觉：** '+('以完整200轮和同栅格配对区间判断，阶段分数不是最终结论。' if final is None else ('配对区间不跨0且方向为正才称支持；' if final['scan_delta_ci95'][0]>0 else '差值区间跨0或为负，本实现未证明原始分辨率带来确定增益；')+'开发验证不等于独立测试。')+'\n**代价/下一步：** 输入体素量2.54倍，训练与推理算力同比增加；缓存放临时盘、模型报告持久化；不自动测试，进入test需用户新授权。[核对材料]('+REL+'/RESULT.md)',
      'AFLOW_BRATS_RESULTS.md':'## 2026-09-18｜原始分辨率整脑训练（用户选定的改进方向1）\n\n问题：现有流程各轴不等比缩放到128³，可能丢失空间细节；验证7/108例Dice低于0.5。做法：保持1mm原始裁剪并零填充到固定画布，其余训练配方与原972纯基线一致；基线模型重采样到同一原始栅格作对照。结果：'+brief+' 决定：如实接受任何结果，不按test调参、不自动测试。[材料]('+REL+'/RESULT.md)。'
    }
    for name,body in blocks.items():
        p=ROOT/name;s=p.read_text();block=f'<!-- {MARK} -->\n'+body+f'\n<!-- END_{MARK} -->'
        if f'<!-- {MARK} -->' in s:s=re.sub(f'<!-- {MARK} -->.*?<!-- END_{MARK} -->',lambda m:block,s,flags=re.S)
        else:s+='\n'+block+'\n'
        p.write_text(s)


def archive(log):
    report=OUT/f"validation_e{log['epoch']:03d}.json";raw=json.loads(report.read_text());assert sha(report)==log['validation_report_sha256']
    result=summary_ci(raw['per_case']);b=baseline_row();base=json.loads((R/'BASELINE_E193_NATIVE_VALIDATION.json').read_text())['per_case']
    result.update(status='running',epoch=log['epoch'],completed_epochs=log['epoch'],optimizer_updates=log['cumulative_optimizer_updates'],per_gpu_batch=log['per_gpu_batch'],selected_checkpoint=log['selection'],training=log['train'],augmentation_fallbacks_this_epoch=log['augmentation_fallbacks'],elapsed_seconds=log['elapsed_seconds'],training_seconds_this_epoch=log['training_seconds'],validation_seconds_this_epoch=log['validation_seconds'],peak_allocated_bytes_per_gpu=log['peak_allocated_bytes_per_gpu'],peak_reserved_bytes_per_gpu=log['peak_reserved_bytes_per_gpu'],grid='native 1mm crop; padding excluded',baseline_same_grid=b,paired_this_epoch_minus_baseline=paired(base,raw['per_case']),new_test_pixels_access=False,new_outer_confirmation_access=False,prior_test_informed_development=True,raw_report=str(report.relative_to(ROOT)),raw_report_sha256=sha(report),checkpoint_sha256=log['checkpoint_sha256'],freeze_sha256=sha(R/'FREEZE.json'),preaccess_sha256=sha(R/'TRAIN_PREACCESS.json'),validation_preaccess_sha256=sha(R/'VALIDATION_PREACCESS.json'),data_audit_sha256=sha(R/'DATA_AUDIT.json'))
    write(R/f"VALIDATION_E{log['epoch']:03d}.json",result);write(R/'RESULT.json',result)
    (R/'RESULT.md').write_text('# 原始分辨率整脑纯SegResNet进度\n\n'+f"已完成{log['epoch']}/200轮、{log['cumulative_optimizer_updates']}次更新，每轮完整972扫描无重复/丢尾，每卡{log['per_gpu_batch']}例。固定复用108扫描/50患者验证，在1mm原始裁剪栅格上（排除画布填充）本轮Scan Dice {result['scan_dice']:.6f}、Subject {result['subject_dice']:.6f}、Point hit {result['point_hit']:.6f}；Scan患者整簇95%区间{result['scan_ci95']}。最佳验证第{log['selection']['epoch']}轮Scan {log['selection']['scan_dice']:.6f}。\n\n"+f"同栅格对照：原N4纯193把128³预测三线性重采样回原始裁剪栅格，Scan {b['scan_dice']:.6f}、Subject {b['subject_dice']:.6f}、Point {b['point_hit']:.6f}。本轮配对差值（新−基线）{json.dumps(result['paired_this_epoch_minus_baseline'],ensure_ascii=False)}。\n\n"+'模型是470万参数纯SegResNet，随机初始化（同基线seed），N4、图像自身脑区裁剪、非零z-score与基线一致；唯一改动是不缩放到128³，而是零填充到固定160×208×160画布，训练与部署都是1次整脑前向。Dice+0.1交叉熵；无蒸馏、灰度项、EMA、融合、镜像增强或后处理。\n\n'+f"双A100每卡{log['per_gpu_batch']}例；前100轮lr2e-4，后100轮余弦降到1e-5。本轮训练{log['training_seconds']:.1f}秒、验证{log['validation_seconds']:.1f}秒，每卡峰值分配{log['peak_allocated_bytes_per_gpu']/2**30:.2f}GiB、保留{log['peak_reserved_bytes_per_gpu']/2**30:.2f}GiB。\n\n"+'本组没有test/outer/confirmation像素评估，研究方向已受旧测试揭示影响，不称盲测。仅复用验证选优，CI是描述性区间；不根据测试调参或自动测试。\n\n[机器指标与哈希](RESULT.json) · [固定七问](IDEA.md) · [心跳](HEARTBEAT.json) · [数据审计](DATA_AUDIT.json) · [batch决定](BATCH_DECISION.json) · [基线同栅格对照](BASELINE_E193_NATIVE_VALIDATION.json)\n')
    return result


def finalize(logs):
    per_gpu=batch_decision();steps=steps_per_epoch(per_gpu)
    assert [r['epoch'] for r in logs]==list(range(1,201)) and all(r['samples']==972 and r['unique_scans']==972 and r['optimizer_updates']==steps for r in logs)
    state=json.loads((OUT/'STATUS.json').read_text());assert state['status']=='completed' and state['optimizer_updates']==200*steps
    assert json.loads((R/'RELOAD_CHECK.json').read_text())['status']=='passed'
    best=json.loads((OUT/'BEST.json').read_text());report=OUT/best['report'];assert sha(report)==best['report_sha256'] and sha(OUT/'best.pth')==best['checkpoint_sha256']
    raw=json.loads(report.read_text());d=summary_ci(raw['per_case']);bfile=json.loads((R/'BASELINE_E193_NATIVE_VALIDATION.json').read_text())
    d.update(status='completed200',completed_epochs=200,cumulative_optimizer_updates=200*steps,per_gpu_batch=per_gpu,selected_epoch=best['epoch'],grid='native 1mm crop; padding excluded',new_test_pixels_access=False,new_outer_confirmation_access=False,prior_test_informed_development=True,training_seconds=sum(r['training_seconds'] for r in logs),validation_seconds=sum(r['validation_seconds'] for r in logs),elapsed_seconds=state['elapsed_seconds'],allocated_gpu_hours=2*state['elapsed_seconds']/3600,peak_allocated_bytes_per_gpu=max(r['peak_allocated_bytes_per_gpu'] for r in logs),peak_reserved_bytes_per_gpu=max(r['peak_reserved_bytes_per_gpu'] for r in logs),resumes=sum(1 for r in logs if r.get('resumed_this_epoch')),augmentation_fallbacks=sum(r['augmentation_fallbacks'] for r in logs),checkpoint_storage_bytes=sum(p.stat().st_size for p in OUT.glob('*.pth')),evidence={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'best.pth',OUT/'latest.pth',OUT/'initial.pth',report,OUT/'TRAIN.jsonl',R/'FREEZE.json',R/'TRAIN_PREACCESS.json',R/'VALIDATION_PREACCESS.json',R/'DATA_AUDIT.json',R/'RELOAD_CHECK.json',R/'BATCH_DECISION.json',R/'BASELINE_E193_NATIVE_VALIDATION.json']})
    d['baseline_same_grid']={'trilinear_logits':bfile['native_trilinear_logits'],'nearest_mask':bfile['native_nearest_mask'],'grid128_recomputed':bfile['grid128_recomputed']}
    d['comparison_native_minus_baseline']=paired(bfile['per_case'],raw['per_case'])
    d['comparison_scope']='same reused validation108 on the same native crop grid; candidate selected on this validation, baseline selected on its own 128^3 grid; development comparison, not independent confirmation'
    write(R/'FINAL_RESULT.json',d);write(R/'RESULT.json',d);write(R/'RAW_SELECTED_VALIDATION.json',raw)
    with (R/'RESULT.md').open('a') as f:
        f.write(f"\n200轮完成，{200*steps}次更新，best/latest重载一致性通过。选第{best['epoch']}轮，原始栅格Scan/Subject={d['scan_dice']:.6f}/{d['subject_dice']:.6f}，Scan95%CI={d['scan_ci95']}，Point {d['point_hit']:.6f}；未运行test。[最终报告](FINAL_RESULT.json)。\n")
        f.write('\n新−基线（同108原始裁剪栅格）配对差值：'+json.dumps(d['comparison_native_minus_baseline'],ensure_ascii=False)+'。相同复用验证上的选优比较，非独立确认。\n')
    return best,d['comparison_native_minus_baseline']


def launch(handle):
    proc=subprocess.Popen([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','scripts/train_segresnet_native_res.py','train'],cwd=ROOT,env=ENV,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    with (R/'LAUNCH.jsonl').open('a') as f:f.write(json.dumps({'utc':now(),'pid':proc.pid,'supervisor_pid':os.getpid(),'worker':'1100631','resume':(OUT/'latest.pth').exists()})+'\n')
    return proc


def stop(proc):
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=30)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()


def main():
    lock=(R/'SUPERVISOR.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
    assert json.loads((R/'GPU_SMOKE.json').read_text())['status']=='passed'
    logpath=R/'train.log';handle=logpath.open('a');proc=None;last=0;attempt=0
    try:
        if not (R/'RESULT.json').exists():
            write(R/'RESULT.json',{'status':'starting','completed_epochs':0,'target_epochs':EPOCHS,'per_gpu_batch':batch_decision(),'test_access':False,'new_accuracy_result':False})
            (R/'RESULT.md').write_text('# 原始分辨率整脑纯SegResNet进度\n\n训练已授权并启动，尚未完成第1轮，无新精度结果。缓存/基线对照/显存探测已通过；见[七问](IDEA.md)、[batch决定](BATCH_DECISION.json)、[基线同栅格对照](BASELINE_E193_NATIVE_VALIDATION.json)。\n')
        documents('已启动，等待首轮');checks()
        while True:
            attempt+=1;proc=launch(handle)
            write(R/'HEARTBEAT.json',{'utc':now(),'alive':True,'status':'starting','attempt':attempt,'last_complete_epoch':last})
            while proc.poll() is None:
                rows=[json.loads(x) for x in (OUT/'TRAIN.jsonl').read_text().splitlines()] if (OUT/'TRAIN.jsonl').exists() else []
                if rows and rows[-1]['epoch']!=last:
                    for row in rows:
                        if row['epoch']>last:archive(row);last=row['epoch']
                    documents('训练中',last,rows[-1]['selection']);checks()
                write(R/'HEARTBEAT.json',{'utc':now(),'alive':True,'status':'running','attempt':attempt,'pid':proc.pid,'last_complete_epoch':last,'latest_validation':rows[-1].get('validation') if rows else None,'best':rows[-1].get('selection') if rows else None,'test_pixels_access':False})
                if time.time()-logpath.stat().st_mtime>2400:stop(proc);raise RuntimeError('training log stalled >40min')
                time.sleep(20)
            rows=[json.loads(x) for x in (OUT/'TRAIN.jsonl').read_text().splitlines()] if (OUT/'TRAIN.jsonl').exists() else []
            done=(OUT/'STATUS.json').exists() and json.loads((OUT/'STATUS.json').read_text())['status']=='completed'
            if proc.returncode==0 and done:break
            with (R/'ATTEMPTS.jsonl').open('a') as f:f.write(json.dumps({'utc':now(),'attempt':attempt,'returncode':proc.returncode,'last_complete_epoch':rows[-1]['epoch'] if rows else 0})+'\n')
            if attempt>=MAX_ATTEMPTS:raise RuntimeError(f'training failed after {attempt} attempts (returncode {proc.returncode})')
            time.sleep(60)
        for row in rows:
            if row['epoch']>last:archive(row);last=row['epoch']
        best,final=finalize(rows);documents('200轮完成',200,best,final);checks()
        write(R/'HEARTBEAT.json',{'utc':now(),'alive':False,'status':'completed','last_complete_epoch':200,'best':best,'paired_vs_baseline':final,'test_pixels_access':False})
        p=ROOT/'AFLOW_BRATS_WEEKLY_REPORT.md';p.write_text(p.read_text()+'\n原始分辨率整脑纯SegResNet（不缩放到128³）200轮完成，最佳验证原始栅格Scan='+f"{best['scan_dice']:.6f}"+'，同栅格配对差值（新−原193）Scan='+f"{final['scan_delta']:+.6f}"+'、95%CI='+str(final['scan_delta_ci95'])+'，无新增test。[最终结果]('+REL+'/FINAL_RESULT.json)。\n');checks()
    except BaseException as exc:
        stop(proc)
        if (OUT/'STATUS.json').exists():last=json.loads((OUT/'STATUS.json').read_text())['last_complete_epoch']
        failure={'utc':now(),'alive':False,'status':'failed','last_complete_epoch':last,'attempt':attempt,'error':repr(exc),'traceback':traceback.format_exc()};write(R/'FAILURE.json',failure);write(R/'HEARTBEAT.json',failure)
        try:documents('执行中断，未完成200轮',last)
        except BaseException:pass
        with (R/'RESULT.md').open('a') as f:f.write('\n执行中断：'+repr(exc)+'，实际完成'+str(last)+'轮，不能称200轮完成。\n')
        raise
    finally:handle.close()

if __name__=='__main__':main()
