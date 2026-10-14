"""Single fixed 972/108 run; archival only, no dataset pixel access."""
import datetime,hashlib,json,os,subprocess,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'archive/research_records/2026-09-16_sammed3d_graymean_single972';W=ROOT/'workspace/BraTS/sammed3d_t1ce_wt/graymean_single972_20260916';PY=str(ROOT.parent/'sammed3d-venv/bin/python')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def intervals(rows):
 subjects=sorted({r['subject_id'] for r in rows});n=np.array([sum(r['subject_id']==s for r in rows) for s in subjects]);sums=np.array([sum(r['final5'] for r in rows if r['subject_id']==s) for s in subjects]);ix=np.random.default_rng(20260916).integers(0,len(subjects),(10000,len(subjects)))
 return {'scan_final5':float(sums.sum()/n.sum()),'subject_final5':float((sums/n).mean()),'scan_final5_ci95':np.quantile(sums[ix].sum(1)/n[ix].sum(1),[.025,.975]).tolist(),'subject_final5_ci95':np.quantile((sums/n)[ix].mean(1),[.025,.975]).tolist(),'scan_best5':float(np.mean([r['best5'] for r in rows])),'scans':len(rows),'subjects':len(subjects)}
def main():
 a=W/'graymean';status=json.loads((a/'STATUS.json').read_text())
 if status['status']!='completed' or status['last_complete_epoch']!=200:raise RuntimeError('200 epochs not completed; do not finalize')
 best=json.loads((a/'BEST.json').read_text());report=a/best['report'];ck=a/'best.pth'
 if sha(report)!=best['report_sha256'] or sha(ck)!=best['checkpoint_sha256']:raise RuntimeError('selected evidence mismatch')
 raw=json.loads(report.read_text());result=intervals(raw['rows'])
 logs=[json.loads(x) for x in (a/'TRAIN.jsonl').read_text().splitlines()]
 if [x['epoch'] for x in logs]!=list(range(1,201)) or any(x['steps']!=972 for x in logs):raise RuntimeError('200 full972 epochs missing')
 index=json.loads((R/'CACHE_INDEX.json').read_text());files=[report,ck,a/'latest.pth',a/'PREACCESS.json',a/'DEVELOPMENT_PREACCESS.json',a/'BEST.json',R/'PREREG.json',R/'IMPLEMENTATION_FREEZE.json',R/'CACHE_INDEX.json']
 result.update(status='completed_200_epochs_single_graymean',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected_epoch=best['epoch'],epochs=200,train_scans=972,train_subjects=444,cohort='reused fixed validation108/50; same-cohort selection',test_outer_confirmation_evaluation=False,old_confirmation_subset_used_as_original_train=True,exact_author_reproduction=False,bootstrap='10000 patient-cluster percentile95 seed20260916',cost={'active_gpu_count':1,'allocated_gpu_count':2,'training_seconds':sum(x['training_seconds'] for x in logs),'elapsed_seconds':status['elapsed_seconds'],'peak_allocated_bytes':max(x['peak_allocated_bytes'] for x in logs),'cache_bytes':index['bytes'],'checkpoint_bytes':ck.stat().st_size+(a/'latest.pth').stat().st_size,'gpu_hours':status['elapsed_seconds']/3600,'second_gpu_training_hours':0,'inference':'one image encoder + five interactive prompt decoder calls; no extra grayscale inference'},evidence={str(p.relative_to(ROOT)):sha(p) for p in files})
 write(R/'RAW_VALIDATION.json',raw);result['raw_report_sha256']=sha(R/'RAW_VALIDATION.json');write(R/'RESULT.json',result)
 summary=f"单组SAM-Med3D灰度项已完成同一972例的200轮；固定复用108扫描/50患者验证，选第{best['epoch']}轮，第5点击Scan Dice {result['scan_final5']:.6f}、Subject {result['subject_final5']:.6f}，Scan 95%区间{result['scan_final5_ci95']}。"
 (R/'RESULT.md').write_text('# 单组972训练200轮结果\n\n'+summary+'\n\n只训练灰度项模型，无对照，不据此宣称灰度的独立因果收益。师兄平均亮度辅助轮廓的直觉是否优于基础模型尚未由本单组实验检验；不冒称作者代码精确复制或复现0.731。保持原阈值/空间/交互/选择协议。真值辅助裁剪和点击不能进入自动主榜；同验证选优区间不是独立确认。未访问测试像素；旧训练池子确认已回归训练，不能作为本轮确认。\n\n完整指标、GPU时间、显存、存储和SHA见[机器报告](RESULT.json)，逐病例见[原始报告](RAW_VALIDATION.json)，设置见[七问](IDEA.md)。单张A100训练，另卡空闲；推理一次编码加五次交互解码，无灰度额外前向。下一步审核结果，无自动测试准入。\n')
 for name in ['AFLOW_BRATS_HANDOFF_CURRENT.md','AFLOW_BRATS_WEEKLY_REPORT.md']:
  p=ROOT/name;t=p.read_text();t+='\n'+summary+' [完整结果](archive/research_records/2026-09-16_sammed3d_graymean_single972/RESULT.md)。\n';p.write_text(t)
 p=ROOT/'AFLOW_BRATS_RESULTS.md';t=p.read_text();entry='\n## '+datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d')+'｜单组SAM-Med3D灰度项200轮完成\n\n**问题：** 固定972训练的完整效果待验证。\n**做法：** 唯一灰度组、固定108验证。\n**结果：** '+summary+'\n**决定：** 单组结果不作因果消融，不自动进测试。[核对材料](archive/research_records/2026-09-16_sammed3d_graymean_single972/RESULT.md)\n';marker='\n---\n\n[导师汇报正文]';p.write_text(t.replace(marker,entry+marker) if marker in t else t+entry)
 p=ROOT/'AFLOW_BRATS_METHOD_CATALOG.md';t=p.read_text();t+='\n## SAM-Med3D固定972单组灰度项｜200轮完成\n\n**一句话：** 用平均亮度辅助轮廓学习。 **出发点：** 需要完整972训练200轮的灰度模型。\n**具体做法：** 唯一灰度组，其余数据/提示/阈值固定。\n**结果：** '+summary+'\n**是否支持直觉：** 无对照，仅完成单组复现训练，因果收益未检验。\n**代价/下一步：** 单A100训练，另卡闲置，无灰度额外推理；审核结果，不自动进测试。[核对材料](archive/research_records/2026-09-16_sammed3d_graymean_single972/RESULT.md)\n';p.write_text(t)
 checks=[]
 for n in ['verify_aflow_brats_doc_layout.py','verify_aflow_brats_method_catalog.py','verify_aflow_brats_leaderboard.py','verify_aflow_root_artifacts.py']:
  p=subprocess.run([PY,'scripts/'+n],cwd=ROOT,capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'});checks.append({'name':n,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
 write(R/'FINAL_VALIDATORS.json',checks)
 if any(x['returncode'] for x in checks):raise RuntimeError('final validators failed')
 print(summary,flush=True)
if __name__=='__main__':main()
