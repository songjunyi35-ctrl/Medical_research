"""Keep current documents consistent with durable training status; no model/data access."""
import datetime,json,re,subprocess,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'archive/research_records/2026-09-16_sammed3d_graymean_single972';PY=str(ROOT.parent/'sammed3d-venv/bin/python')
LINK='archive/research_records/2026-09-16_sammed3d_graymean_single972/'
def section_replace(text,title,new):
 start=text.find(title)
 if start<0:return text+'\n'+new+'\n'
 end=text.find('\n## ',start+3)
 return text[:start]+new+'\n'+(text[end:] if end>=0 else '')
def publish(stage):
 completed=stage=='200轮完成，报告和校验已归档'
 failed='失败' in stage
 explanation='固定972训练/108验证/270测试；唯一SAM-Med3D灰度组，目标200轮。'+stage+'。正式训练使用1张A100，第二张空闲，无新增测试评估。'
 if completed:
  d=json.loads((R/'RESULT.json').read_text());assert d['epochs']==200 and d['status']=='completed_200_epochs_single_graymean'
  explanation=f"固定972例单组灰度训练200轮完成；108例复用验证，第{d['selected_epoch']}轮最佳，第5点击Scan/Subject Dice {d['scan_final5']:.6f}/{d['subject_final5']:.6f}。未运行测试；无对照，不能单独证明灰度收益。"
 p=ROOT/'AFLOW_BRATS_HANDOFF_CURRENT.md';t=p.read_text();block='<!-- SAMMED3D_SINGLE972_STATUS -->\n'+explanation+' [运行记录]('+LINK+'RUN_STATUS.json)'+(' · [最终报告]('+LINK+'RESULT.md)' if completed else '')+'\n<!-- END_SAMMED3D_SINGLE972_STATUS -->'
 pattern=r'<!-- SAMMED3D_SINGLE972_STATUS -->.*?<!-- END_SAMMED3D_SINGLE972_STATUS -->'
 t=re.sub(pattern,lambda m:block,t,flags=re.S) if '<!-- SAMMED3D_SINGLE972_STATUS -->' in t else t+'\n'+block+'\n';p.write_text(t)
 p=ROOT/'AFLOW_BRATS_METHOD_CATALOG.md';t=p.read_text();title='## 师兄灰度校准与SAM-Med3D｜'
 card='## 师兄灰度校准与SAM-Med3D｜'+('200轮完成' if completed else ('执行失败，未完成200轮' if failed else '当前唯一单组，972例、200轮'))+'\n\n**一句话：** 用区域平均亮度辅助肿瘤轮廓学习。 **出发点：** 用户要求固定原始972例、单组完整200轮。\n**具体做法：** SAM-Med3D首次提示加灰度均值误差，固定972/108/270、输入处理和五点击评估，不增加对照。\n**结果：** '+explanation+'\n**是否支持直觉：** '+('单组训练完成；无对照不能证明独立收益。' if completed else '正式200轮结果尚未完成，不能提前下结论。')+'\n**代价/下一步：** 已分配双A100，单卡正式训练；灰度项不增推理前向。'+('审核最终报告，无自动测试准入。' if completed else '按冻结协议完成200轮并归档，工程失败留证据。')+' [核对材料]('+LINK+('RESULT.md' if completed else 'IDEA.md')+')\n'
 t=section_replace(t,title,card)
 if completed:
  duplicate='## SAM-Med3D固定972单组灰度项｜200轮完成';start=t.find(duplicate)
  if start>=0:
   end=t.find('\n## ',start+3);t=t[:start]+(t[end:] if end>=0 else '')
 p.write_text(t)
 if completed:
  for name in ['AFLOW_BRATS_HANDOFF_CURRENT.md','AFLOW_BRATS_WEEKLY_REPORT.md']:
   p=ROOT/name;t=p.read_text();t=t.replace('正式单卡训练待启动','已完成200轮').replace('合成smoke通过，正在补齐216例缓存','已完成200轮，结果已归档');p.write_text(t)
 checks=[]
 for n in ['verify_aflow_brats_doc_layout.py','verify_aflow_brats_method_catalog.py','verify_aflow_brats_leaderboard.py','verify_aflow_root_artifacts.py']:
  p=subprocess.run([PY,'scripts/'+n],cwd=ROOT,capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'});checks.append({'name':n,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
 (R/'PUBLICATION_VALIDATORS.json').write_text(json.dumps({'stage':stage,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks},indent=2)+'\n')
 if any(x['returncode'] for x in checks):raise RuntimeError('publication validators failed')
 return completed or failed
if __name__=='__main__':
 last=None;deadline=time.monotonic()+60*3600
 while time.monotonic()<deadline:
  stage=json.loads((R/'RUN_STATUS.json').read_text())['stage']
  if stage!=last:
   if publish(stage):break
   last=stage
  time.sleep(30)
