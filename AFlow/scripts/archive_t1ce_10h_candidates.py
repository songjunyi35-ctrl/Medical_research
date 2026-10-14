"""Per-candidate negative/positive archival, including validation-only stop cases."""
import sys,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.t1ce_10h_common import *

def main():
 catalog=ROOT/'AFLOW_BRATS_METHOD_CATALOG.md';text=catalog.read_text()
 for arm in 'ABC':
  rec=R/arm;out=OUT/arm
  if not (rec/'STOP_RESULT.json').exists():continue
  state=read(rec/'STOP_RESULT.json');reports=[read(p) for p in (R/'reports').glob(arm+'_*.json')];reports.sort(key=lambda x:-x['metrics']['scan_dice'])
  selection=read(out/'BEST.json') if (out/'BEST.json').exists() else None
  val=read(out/selection['report']) if selection else None
  initial=read(out/'validation_e000.json') if (out/'validation_e000.json').exists() else None
  best=reports[0] if reports else None
  unchanged=(selection and selection['epoch']==0)
  if state.get('state')=='not_started':interpretation='初始错误诊断未达到预设启动条件，没有训练或本候选测试；不能据此否定该损失的一般价值。'
  elif unchanged:interpretation='验证没有选中任何新增训练轮，所报最好检查点仍是初始化模型；本实现未支持继续训练带来增益，不能把初始化模型成绩当本方法收益。'
  elif best and best['vs_strongest_reference']['scan_delta']>=.03 and best['metrics']['scan_dice']>=.8:interpretation='在已复用270定向优化中达到数值目标；不证明独立泛化或该因素的净因果效果，额外训练未设匹配预算对照。'
  elif best and best['vs_strongest_reference']['scan_delta']>0:interpretation='只得到定向队列上的正点估计，未达到质变目标；需连同配对区间解释，不作独立泛化或因果声明。'
  else:interpretation='未支持超过本轮最强参考；仅约束这个实现和预算，不证明方向普遍无效。'
  data={'utc':now(),'arm':arm,'training':state,'validation_selection':selection,'validation':val,'initial_validation':initial,'best_target_gate':best,'target_gate_ids':[d['id'] for d in reports],'test_access':bool(reports),'test_tuned':True,'test_labels_in_segmentation_training':False,'outer_confirmation_access':False,'interpretation':interpretation}
  evidence=[p for p in [rec/'STOP_RESULT.json',rec/'TRAIN_PREACCESS.json',rec/'GPU_SMOKE.json',rec/'FREEZE.json',rec/'WEIGHTS.json',out/'best.pth',out/'latest.pth',out/selection['report'] if selection else rec/'NONEXISTENT'] if p.exists()]
  evidence += [R/'reports'/(d['id']+'.json') for d in reports]
  data['evidence_sha256']={str(p.relative_to(ROOT)):sha(p) for p in evidence};write(rec/'RESULT.json',data)
  metrics=(f"108/50原始栅格验证选新增第{selection['epoch']}轮，Scan {val['metrics']['scan_dice']:.9f}；分支第0轮{initial['metrics']['scan_dice']:.9f}。" if selection else '未启动，未产生本候选验证选择。')
  target=(f"270/119同原始栅格门审最高Scan {best['metrics']['scan_dice']:.9f}、Subject {best['metrics']['subject_dice']:.9f}、Point {best['metrics']['point_hits']}/270；对最强参考差{best['vs_strongest_reference']['scan_delta']:+.9f}，患者配对95%区间{best['vs_strongest_reference']['scan_delta_ci95']}。" if best else '本候选没有270模型访问。')
  cp=(f"[验证选定模型](../../../../{str((out/'best.pth').relative_to(ROOT))})" if selection else '无新增模型。')
  (rec/'RESULT.md').write_text(f"# 10小时候选{arm}结果\n\n真实完成{state.get('last_complete_epoch',0)}轮、{state.get('completed_updates',0)}次完整轮内更新；状态{state.get('state')}，原因：{state.get('reason')}。耗时{state.get('elapsed',0)/3600:.3f}小时。\n\n{metrics}\n\n{target}\n\n{interpretation}\n\n输入仅T1ce，原972/108/270与患者隔离不变；270只诊断与选择，不进入分割梯度。test-tuned，配对区间仅描述性，不消除反复选择偏差；无outer/confirmation。单因素意图与控制见[七问](IDEA.md)，具体实现变化和工程修订在总归档，不能称匹配预算的因果消融。\n\nA部署8块前向，B/C一次128³前向；训练大缓存仅/tmp，保留历史文件。此分支关闭，不自动恢复或加参。\n\n[机器结果](RESULT.json) · {cp} · [总结果](../RESULT.md)\n")
  marker=f'## 10小时候选{arm}：';a=text.index(marker);b=text.find('\n## ',a+3);b=len(text) if b<0 else b;card=text[a:b]
  card=re.sub(r'｜[^\n]+','｜'+state.get('state','unknown'),card,count=1)
  card=re.sub(r'\*\*结果：\*\*[^\n]+','**结果：** '+metrics+' '+target,card)
  card=re.sub(r'\*\*是否支持直觉：\*\*[^\n]+','**是否支持直觉：** '+interpretation,card)
  card=re.sub(r'\*\*代价/下一步：\*\*[^\n]+',f"**代价/下一步：** 完成{state.get('last_complete_epoch',0)}轮，耗时{state.get('elapsed',0)/3600:.3f}小时，分支关闭。[结果]({str((rec/'RESULT.md').relative_to(ROOT))})。",card)
  text=text[:a]+card+text[b:]
 catalog.write_text(text)

if __name__=='__main__':main()
