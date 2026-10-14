"""Finite top-three output panel, deployment export and campaign publication."""
import sys,os,json,time,itertools,subprocess,shutil,hashlib,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.t1ce_10h_common import *
from scripts.evaluate_t1ce_10h import register_report

def state_hash(path):
 s=torch.load(path,map_location='cpu',weights_only=False)['model_state_dict'];h=hashlib.sha256()
 for name,t in sorted(s.items()):h.update(name.encode());h.update(t.cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def select():
 verify();all_reports=[read(p) for p in (R/'reports').glob('*.json') if not p.name.startswith('panel_')]
 all_reports.sort(key=lambda d:(-d['metrics']['scan_dice'],d['forward_equivalents'],d.get('gate_epoch') or 0,d['id']))
 chosen=[];seen=set()
 for d in all_reports:
  signature=(state_hash(ROOT/d['checkpoint_path']),d['mode'])
  if signature in seen:continue
  seen.add(signature);chosen.append(d)
  if len(chosen)==3:break
 recipes=[]
 for count in [1,2,3]:
  for group in itertools.combinations(chosen,count):
   cost=sum(d['forward_equivalents'] for d in group)
   if cost>8+1e-10:continue
   for threshold in ([.5] if count==3 else [.4,.5,.6]):
    recipes.append({'id':'panel_'+str(len(recipes)+1).zfill(2),'components':[d['id'] for d in group],'threshold':threshold,'forward_equivalents':cost,'formula':'arithmetic mean of native-crop float32 logits, then common threshold'})
 assert len(recipes)<=19
 write(R/'FINAL_PANEL_FREEZE.json',{'utc':now(),'selected':[{'id':d['id'],'report_sha256':sha(R/'reports'/(d['id']+'.json')),'checkpoint_sha256':d['checkpoint_sha256'],'logits_dir':d['logits_dir']} for d in chosen],'recipes':recipes,'selection':'top3 unique model state + inference mode, Scan then cost then earliest','test_tuned':True,'new_training_allowed':False})
 print(json.dumps({'selected':[d['id'] for d in chosen],'recipes':len(recipes)},indent=2))

def panel():
 freeze=read(R/'FINAL_PANEL_FREEZE.json');chosen={x['id']:read(R/'reports'/(x['id']+'.json')) for x in freeze['selected']}
 idx,targets=native_targets('test');case_ids=[r['case_id'] for r in rows('test')]
 # Transfer receipts preserve raw float32 logs; do not regenerate model outputs.
 paths={}
 for key,d in chosen.items():
  p=Path(d['logits_dir'])
  if not p.exists():p=SCRATCH/'imported_logits'/key
  assert all((p/(cid+'.npy')).is_file() for cid in case_ids),key
  paths[key]=p
 recipes=freeze['recipes'];results={r['id']:[] for r in recipes};tick=time.monotonic()
 write(R/'PANEL_PREACCESS.json',{'utc':now(),'freeze_sha256':sha(R/'FINAL_PANEL_FREEZE.json'),'cached_predictions_only':True,'test_tuned':True})
 for cid in case_ids:
  z={key:torch.from_numpy(np.load(path/(cid+'.npy'),allow_pickle=False)) for key,path in paths.items()};y=torch.from_numpy(targets[cid])
  for r in recipes:
   mixed=sum(z[key] for key in r['components'])/len(r['components']);results[r['id']].append(case_metric(mixed,y,cid,r['threshold']))
 targets.close()
 for recipe in recipes:
  comps=[chosen[x] for x in recipe['components']];ident=recipe['id'];manifest=R/'deployment_candidates'/(ident+'.json')
  write(manifest,{'components':[{'checkpoint_path':c['checkpoint_path'],'checkpoint_sha256':c['checkpoint_sha256'],'mode':c['mode']} for c in comps],**recipe,'only_external_input':'T1ce','no_labels_or_case_lookup':True})
  cases=results[ident];ref=read(ROOT/read(R/'REFERENCE_PANEL.json')['strongest_report']);pure=read(R/'reports/ref_pure193.json')
  d={'id':ident,'utc':now(),'split':'test','role':'fixed_output_panel','mode':'ensemble' if len(comps)>1 else comps[0]['mode'],'metrics':summary(cases),'per_case':cases,'threshold':recipe['threshold'],'forward_equivalents':recipe['forward_equivalents'],'checkpoint_path':str(manifest.relative_to(ROOT)),'checkpoint_sha256':sha(manifest),'components':recipe['components'],'logit_formula':recipe['formula'],'test_tuned':True,'vs_strongest_reference':compare(ref['per_case'],cases),'vs_fixed193':compare(pure['per_case'],cases),'panel_compute_seconds':time.monotonic()-tick,'new_model_inference':False}
  path=R/'reports'/(ident+'.json');write(path,d);register_report(path)
 reports=[read(R/'reports'/(r['id']+'.json')) for r in recipes];reports.sort(key=lambda d:(-d['metrics']['scan_dice'],d['forward_equivalents'],d['id']))
 winner=reports[0];best_single=next(d for d in reports if len(d['components'])==1)
 write(R/'FINAL_SELECTION.json',{'utc':now(),'overall':winner['id'],'best_single_model':best_single['id'],'selection':'maximum Scan Dice, lower compute then earlier panel ID','threshold':winner['threshold'],'target_passed':winner['metrics']['scan_dice']>=.8 and winner['vs_strongest_reference']['scan_delta']>=.03,'test_tuned':True})
 export=OUT/'deployment';export.mkdir(exist_ok=True);recipe=read(ROOT/winner['checkpoint_path']);components=[]
 for c in recipe['components']:
  dest=export/(Path(c['checkpoint_path']).parent.name+'_'+c['checkpoint_sha256'][:16]+'.pth')
  if not dest.exists():shutil.copy2(ROOT/c['checkpoint_path'],dest)
  assert sha(dest)==c['checkpoint_sha256'];components.append({**c,'checkpoint_path':str(dest.relative_to(ROOT))})
 write(export/'manifest.json',{**recipe,'components':components,'test_tuned':True,'source_report':str((R/'reports'/(winner['id']+'.json')).relative_to(ROOT))})
 print(json.dumps(read(R/'FINAL_SELECTION.json'),indent=2))

def update_block(path,mark,body):
 p=Path(path);s=p.read_text();a=f'<!-- {mark} -->';b=f'<!-- END_{mark} -->';block=a+'\n'+body+'\n'+b
 if a in s:s=s[:s.index(a)]+block+s[s.index(b)+len(b):]
 else:s+='\n\n'+block+'\n'
 p.write_text(s)

def publish(final=False):
 sel=read(R/'FINAL_SELECTION.json');win=read(R/'reports'/(sel['overall']+'.json'));single=read(R/'reports'/(sel['best_single_model']+'.json'));ref=read(R/'REFERENCE_PANEL.json')
 states={c:read(R/c/'STOP_RESULT.json') if (R/c/'STOP_RESULT.json').exists() else {'state':'unknown'} for c in 'ABC'}
 outcome={'utc':now(),'status':'complete' if final else 'results_ready_monitoring_until_deadline','selection':sel,'overall':{k:v for k,v in win.items() if k!='per_case'},'best_single_model':{k:v for k,v in single.items() if k!='per_case'},'reference':ref,'training':states,'test_tuned':True,'test_labels_in_segmentation_training':False,'outer_confirmation_access':False,'disk':disk(),'protocol':P,'monitoring_completed':final}
 write(R/'RESULT.json',outcome)
 m=win['metrics'];delta=win['vs_strongest_reference'];sm=single['metrics'];rel=str(R.relative_to(ROOT))
 verdict='达到本轮目标' if sel['target_passed'] else '未达到0.80且至少+0.03目标'
 body=f'''# 10小时单T1ce定向优化结果

状态：{'10小时监督结束' if final else '模型优化结束，监督仍持续至硬截止'}；{verdict}。

270 scans /119 subjects，原始1mm crop，排除填充。总体选{win['id']}：Scan **{m['scan_dice']:.9f}**，Subject **{m['subject_dice']:.9f}**，Point {m['point_hits']}/270；Scan患者簇95%区间{m['scan_ci95']}。最佳单模型{single['id']} Scan {sm['scan_dice']:.9f}，Subject {sm['subject_dice']:.9f}；单模型不等于单次前向。

固定五模型最强参考{ref['strongest_id']} Scan {ref['strongest_scan']:.9f}；总体差{delta['scan_delta']:+.9f}，患者配对95%区间{delta['scan_delta_ci95']}。相对固定193同栅格差{win['vs_fixed193']['scan_delta']:+.9f}，区间{win['vs_fixed193']['scan_delta_ci95']}。所有区间仅描述性，不消除反复选择偏差。

本轮明确test-tuned：270标签用于错误诊断、门审、模型/阈值选择；仅原972参与分割训练；108用于分支检查点选择。不是独立泛化、盲测或外部确认，无新outer/confirmation。保留传统榜和全部失败门审，不能用本榜冒充传统最佳。

方法直觉、单因素和控制见[七问](IDEA.md)。训练状态：{json.dumps(states,ensure_ascii=False)}。输出预算{win['forward_equivalents']:.3f}个128³前向等价量，阈值{win['threshold']}；实际耗时和重载检验见DEPLOYMENT_CHECK.json。持久磁盘新增{outcome['disk']['persistent_bytes']/2**30:.3f}GiB；临时缓存不作为持久交付。负结果仅评价本实现与预算，不证明一般方向无效。

[机器结果](RESULT.json) · [逐例总体报告](reports/{win['id']}.json) · [单模型报告](reports/{single['id']}.json) · [选择收据](FINAL_SELECTION.json) · [部署清单](../../../workspace/BraTS/segresnet_t1ce_wt/test270_10h_20260921/deployment/manifest.json) · [访问账本](ACCESS_SUMMARY.json) · [协议](PROTOCOL.json)
'''
 (R/'RESULT.md').write_text(body)
 short=f"10小时单T1ce定向优化：{outcome['status']}。270/119原始栅格总体Scan {m['scan_dice']:.6f}、Subject {m['subject_dice']:.6f}、Point {m['point_hits']}/270；最佳单模型Scan {sm['scan_dice']:.6f}。对最强同栅格参考差{delta['scan_delta']:+.6f}、95%CI{delta['scan_delta_ci95']}，{verdict}。全部test-tuned，非独立泛化；270不进分割梯度，无outer/confirmation。[报告]({rel}/RESULT.md)。"
 for fn in ['AFLOW_BRATS_HANDOFF_CURRENT.md','AFLOW_BRATS_RESULTS.md','AFLOW_BRATS_WEEKLY_REPORT.md']:update_block(ROOT/fn,'TEST270_10H',short)
 cat=ROOT/'AFLOW_BRATS_METHOD_CATALOG.md';s=cat.read_text()
 for c in 'ABC':
  marker=f'## 10小时候选{c}：';a=s.index(marker);b=s.find('\n## ',a+3);b=len(s) if b<0 else b;card=s[a:b];state=states[c]
  reports=[read(p) for p in (R/'reports').glob(c+'_*.json')];best=max(reports,key=lambda d:d['metrics']['scan_dice']) if reports else None
  result=f"真实完成{state.get('last_complete_epoch',0)}轮/{state.get('completed_updates',0)}更新；{state.get('reason','')}。"+(f" 270定向门审最高Scan {best['metrics']['scan_dice']:.6f}。" if best else ' 未访问本候选270。')
  card=re.sub(r'｜[^\n]+','｜'+state.get('state','unknown'),card,count=1)
  card=re.sub(r'\*\*结果：\*\*[^\n]+','**结果：** '+result,card)
  card=re.sub(r'\*\*是否支持直觉：\*\*[^\n]+','**是否支持直觉：** '+('需结合定向误差与参考差值解释，不作为独立泛化证据。' if best else '初始诊断或工程条件未支持启动，不能宣称方法无效。'),card)
  card=re.sub(r'\*\*代价/下一步：\*\*[^\n]+',f"**代价/下一步：** 本分支关闭，不自动追加训练。真实用时{state.get('elapsed',0)/3600:.3f}小时；[完整报告]({rel}/RESULT.md)。",card)
  s=s[:a]+card+s[b:]
 cat.write_text(s)

if __name__=='__main__':
 {'select':select,'panel':panel,'publish':publish,'complete':lambda:publish(True)}[sys.argv[1]]()
