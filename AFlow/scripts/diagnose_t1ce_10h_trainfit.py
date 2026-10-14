"""Post-selection clean train/validation diagnostics, no additional test access."""
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.t1ce_10h_common import *

@torch.no_grad()
def main():
 verify();assert (R/'FINAL_SELECTION.json').exists() and (OUT/'deployment/manifest.json').exists()
 if (R/'TRAIN_FIT_DIAGNOSTIC.json').exists():raise RuntimeError('diagnostic already completed')
 manifest=read(OUT/'deployment/manifest.json');manifest_sha=sha(OUT/'deployment/manifest.json')
 configs={
  'fixed193':{'threshold':.5,'components':[{'checkpoint_path':str(source('pure193').relative_to(ROOT)),'checkpoint_sha256':SOURCES['pure193'][1],'mode':'128'}]},
  'native173':{'threshold':.5,'components':[{'checkpoint_path':str(source('native173').relative_to(ROOT)),'checkpoint_sha256':SOURCES['native173'][1],'mode':'native'}]},
  'selected_workflow':{'threshold':manifest['threshold'],'components':manifest['components']}}
 write(R/'TRAIN_FIT_PREACCESS.json',{'utc':now(),'configs':configs,'deployment_manifest_sha256':manifest_sha,'roles':['train972/444','validation108/50'],'test_pixels_or_predictions_accessed':False,'used_for_selection_or_training':False,'augmentation':False,'grid':'native crop; existing train/validation canvas160x208x160; excludes padding','purpose':'describe fitting gap, not causal diagnosis or independent generalization','code_sha256':sha(Path(__file__))})
 numeric();seed(SEED);torch.set_num_threads(2);device=torch.device('cuda:0');pool={}
 for config in configs.values():
  for comp in config['components']:
   key=(comp['checkpoint_sha256'],comp['mode'])
   if key not in pool:
    path=ROOT/comp['checkpoint_path'];assert sha(path)==comp['checkpoint_sha256'];model=model_for(device,path);model.eval();pool[key]=model
 results={};started=time.monotonic()
 for split in ['train','validation']:
  native=GridDataset(split,native=True);grid=GridDataset(split);slots={c:i for i,c in enumerate(grid.ids)}
  cases={name:[] for name in configs};tick=time.monotonic()
  for start in range(0,len(native),2):
   if time.monotonic()-started>90*60 or time.time()>timestamp(P['training_stop_utc']):raise RuntimeError('diagnostic time budget reached')
   items=[native[i] for i in range(start,min(start+2,len(native)))];small=[grid[slots[it['case_id']]] for it in items]
   xn=torch.stack([it['image'] for it in items]).to(device);xg=torch.stack([it['image'] for it in small]).to(device)
   logits={}
   for key,model in pool.items():
    mode=key[1]
    with torch.autocast('cuda',dtype=torch.bfloat16):logits[key]=predict(model,xg if mode=='128' else xn,mode)
    assert torch.isfinite(logits[key]).all()
   for k,it in enumerate(items):
    cid=it['case_id'];geo=native.index['geometry'][cid];sl=crop_slices(geo);y=it['label'][0][sl].to(device);crops={}
    for key,z in logits.items():
     crops[key]=F.interpolate(z[k:k+1],size=geo['extent'],mode='trilinear',align_corners=False)[0,0] if key[1]=='128' else z[k,0][sl]
    for name,config in configs.items():
     mixed=sum(crops[(c['checkpoint_sha256'],c['mode'])] for c in config['components'])/len(config['components'])
     cases[name].append(case_metric(mixed,y,cid,config['threshold']))
   if start%50==0:
    write(R/'TRAIN_FIT_PROGRESS.json',{'utc':now(),'split':split,'done':start+len(items),'total':len(native),'elapsed_seconds':time.monotonic()-started})
    print(json.dumps({'utc':now(),'split':split,'done':start+len(items),'total':len(native)}),flush=True)
  results[split]={name:{'metrics':summary(rr),'per_case':rr} for name,rr in cases.items()}
  assert all(len(rr)==len(native) for rr in cases.values())
  write(R/f'RAW_CLEAN_{split.upper()}.json',{'utc':now(),'split':split,'results':results[split],'seconds':time.monotonic()-tick,'no_test_access':True,'not_used_for_selection':True})
 gaps={name:results['train'][name]['metrics']['scan_dice']-results['validation'][name]['metrics']['scan_dice'] for name in configs}
 assert sha(OUT/'deployment/manifest.json')==manifest_sha
 write(R/'TRAIN_FIT_DIAGNOSTIC.json',{'utc':now(),'status':'complete','configs':configs,'train':{k:v['metrics'] for k,v in results['train'].items()},'validation':{k:v['metrics'] for k,v in results['validation'].items()},'train_minus_validation_scan':gaps,'seconds':time.monotonic()-started,'test_access':False,'no_selection_or_training_changed':True,'interpretation':'clean fitting-gap description only; reused validation and target-selected workflow, not an independent generalization estimate; cannot infer overfitting cause or modality ceiling from gap alone','deployment_manifest_sha256':manifest_sha})
 lines=[]
 for name in configs:
  tr=results['train'][name]['metrics'];va=results['validation'][name]['metrics'];lines.append(f"| {name} | {tr['scan_dice']:.6f} | {va['scan_dice']:.6f} | {gaps[name]:+.6f} |")
 (R/'TRAIN_FIT_DIAGNOSTIC.md').write_text('# 固定模型的无增强训练/验证拟合诊断\n\n最终270配置固定后才执行，不用于选择模型、阈值或后续训练，无新270前向/像素访问。训练972/444与验证108/50均使用原生crop指标及160×208×160训练/验证画布，排除填充；目标测试使用160×208×168画布，勿把诊断验证数值替换原选优记录。\n\n| 固定模型/流程 | 训练Scan Dice | 验证Scan Dice | 训练−验证 |\n|---|---:|---:|---:|\n'+'\n'.join(lines)+'\n\n这是干净输入下的拟合差距描述，不是因果过拟合诊断或T1ce性能上限证明。验证与270均已复用，不能称独立确认。\n\n[机器结果](TRAIN_FIT_DIAGNOSTIC.json) · [访问前固定](TRAIN_FIT_PREACCESS.json)\n')

if __name__=='__main__':main()
