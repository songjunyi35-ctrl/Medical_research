"""Explicit test access ledger; fixed references and budgeted candidate gates."""
import sys,fcntl,time,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.t1ce_10h_common import *

def register_report(report_path):
 # Separate target-tuned ranking; historical rankings are never re-sorted here.
 d=read(report_path);assert d['split']=='test'
 lock=(R/'board.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 try:
  p=ROOT/'AFLOW_BRATS_LEADERBOARD.json';board=read(p);key='test270_tuned_native_rank_by_scan'
  old=[r for r in board.get(key,[]) if r['method']!=d['id']];m=d['metrics']
  row={'method':d['id'],'scan_dice':m['scan_dice'],'subject_dice':m['subject_dice'],'point_hit':m['point_hit'],'point_hits':m['point_hits'],'point_denominator':270,'cohort':{'scans':270,'subjects':119},'checkpoint_path':d['checkpoint_path'],'checkpoint_sha256':d['checkpoint_sha256'],'grid':'native 1mm crop; padding excluded','threshold':d['threshold'],'forward_equivalents':d['forward_equivalents'],'report_path':str(Path(report_path).relative_to(ROOT)),'report_sha256':sha(report_path),'evidence_status':'test-tuned; reused internal; not independent confirmation','test_tuned':True,'scan_ci95':m['scan_ci95'],'subject_ci95':m['subject_ci95'],'role':d.get('role','candidate'),'logit_formula':d.get('logit_formula','single model')}
  ranked=old+[row];ranked.sort(key=lambda r:(-r['scan_dice'],r['forward_equivalents'],r['method']))
  for i,r in enumerate(ranked,1):r['rank']=i
  board[key]=ranked;board['updated_utc']=now();write(p,board)
  md=ROOT/'AFLOW_BRATS_LEADERBOARD.md';s=md.read_text();a='<!-- TEST270_TUNED_NATIVE -->';b='<!-- END_TEST270_TUNED_NATIVE -->'
  table='\n'.join(f"| {r['rank']} | {r['method']} | {r['scan_dice']:.9f} | {r['subject_dice']:.9f} | {r['point_hits']}/270 | {r['forward_equivalents']:.3f}× | [报告]({r['report_path']}) |" for r in ranked)
  block=a+'\n## 270例定向优化：原始栅格，test-tuned，独立分榜\n\n本榜所有选择使用复用270，非独立泛化证据；包含固定参考及全部已完成门审/组合，历史传统排名不变。统一原始1mm crop，排除填充；128³ logits三线性回原栅格。\n\n| 排名 | 方法/快照 | Scan Dice | Subject Dice | Point | 128³前向等价量 | 证据 |\n|---:|---|---:|---:|---:|---:|---|\n'+table+'\n'+b
  if a in s:s=s[:s.index(a)]+block+s[s.index(b)+len(b):]
  else:s+='\n\n'+block+'\n'
  tmp=md.with_suffix('.md.tmp');tmp.write_text(s);os.replace(tmp,md)
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

def score_checkpoint(path,mode,ident,role='candidate',arm=None,gate_epoch=None):
 verify();path=Path(path);h=sha(path);directory=R/'reports';directory.mkdir(exist_ok=True)
 report_path=directory/(ident+'.json')
 if report_path.exists():
  d=read(report_path);assert d['checkpoint_sha256']==h and d['mode']==mode
  register_report(report_path);return d
 rec=R/'access'/(ident+'.json')
 if rec.exists():raise RuntimeError('incomplete prior model access; requires explicit recovery receipt: '+ident)
 if arm:
  completed=[read(p) for p in directory.glob(arm+'_*.json')]
  for old in completed:
   if old['checkpoint_sha256']==h and old['mode']==mode:return old
  attempted=list((R/'access').glob(arm+'_*.json'))
  if len(attempted)>=4:raise RuntimeError('candidate test access budget exhausted')
 write(rec,{'utc':now(),'id':ident,'checkpoint_sha256':h,'mode':mode,'arm':arm,'gate_epoch':gate_epoch,'test_tuned':True,'test_pixels_access':True,'segmentation_training_uses_test_labels':False,'freeze_sha256':sha(R/'FREEZE.json')})
 device=torch.device('cuda:0');torch.cuda.set_device(device);torch.set_num_threads(2);seed(SEED)
 model=model_for(device,path);logdir=SCRATCH/'logits'/ident;disk(4*2**30)
 cases,seconds=evaluate(model,mode,'test',logit_dir=logdir)
 assert len(cases)==270 and len({c['subject_id'] for c in cases})==119
 assert {c['case_id'] for c in cases}=={r['case_id'] for r in rows('test')}
 order={r['case_id']:i for i,r in enumerate(rows('test'))};cases.sort(key=lambda x:order[x['case_id']])
 m=summary(cases);d={'id':ident,'utc':now(),'split':'test','role':role,'arm':arm,'gate_epoch':gate_epoch,'mode':mode,'threshold':.5,'metrics':m,'per_case':cases,'checkpoint_path':str(path.relative_to(ROOT)),'checkpoint_sha256':h,'forward_equivalents':8 if mode=='patch' else (160*208*168/128**3 if mode=='native' else 1),'inference_seconds':seconds,'logits_dir':str(logdir),'test_tuned':True,'new_outer_confirmation_access':False,'access_receipt_sha256':sha(rec),'grid':'native1mm crop'}
 if (R/'REFERENCE_PANEL.json').exists():
  ref=read(R/'REFERENCE_PANEL.json');base=read(ROOT/ref['strongest_report']);pure=read(ROOT/ref['pure193_report'])
  d['vs_strongest_reference']=compare(base['per_case'],cases);d['vs_fixed193']=compare(pure['per_case'],cases)
 write(report_path,d);register_report(report_path);del model;torch.cuda.empty_cache();return d

def references():
 verify();numeric();results=[]
 for name in SOURCES:
  d=score_checkpoint(source(name),SOURCES[name][2],'ref_'+name,role='frozen_reference');results.append(d)
 strongest=max(results,key=lambda d:d['metrics']['scan_dice']);pure=next(d for d in results if d['id']=='ref_pure193')
 panel={'utc':now(),'strongest_id':strongest['id'],'strongest_scan':strongest['metrics']['scan_dice'],'strongest_report':str((R/'reports'/(strongest['id']+'.json')).relative_to(ROOT)),'pure193_report':str((R/'reports'/'ref_pure193.json').relative_to(ROOT)),'references':[{k:d[k] for k in ['id','metrics','checkpoint_sha256','mode']} for d in results],'test_tuned':True,'selected_before_new_training':True}
 write(R/'REFERENCE_PANEL.json',panel)
 c=strongest['per_case'];fnr=float(np.mean([r['fnr'] for r in c]));fdr=float(np.mean([r['fdr'] for r in c]))
 write(R/'DIAGNOSTIC.json',{'utc':now(),'source':strongest['id'],'mean_fnr':fnr,'mean_fdr':fdr,'difference':fnr-fdr,'enable_C':fnr-fdr>=.05,'rule':'mean FNR minus mean FDR >=0.05 on frozen strongest reference','low_dice_cases':sorted(c,key=lambda r:r['locator_dsc'])[:30],'test_tuned':True,'diagnostic_is_not_proof_of_cause':True})
 write(R/'REFERENCE_READY.json',{'utc':now(),'reference_panel_sha256':sha(R/'REFERENCE_PANEL.json'),'diagnostic_sha256':sha(R/'DIAGNOSTIC.json')})

if __name__=='__main__':
 try:
  if sys.argv[1]=='references':references()
  else:raise ValueError('references')
 except BaseException:
  import traceback
  write(R/f'EVALUATION_FAILURE_{int(time.time())}.json',{'utc':now(),'traceback':traceback.format_exc()});raise
