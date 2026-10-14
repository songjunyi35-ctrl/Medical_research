"""Top three candidate models: one target-best checkpoint per training branch."""
import sys,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.t1ce_10h_common import *
from scripts.finalize_t1ce_10h import state_hash

def select():
 verify();reports=[read(p) for p in (R/'reports').glob('*.json') if not p.name.startswith('panel_')]
 reports.sort(key=lambda d:(-d['metrics']['scan_dice'],d['forward_equivalents'],d.get('gate_epoch') or 0,d['id']))
 seen_branches=set();seen_states=set();chosen=[]
 for d in reports:
  branch=d.get('arm') or d['id']
  if branch in seen_branches:continue
  seen_branches.add(branch);signature=(state_hash(ROOT/d['checkpoint_path']),d['mode'])
  if signature in seen_states:continue
  seen_states.add(signature);chosen.append(d)
  if len(chosen)==3:break
 cap=8
 amendment=R/'INFERENCE_COST_AMENDMENT.json'
 if amendment.exists():
  a=read(amendment);assert a['explicit_user_authorization'] is True and a['max_forward_equivalents']==11;cap=11
 recipes=[]
 for count in [1,2,3]:
  for group in itertools.combinations(chosen,count):
   cost=sum(d['forward_equivalents'] for d in group)
   if cost>cap+1e-10:continue
   for threshold in ([.5] if count==3 else [.4,.5,.6]):
    recipes.append({'id':'panel_'+str(len(recipes)+1).zfill(2),'components':[d['id'] for d in group],'threshold':threshold,'forward_equivalents':cost,'formula':'arithmetic mean of native-crop float32 logits, then common threshold'})
 assert len(recipes)<=19
 write(R/'FINAL_PANEL_FREEZE.json',{'utc':now(),'selected':[{'id':d['id'],'report_sha256':sha(R/'reports'/(d['id']+'.json')),'checkpoint_sha256':d['checkpoint_sha256'],'logits_dir':d['logits_dir']} for d in chosen],'recipes':recipes,'compute_cap':cap,'selection':'one target-best snapshot per training branch, then top3 unique model states; Scan then cost then earliest gate','test_tuned':True,'new_training_allowed':False,'cost_amendment_sha256':sha(amendment) if amendment.exists() else None})
 print(json.dumps({'selected':[d['id'] for d in chosen],'recipes':len(recipes),'compute_cap':cap},indent=2))

if __name__=='__main__':select()
