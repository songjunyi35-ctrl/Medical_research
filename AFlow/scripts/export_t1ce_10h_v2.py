"""Repair artifact compilation from completed immutable panel reports only."""
import sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.t1ce_10h_common import *

def export():
 selection=read(R/'FINAL_SELECTION.json');freeze=read(R/'FINAL_PANEL_FREEZE.json');recipes={r['id']:r for r in freeze['recipes']}
 target=OUT/'deployment';target.mkdir(parents=True,exist_ok=True);exports=[]
 for role,ident,filename in [('overall',selection['overall'],'manifest.json'),('best_single',selection['best_single_model'],'best_single_manifest.json')]:
  report=read(R/'reports'/(ident+'.json'));recipe=recipes[ident];assert report['components']==recipe['components'] and report['threshold']==recipe['threshold']
  components=[]
  for source_id in recipe['components']:
   src=read(R/'reports'/(source_id+'.json'));path=ROOT/src['checkpoint_path'];assert sha(path)==src['checkpoint_sha256']
   dest=target/(path.parent.name+'_'+src['checkpoint_sha256'][:16]+'.pth');disk(64<<20)
   if not dest.exists():shutil.copy2(path,dest)
   assert sha(dest)==src['checkpoint_sha256']
   state=torch.load(dest,map_location='cpu',weights_only=False)
   components.append({'source_id':source_id,'checkpoint_path':str(dest.relative_to(ROOT)),'checkpoint_sha256':src['checkpoint_sha256'],'mode':src['mode'],'stored_epoch':state.get('epoch'),'epoch_scope':'additional after '+P['sources'][src['arm']] if src.get('arm') else 'original source run'})
   del state
  manifest={**recipe,'component_ids':recipe['components'],'components':components,'only_external_input':'T1ce','threshold_space':'sigmoid_probability','threshold_raw_logit':math.log(recipe['threshold']/(1-recipe['threshold'])),'no_labels_or_case_lookup':True,'test_tuned':True,'source_report':str((R/'reports'/(ident+'.json')).relative_to(ROOT)),'source_report_sha256':sha(R/'reports'/(ident+'.json')),'role':role}
  write(target/filename,manifest);exports.append({'role':role,'path':str((target/filename).relative_to(ROOT)),'sha256':sha(target/filename)})
 write(R/'EXPORT_REPAIR_RECEIPT.json',{'utc':now(),'status':'passed','exports':exports,'selection_sha256':sha(R/'FINAL_SELECTION.json'),'panel_freeze_sha256':sha(R/'FINAL_PANEL_FREEZE.json'),'new_model_forward':False,'new_scoring_or_selection':False,'reason':'original recipe expansion replaced component dictionaries with string IDs during packaging; all completed raw metrics and logical recipes preserved'})

if __name__=='__main__':export()
