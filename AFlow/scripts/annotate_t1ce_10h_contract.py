"""Clarify probability thresholds and actual forward counts without altering scores."""
import sys,math,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.t1ce_10h_common import *

def main():
 lock=(R/'board.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 try:
  path=ROOT/'AFLOW_BRATS_LEADERBOARD.json';board=read(path)
  for row in board.get('test270_tuned_native_rank_by_scan',[]):
   d=read(ROOT/row['report_path']);components=[read(R/'reports'/(x+'.json')) for x in d['components']] if d.get('components') else [d]
   row.update(threshold_space='sigmoid_probability',threshold_probability=d['threshold'],threshold_raw_logit=math.log(d['threshold']/(1-d['threshold'])),comparison_operator='>',input_modality='T1ce',input_mode=d['mode'],ensemble_size=len(components),locator_forwards=sum(8 if c['mode']=='patch' else 1 for c in components),uses_GT_for_localization=False,model_precision='bfloat16',fusion_precision='float32')
  board['updated_utc']=now();write(path,board)
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 write(R/'EVALUATION_CONTRACT.json',{'utc':now(),'threshold_space':'sigmoid_probability','threshold_rule':'raw logit > log(p/(1-p)); p in {0.4,0.5,0.6}','point_rule':'argmax of native-crop fused raw logits; independent of binary threshold','compute_unit':'one4,700,897-parameter SegResNet128^3 forward; native160x208x168 costs2.666015625 equivalents; patch costs8','actual_forward_counts_recorded_separately':True,'fusion':'native float32 raw logits arithmetic mean; 128 logits trilinear align_corners=False to native crop','scoring':'WT1/2/3 excluding4; native crop; padding excluded','new_inference':False,'changed_metrics_or_selection':False})

if __name__=='__main__':main()
