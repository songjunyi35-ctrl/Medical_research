"""Posthoc saved-point/GT verification and failure description; zero model forwards."""
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import numpy as np
from scipy.ndimage import distance_transform_edt
from scripts import structured_background_protocol as p

RECORD=p.ROOT/'archive/research_records/2026-09-10_subject_fixed_flip8'


def main():
    output=RECORD/'POINT_AUDIT.json'
    if output.exists():raise FileExistsError(output)
    raw_path=RECORD/'RAW_DEVELOPMENT.json';raw=json.loads(raw_path.read_text());v=json.loads((RECORD/'VERIFICATION_RECEIPT.json').read_text());p.checked(raw_path,v['raw_sha256'])
    _,dev=p.check_roles();aux=json.loads((p.RECORD/'AUX_CACHE_RECEIPT.json').read_text());p.checked(p.CACHE/'labels.u8',aux['source_labels_sha256'])
    cache=json.loads((p.CACHE/'index.json').read_text());slots={case:i for i,case in enumerate(cache['case_ids'])}
    labels=np.memmap(p.CACHE/'labels.u8',mode='r',dtype=np.uint8,shape=(972,128,128,128));results={};verified=0
    for seed in p.SEEDS:
        pair=raw['results'][str(seed)];transitions=[];miss_distances=[]
        for meta,a,b in zip(dev,pair['identity']['per_case'],pair['flip8']['per_case'],strict=True):
            assert meta['case_id']==a['case_id']==b['case_id']
            truth=np.array(labels[slots[meta['case_id']]],dtype=bool)
            for row in (a,b):
                assert int(truth.sum())==row['target_voxels'] and bool(truth.reshape(-1)[row['point_flat_index']])==row['point_hit'];verified+=1
            old=np.array(np.unravel_index(a['point_flat_index'],truth.shape));new=np.array(np.unravel_index(b['point_flat_index'],truth.shape))
            distance=0.
            if not b['point_hit']:
                distance=float(distance_transform_edt(~truth)[tuple(new)]);miss_distances.append(distance)
            kind='lost' if a['point_hit'] and not b['point_hit'] else 'gained' if b['point_hit'] and not a['point_hit'] else 'both_hit' if a['point_hit'] else 'both_miss'
            transitions.append({'case_id':meta['case_id'],'subject_id':meta['subject_id'],'transition':kind,'identity_point':old.tolist(),'flip8_point':new.tolist(),'point_shift_input_voxels':float(np.linalg.norm(new-old)),'flip8_point_distance_to_gt_input_voxels':distance,'dice_delta':b['dice']-a['dice']})
        lost=[r for r in transitions if r['transition']=='lost'];gained=[r for r in transitions if r['transition']=='gained']
        results[str(seed)]={'lost':len(lost),'gained':len(gained),'net':len(gained)-len(lost),'lost_point_cases_with_dice_improvement':sum(r['dice_delta']>0 for r in lost),'lost_point_distance_voxels':{'minimum':min((r['flip8_point_distance_to_gt_input_voxels'] for r in lost),default=None),'maximum':max((r['flip8_point_distance_to_gt_input_voxels'] for r in lost),default=None),'median':statistics.median([r['flip8_point_distance_to_gt_input_voxels'] for r in lost]) if lost else None},'flip8_miss_count':len(miss_distances),'flip8_misses_within_one_input_voxel':sum(d<=1 for d in miss_distances),'per_case':transitions}
        assert results[str(seed)]['net']==raw['comparisons'][str(seed)]['point_delta']
    p.write_new(output,{'status':'passed_saved_point_truth_verification','created_utc':datetime.now(timezone.utc).isoformat(),'points_checked':verified,'results':results,'raw_sha256':p.sha(raw_path),'labels_sha256':aux['source_labels_sha256'],'script_sha256':p.sha(Path(__file__)),'new_model_forwards':0,'access':'reused_dev180_binary_labels_and_saved_predictions_only','distance_units':'128-grid input voxels, not mm','posthoc_not_a_new_hypothesis_test':True,'no_rule_or_gate_change':True})
    print(json.dumps({'points_checked':verified,'results':{s:{k:v for k,v in r.items() if k!='per_case'} for s,r in results.items()}},indent=2),flush=True)


if __name__=='__main__':main()
