"""Register one audited current-study candidate without deleting historical rows."""
import argparse,copy,re
from datetime import datetime,timezone
from scripts.segresnet_auto_inference12h_20261028.common import *


def main(candidate):
    folder=R/'candidates'/candidate;result=read(folder/'RESULT.json');assert result['all270_masks_independently_verified']
    metrics=result['metrics'];board=read(ROOT/'AFLOW_BRATS_LEADERBOARD.json');key='test270_tuned_native_rank_by_scan';rows=board[key]
    if any(r['method']==result['method'] for r in rows):return
    row=copy.deepcopy(next(v for v in rows if v['method']=='segresnet_w32_e171_mirror5_volumeT2_test270_20261028'))
    for field in ['paired_comparisons','inference_validation_utility_gate','views']:row.pop(field,None)
    cost=result.get('locator_forwards',5)
    row.update(method=result['method'],scan_dice=metrics['scan_dice'],subject_dice=metrics['subject_dice'],point_hits=metrics['point_hits'],
      point_hit=metrics['point_hits']/270,scan_ci95=metrics['scan_ci95'],subject_ci95=metrics['subject_ci95'],locator_forwards=cost,
      forward_equivalents=cost*(160*208*168)/128**3,current_test_retuning=True,postprocessing=True,
      evidence_status='interim: globally selected directly on reused270; independently reconstructed search masks; post-selection test-tuned descriptive evidence',
      research_continues=True,threshold_space='selected global inference decoder; see frozen recipe',comparison_operator='see recipe',
      logit_formula='same fixed checkpoint; globally selected aggregation and decoder in recipe',
      forward_cost_note='full-canvas voxel equivalence per128^3; not FLOPs/latency; ROI counts include shared base once',
      paired_comparisons={'study_start_mirror5':result['comparison_to_previous_incumbent']})
    for kind,name in [('report','RAW_TEST270.json'),('result','RESULT.json'),('selected_recipe','RECIPE.json')]:
        p=folder/name;row[kind+'_path']=str(p.relative_to(ROOT));row[kind+'_sha256']=sha(p)
    rows.append(row);rows.sort(key=lambda r:(-r['scan_dice'],-r['subject_dice'],r['method']))
    for i,v in enumerate(rows,1):v['rank']=i
    board['latest_segresnet_auto12h_interim_20261028']=dict(method=row['method'],scan_dice=row['scan_dice'],test_tuned=True,research_continues=True,result_path=row['result_path'])
    board['updated_utc']=datetime.now(timezone.utc).isoformat()
    write(ROOT/'AFLOW_BRATS_LEADERBOARD.json',board)
    path=ROOT/'AFLOW_BRATS_LEADERBOARD.md';text=path.read_text();a=text.index('<!-- TEST270_TUNED_NATIVE -->');b=text.index('<!-- END_TEST270_TUNED_NATIVE -->',a);block=text[a:b]
    first=next(re.finditer(r'^\| 1 \|',block,re.M)).start()
    table='\n'.join(f"| {v['rank']} | {v['method']} | {v['scan_dice']:.9f} | {v['subject_dice']:.9f} | {v['point_hits']}/270 | {v['forward_equivalents']:.3f}× | [报告]({v['report_path']}) |" for v in rows)+'\n'
    path.write_text(text[:a]+block[:first]+table+'\n'+text[b:])
    from scripts.verify_aflow_brats_leaderboard import validate_ranked_rows
    validate_ranked_rows(key,rows);print(dict(method=row['method'],rank=row['rank'],native_rows=len(rows)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);a=p.parse_args();main(a.candidate)
