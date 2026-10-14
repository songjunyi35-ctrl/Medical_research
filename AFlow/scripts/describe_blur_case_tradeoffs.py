"""Saved-count, scan-level descriptions; not new evaluation or model selection."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
from scripts import image_blur_protocol as p
from scripts.require_development_mask_verification import require_masks


def category(delta_tp,delta_fp):
    if delta_tp==0 and delta_fp==0:return 'count_tie'
    if delta_tp>=0 and delta_fp<=0:return 'pareto_better'
    if delta_tp<=0 and delta_fp>=0:return 'pareto_worse'
    if delta_tp>0 and delta_fp>0:return 'expansion'
    if delta_tp<0 and delta_fp<0:return 'contraction'
    raise AssertionError('unclassified count difference')


def main():
    rec=p.RECORD;require_masks(rec)
    if (rec/'SCAN_TRADEOFFS.json').exists():raise FileExistsError('already described')
    raw=json.loads((rec/'RAW_DEVELOPMENT.json').read_text());pre=json.loads((rec/'DEV_PREACCESS.json').read_text())
    old_info=pre['source_raws']['subject'];p.checked(old_info['path'],old_info['sha256']);old=json.loads(Path(old_info['path']).read_text())
    result={};all_rows=[]
    for seed in p.SEEDS:
        rows=raw['results'][f'{seed}_subject_blur']['per_case'];control=old['results'][str(seed)]['candidate']['per_case'];counts=Counter();signs={};changes=[]
        for a,b in zip(control,rows,strict=True):
            assert (a['case_id'],a['subject_id'],a['target_voxels'])==(b['case_id'],b['subject_id'],b['target_voxels'])
            dtp=b['intersection']-a['intersection'];dfp=(b['predicted_voxels']-b['intersection'])-(a['predicted_voxels']-a['intersection'])
            group=category(dtp,dfp);dd=b['dice']-a['dice'];counts[group]+=1
            if group=='pareto_better':assert dd>=-1e-12
            if group=='pareto_worse':assert dd<=1e-12
            signs.setdefault(group,Counter())['dice_up' if dd>0 else 'dice_down' if dd<0 else 'dice_tie']+=1
            record={'seed':seed,'case_id':a['case_id'],'subject_id':a['subject_id'],'delta_true_positive_voxels':dtp,'delta_false_positive_voxels':dfp,'delta_dice':dd,'category':group,'point_hit_delta':int(b['point_hit'])-int(a['point_hit'])}
            changes.append(record);all_rows.append(record)
        assert sum(counts.values())==180
        result[str(seed)]={'counts':dict(counts),'dice_directions_by_category':{k:dict(v) for k,v in signs.items()},'pooled_true_positive_change':sum(r['delta_true_positive_voxels'] for r in changes),'pooled_false_positive_change':sum(r['delta_false_positive_voxels'] for r in changes)}
    report={'status':'posthoc_saved_count_description','created_utc':datetime.now(timezone.utc).isoformat(),'results':result,'per_case':all_rows,'candidate_raw_sha256':p.sha(rec/'RAW_DEVELOPMENT.json'),'control_raw':old_info,'new_forwards':0,'confidence_intervals':None,'promotion_or_parameters_changed':False,'count_tie_not_mask_identity':True,'code_sha256':p.sha(__file__)}
    p.write_new(rec/'SCAN_TRADEOFFS.json',report)
    lines=['# 模糊误差解释：逐扫描而非仅总体体素','', '仅使用已经验证的180例逐扫描计数。真阳性增加且假阳性减少称计数改善；反向称计数恶化；两者都增加称扩张、都减少称收缩。后两类未必改善Dice；计数相同也不证明掩膜形状相同。没有新模型前向、统计检验或晋级门。','', '| seed | 计数改善 | 计数恶化 | 扩张 | 收缩 | 计数相同 |','|---|---:|---:|---:|---:|---:|']
    for seed,value in result.items():
        c=value['counts'];lines.append(f"| {seed} | {c.get('pareto_better',0)} | {c.get('pareto_worse',0)} | {c.get('expansion',0)} | {c.get('contraction',0)} | {c.get('count_tie',0)} |")
    lines+=['', '类别内Dice改善/下降数量和逐例数据见机器报告。总体体素趋势不能代替所有病例的共同趋势，也不能据此重新挑例、调阈值或改变原负结果。','', '[计数与逐例证据](SCAN_TRADEOFFS.json)、[正式结果](RESULT.json)。']
    with (rec/'SCAN_TRADEOFFS.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
