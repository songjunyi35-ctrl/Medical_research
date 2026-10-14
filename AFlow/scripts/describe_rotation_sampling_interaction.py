"""Posthoc saved-score factorial description; no new forward or promotion gate."""
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
from scripts import small_rotation_protocol as p
from scripts.require_development_mask_verification import require_masks


def interaction(a,b,c,d):
    """(subject+rotation - subject) - (native+rotation - native)."""
    return (d-c)-(b-a)


def main():
    rec=p.RECORD;require_masks(rec)
    if (rec/'SAMPLING_INTERACTION.json').exists():raise FileExistsError('already described')
    result=json.loads((rec/'RESULT.json').read_text());raw=json.loads((rec/'RAW_DEVELOPMENT.json').read_text());pre=json.loads((rec/'DEV_PREACCESS.json').read_text())
    sources={}
    for key,info in pre['source_raws'].items():p.checked(info['path'],info['sha256']);sources[key]=json.loads(Path(info['path']).read_text())
    summaries={};records=[]
    for seed in p.SEEDS:
        a=sources['native']['results'][f'seed{seed}_control']['per_case']
        c=sources['subject']['results'][str(seed)]['candidate']['per_case']
        b=raw['results'][f'{seed}_native_rotation']['per_case'];d=raw['results'][f'{seed}_subject_rotation']['per_case']
        rows=[];groups=defaultdict(list)
        for aa,bb,cc,dd in zip(a,b,c,d,strict=True):
            assert len({(r['case_id'],r['subject_id']) for r in (aa,bb,cc,dd)})==1
            value=interaction(aa['dice'],bb['dice'],cc['dice'],dd['dice'])
            point=interaction(int(aa['point_hit']),int(bb['point_hit']),int(cc['point_hit']),int(dd['point_hit']))
            row={'seed':seed,'case_id':aa['case_id'],'subject_id':aa['subject_id'],'native':aa['dice'],'native_rotation':bb['dice'],'subject':cc['dice'],'subject_rotation':dd['dice'],'dice_interaction':value,'point_interaction':point}
            rows.append(row);groups[aa['subject_id']].append(value)
        scan=statistics.fmean(r['dice_interaction'] for r in rows);subject=statistics.fmean(statistics.fmean(v) for v in groups.values());point=sum(r['point_interaction'] for r in rows)
        native=result['comparisons'][f'{seed}_native_rotation']['versus_own_sampling_control'];sample=result['comparisons'][f'{seed}_subject_rotation']['versus_own_sampling_control']
        assert abs(scan-(sample['scan_delta']-native['scan_delta']))<1e-12
        assert abs(subject-(sample['subject_delta']-native['subject_delta']))<1e-12
        assert point==sample['point_delta']-native['point_delta']
        summaries[str(seed)]={'scan_dice_interaction':scan,'subject_dice_interaction':subject,'point_count_interaction':point,'negative_scan_interaction_cases':sum(r['dice_interaction']<0 for r in rows),'negative_subject_mean_interaction_patients':sum(statistics.fmean(v)<0 for v in groups.values()),'scans':180,'subjects':78}
        records.extend(rows)
    report={'status':'posthoc_descriptive_only','created_utc':datetime.now(timezone.utc).isoformat(),'formula':'(subject_rotation-subject_control)-(native_rotation-native_control)','summary':summaries,'per_case':records,'source_raws':pre['source_raws'],'candidate_raw_sha256':p.sha(rec/'RAW_DEVELOPMENT.json'),'result_sha256':p.sha(rec/'RESULT.json'),'new_forwards':0,'new_hypothesis_test_or_promotion_gate':False,'confidence_intervals':None,'same_development_patients_reused':True,'code_sha256':p.sha(__file__)}
    p.write_new(rec/'SAMPLING_INTERACTION.json',report)
    lines=['# 患者采样与旋转：已有结果的事后交互描述','', '计算（患者采样+旋转−患者采样对照）−（原采样+旋转−原采样对照）。若为0，两个因素在该平均指标上可按简单相加描述；负值表示旋转在患者采样条件下的收益更差。仅复用已核验的逐例分数，不增加模型前向，不改变原门，不作新的显著性主张。','', '| seed | Scan Dice交互 | Subject Dice交互 | Point计数交互 | 负交互扫描/180 | 负交互患者/78 |','|---|---:|---:|---:|---:|---:|']
    for seed,s in summaries.items():lines.append(f"| {seed} | {s['scan_dice_interaction']:+.6f} | {s['subject_dice_interaction']:+.6f} | {s['point_count_interaction']:+d} | {s['negative_scan_interaction_cases']} | {s['negative_subject_mean_interaction_patients']} |")
    lines+=['','Dice交互均值在两seed都为负，但Point交互符号不一致。它说明不能把单独设置的表现简单相加来预测组合效果；不证明所有采样或增强组合都不相容。没有新增置信区间或检验，两seed仍是同一78人，事后描述不能替代预注册结论。','', '[逐例与来源SHA](SAMPLING_INTERACTION.json)、[原正式结果](RESULT.json)。']
    with (rec/'SAMPLING_INTERACTION.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
