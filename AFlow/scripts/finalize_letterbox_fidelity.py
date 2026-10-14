"""Independent count arithmetic and comparison with historical native baseline."""
from collections import defaultdict
from datetime import datetime,timezone
import json
import statistics
import numpy as np
from scripts import structured_background_protocol as p
from scripts.publish_research_card import publish

REC=p.ROOT/'archive/research_records/2026-09-10_letterbox_fidelity'


def main():
    if (REC/'RESULT.json').exists():raise FileExistsError('already finalized')
    pre=json.loads((REC/'PREACCESS.json').read_text());done=json.loads((REC/'RUN_DONE.json').read_text())
    p.checked(REC/'PREACCESS.json',done['preaccess_sha256']);p.checked(REC/'RAW_CASES.jsonl',done['raw_sha256'])
    for n,d in pre['files'].items():p.checked(p.ROOT/n,d)
    p.checked(REC/'IDEA.md',pre['idea_sha256'])
    rows=[json.loads(s) for s in (REC/'RAW_CASES.jsonl').read_text().splitlines()]
    oldpath=p.ROOT/'archive/research_records/2026-09-08_fit_spatial_roundtrip_v2/RAW_CASES.jsonl'
    historical={r['case_id']:r for r in [json.loads(s) for s in oldpath.read_text().splitlines()]}
    fit,_=p.check_roles();assert [(r['case_id'],r['subject_id']) for r in rows]==[(r['case_id'],r['subject_id']) for r in fit]
    metrics={};newlost=[]
    for mode in ('direct128','letterbox128'):
        groups=defaultdict(list)
        for r in rows:
            m=r['metrics'][mode]
            assert m['dice']==2*m['intersection']/(m['target_voxels']+m['reconstructed_voxels'])
            assert m['false_negative_voxels']==m['target_voxels']-m['intersection']
            assert m['false_positive_voxels']==m['reconstructed_voxels']-m['intersection']
            assert r['source_label_sha256']==pre['source_label_hashes'][r['case_id']]
            groups[r['subject_id']].append(m['dice'])
            if mode=='direct128':
                old=historical[r['case_id']]['restorations']['nearest-exact']
                assert all(m[k]==old[k] for k in m),'historical native reconstruction must match exactly'
        metrics[mode]={'scan_dice':statistics.fmean(r['metrics'][mode]['dice'] for r in rows),'subject_dice':statistics.fmean(statistics.fmean(v) for v in groups.values()),'empty_native':sum(r['metrics'][mode]['empty_prediction'] for r in rows)}
    for r in rows:
        for c in r['components']:
            if c['old_sampled_voxels']>0 and c['letterbox_sampled_voxels']==0 and c['native_voxels']>125:newlost.append({'case_id':r['case_id'],**c})
    component_metrics={key:{'lost_components':sum(c['crop_voxels']>0 and c[field]==0 for r in rows for c in r['components']),'lost_native_voxels':sum(c['native_voxels'] for r in rows for c in r['components'] if c['crop_voxels']>0 and c[field]==0)} for key,field in [('direct128','old_sampled_voxels'),('letterbox128','letterbox_sampled_voxels')]}
    blocked=bool(newlost) or any(r['letterbox_empty_coarse'] for r in rows)
    result={'status':'fit_only_diagnostic_completed','created_utc':datetime.now(timezone.utc).isoformat(),'scans':684,'subjects':313,'metrics':metrics,'components':component_metrics,'new_missing_over125_components':newlost,'coarse_empty_scans':sum(r['letterbox_empty_coarse'] for r in rows),'median_padding_fraction':statistics.median(r['padding_fraction'] for r in rows),'crop_lost_voxels':sum(r['crop_lost_voxels'] for r in rows),'engineering_training_veto':blocked,'training_admitted':False,'raw_sha256':p.sha(REC/'RAW_CASES.jsonl'),'preaccess_sha256':p.sha(REC/'PREACCESS.json'),'historical_raw_sha256':p.sha(oldpath),'seconds':done['seconds'],'new_forwards':0,'access':{'fit_labels':True,'dev_pixels':False,'test':False,'outer':False,'confirmation':False},'limits':'saved count arithmetic and exact historical baseline matched; not independent reexecution of every new mapping; label roundtrip is not model Dice or accuracy upper bound'}
    p.write_new(REC/'RESULT.json',result)
    p.write_new(REC/'VERIFICATION.json',{'status':'passed_declared_count_and_historical_baseline_checks','result_sha256':p.sha(REC/'RESULT.json'),'verifier_sha256':p.sha(__file__),'cases_verified':684,'historical_direct128_counts_hashes_exact':True,'new_mapping_not_independently_reexecuted':True})
    delta=metrics['letterbox128']['scan_dice']-metrics['direct128']['scan_dice']
    text=f'''## 保长宽比补边：fit标签保真诊断 ✅ 完成，未准入训练

**一句话：** 保住照片形状会减少短轴像素，已量化这项取舍。

**出发点：** 原脑裁剪三轴尺度差中位1.2879；旧标签往返0.965432不是模型上限。

**具体做法：** 唯一变化为最长轴128、短轴同比取整/居中补边；原生WT、image-only裁剪和最近邻前后映射不变，两个方案回相同完整原生标签评分。

**结果：** fit684/313标签Scan往返Direct={metrics['direct128']['scan_dice']:.9f}，补边={metrics['letterbox128']['scan_dice']:.9f}，差{delta:+.9f}；Subject分别{metrics['direct128']['subject_dice']:.9f}/{metrics['letterbox128']['subject_dice']:.9f}。补边缺失组件{component_metrics['letterbox128']['lost_components']}（原{component_metrics['direct128']['lost_components']}）；新增>125体素缺失组件{len(newlost)}，整WT空{result['coarse_empty_scans']}，中位补边比例{result['median_padding_fraction']:.2%}。未运行模型/开发，无置信区间。

**是否支持直觉：** 支持可以保近似长宽比的几何事实，但标签保真差{delta:+.6f}揭示分辨率代价，不证明模型会升或降。固定工程否决门={blocked}；即使不否决也不自动批准训练。原crop外{result['crop_lost_voxels']}体素保留记录、不排例。

**代价/下一步：** {done['seconds']:.1f}秒/4 CPU，0 GPU/模型前向、无新像素缓存。暂不挤占旋转四模型；新训练仍须共同网格评测与独立预算。[七问](archive/research_records/2026-09-10_letterbox_fidelity/IDEA.md)、[结果](archive/research_records/2026-09-10_letterbox_fidelity/RESULT.json)、[原始逐例](archive/research_records/2026-09-10_letterbox_fidelity/RAW_CASES.jsonl)、[限定核验](archive/research_records/2026-09-10_letterbox_fidelity/VERIFICATION.json)。
'''
    with (REC/'RESULT.md').open('x') as f:f.write(text)
    (REC/'CARD.md').write_text(text)
    publish(str(REC.relative_to(p.ROOT)))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
