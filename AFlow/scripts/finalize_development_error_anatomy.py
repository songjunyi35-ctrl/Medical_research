"""Independent saved-mask bincount verification and descriptive aggregation."""
from collections import defaultdict,Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import numpy as np
from sammed3d.packed_prediction_masks import read_packed_case
from scripts import structured_background_protocol as p
from scripts.publish_research_card import publish

REC=p.ROOT/'archive/research_records/2026-09-10_development_error_anatomy'


def main():
    if (REC/'RESULT.json').exists():raise FileExistsError('already finalized')
    raw=json.loads((REC/'RAW_DEVELOPMENT.json').read_text());assert raw['status']=='completed'
    p.checked(REC/'DEV_PREACCESS.json',raw['dev_preaccess_sha256']);pre=json.loads((REC/'DEV_PREACCESS.json').read_text())
    p.checked(REC/'CACHE_PREACCESS.json',pre['cache_preaccess_sha256']);bound=json.loads((REC/'CACHE_PREACCESS.json').read_text())
    for n,d in bound['files'].items():p.checked(p.ROOT/n,d)
    p.checked(REC/'CLASS_CACHE.json',pre['class_cache_receipt_sha256']);cache=json.loads((REC/'CLASS_CACHE.json').read_text());p.checked(Path(cache['path']),cache['sha256'])
    labels=np.memmap(cache['path'],mode='r',dtype=np.uint8,shape=(180,128,128,128))
    p.checked(REC/'MASK_INDEX.json',raw['mask_index_sha256']);masks=json.loads((REC/'MASK_INDEX.json').read_text())
    for item in masks['files'].values():p.checked(Path(item['path']),item['sha256'])
    _,dev=p.check_roles();summary={}
    for key,rows in raw['results'].items():
        assert [(r['case_id'],r['subject_id']) for r in rows]==[(r['case_id'],r['subject_id']) for r in dev]
        for kind,info in pre['models'][key].items():p.checked(info['path'],info['sha256'])
        grouped=defaultdict(list)
        for i,row in enumerate(rows):
            pred=read_packed_case(masks,key,i);target=np.asarray(labels[i]);total=np.bincount(target.ravel(),minlength=5);pos=np.bincount(target[pred],minlength=5);c=row['counts']
            assert c['class_target_voxels']=={str(k):int(total[k]) for k in range(5)}
            assert c['class_predicted_voxels']=={str(k):int(pos[k]) for k in range(5)}
            assert c['false_negatives_by_class']=={str(k):int(total[k]-pos[k]) for k in [1,2,3]}
            assert c['false_positives_by_class']=={str(k):int(pos[k]) for k in [0,4]}
            assert c['point_class']==int(target.reshape(-1)[row['point_flat_index']])
            assert c['point_hit']==(c['point_class'] in [1,2,3])
            assert c['intersection']==int(pos[1:4].sum()) and c['target_voxels']==int(total[1:4].sum()) and c['predicted_voxels']==int(pred.sum())
            assert row['dice']==2*c['intersection']/(c['target_voxels']+c['predicted_voxels'])
            grouped[row['subject_id']].append(row)
        totals={str(k):sum(r['counts']['class_target_voxels'][str(k)] for r in rows) for k in range(5)}
        positives={str(k):sum(r['counts']['class_predicted_voxels'][str(k)] for r in rows) for k in range(5)}
        fn={str(k):totals[str(k)]-positives[str(k)] for k in [1,2,3]};fp={str(k):positives[str(k)] for k in [0,4]}
        rates={}
        for k in [1,2,3]:
            name=str(k);patient_rates=[]
            for subject_rows in grouped.values():
                total=sum(r['counts']['class_target_voxels'][name] for r in subject_rows)
                if total:patient_rates.append(sum(r['counts']['class_predicted_voxels'][name] for r in subject_rows)/total)
            rates[name]={'pooled_voxel_recall':positives[name]/totals[name] if totals[name] else None,'mean_patient_recall_within_patient_voxel_weighted':statistics.fmean(patient_rates) if patient_rates else None,'subjects_present':len(patient_rates),'false_negative_share':fn[name]/sum(fn.values())}
        summary[key]={'scan_dice':statistics.fmean(r['dice'] for r in rows),'subject_dice':statistics.fmean(statistics.fmean(r['dice'] for r in sr) for sr in grouped.values()),'point_hits':sum(r['counts']['point_hit'] for r in rows),'class_target_voxels':totals,'class_predicted_voxels':positives,'false_negatives_by_class':fn,'false_positives_by_class':fp,'class_recall':rates,'rc_share_of_false_positive_voxels':fp['4']/sum(fp.values()),'rc_false_positive_rate':fp['4']/totals['4'],'point_class_histogram':dict(Counter(str(r['counts']['point_class']) for r in rows))}
    result={'status':'completed_descriptive_reused_development','created_utc':datetime.now(timezone.utc).isoformat(),'cohort':{'scans':180,'subjects':78},'summary':summary,'raw_sha256':p.sha(REC/'RAW_DEVELOPMENT.json'),'preaccess_sha256':p.sha(REC/'DEV_PREACCESS.json'),'class_cache_sha256':cache['sha256'],'mask_index_sha256':p.sha(REC/'MASK_INDEX.json'),'new_forwards':raw['new_forwards'],'seconds':raw['seconds'],'peak_bytes':raw['peak_bytes'],'training_changed':False,'test_access':False,'new_candidate_checkpoint_loaded':False,'confidence_intervals':None,'limits':'descriptive posthoc decomposition of already revealed old models; 128 grid; no causal claim, model upper bound, or promotion; patient class recall uses within-patient pooled voxels before patient averaging'}
    p.write_new(REC/'RESULT.json',result);p.write_new(REC/'VERIFICATION.json',{'status':'passed_saved_masks_counts_and_points','result_sha256':p.sha(REC/'RESULT.json'),'raw_sha256':p.sha(REC/'RAW_DEVELOPMENT.json'),'verified_masks':720,'argmax_not_independently_recomputed':True,'new_forwards':0,'verifier_sha256':p.sha(__file__)})
    lines=['## 旧模型错误按原始类别拆解 ✅ 完成，描述性不晋级','', '**一句话：** 已把错题按原始标签类别拆开，模型与评分未改变。','', '**出发点：** 原native与患者模型开发Dice约0.744—0.756，但平均分不足以定位数据处理的具体问题。','', '**具体做法：** 四旧fit684模型重读同dev180/78；720例原计数和Point位置逐例exact，再从原label0—4分类。无新训练、不加载旋转模型，不改其冻结参数。','', '**结果：**','', '| 旧模型 | Point /180 | FN中label2占比 | label2体素召回 | FP中label4占比 |','|---|---:|---:|---:|---:|']
    for key,m in summary.items():lines.append(f"| {key} | {m['point_hits']} | {m['class_recall']['2']['false_negative_share']:.2%} | {m['class_recall']['2']['pooled_voxel_recall']:.2%} | {m['rc_share_of_false_positive_voxels']:.2%} |")
    lines+=['','**是否支持直觉：** 说明错误在原始类别中的分布，不说明某类别是因果瓶颈或其大小就是困难度；完整各类分母、患者等权召回及Point类别见机器结果。无CI，不把四模型当四份独立队列。','',f"**代价/下一步：** 720旧模型前向/{raw['seconds']:.1f}秒，诊断峰值{raw['peak_bytes']/1024**3:.3f}GiB；/tmp类标签360MiB、mask180MiB。独立掩膜核对720/720，未保存logits所以不声称重算argmax。只用于解释，不触发test、不修改本轮旋转/模糊参数。",'', '[七问](archive/research_records/2026-09-10_development_error_anatomy/IDEA.md)、[结果](archive/research_records/2026-09-10_development_error_anatomy/RESULT.json)、[原始](archive/research_records/2026-09-10_development_error_anatomy/RAW_DEVELOPMENT.json)、[核验](archive/research_records/2026-09-10_development_error_anatomy/VERIFICATION.json)。']
    text='\n'.join(lines)+'\n'
    with (REC/'RESULT.md').open('x') as f:f.write(text)
    (REC/'CARD.md').write_text(text);publish(str(REC.relative_to(p.ROOT)))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
