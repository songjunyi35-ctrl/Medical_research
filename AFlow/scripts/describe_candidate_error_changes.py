"""Zero-forward error interpretation versus each candidate's own sampling control."""
import argparse
from datetime import datetime,timezone
import json
from scripts.structured_background_protocol import ROOT,checked,sha,write_new
from scripts.require_development_mask_verification import require_masks


def differences(candidate,control):
    truth=candidate['class_target_voxels'];pred=candidate['class_predicted_voxels']
    assert truth==[control['class_target_voxels'][str(i)] for i in range(5)]
    result={'class_recall':{},'false_positive_voxel_change':{}}
    for i in [1,2,3]:
        a=control['class_predicted_voxels'][str(i)]/truth[i];b=pred[i]/truth[i]
        result['class_recall'][str(i)]={'control':a,'candidate':b,'delta':b-a}
    for i in [0,4]:result['false_positive_voxel_change'][str(i)]=pred[i]-control['class_predicted_voxels'][str(i)]
    ch=candidate['point_class_histogram'];oh=control['point_class_histogram']
    result['point_miss_change_by_class']={str(i):ch.get(str(i),0)-oh.get(str(i),0) for i in [0,4]}
    result['point_hit_delta']=-sum(result['point_miss_change_by_class'].values())
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',choices=['small_rotation','image_blur'],required=True);args=ap.parse_args()
    rec=ROOT/f'archive/research_records/2026-09-10_{args.study}'
    if (rec/'ERROR_CHANGE_DESCRIPTION.json').exists():raise FileExistsError('already described')
    proof=require_masks(rec);result=json.loads((rec/'RESULT.json').read_text())
    base=ROOT/'archive/research_records/2026-09-10_development_error_anatomy';verification=json.loads((base/'VERIFICATION.json').read_text())
    checked(base/'RESULT.json',verification['result_sha256']);old=json.loads((base/'RESULT.json').read_text())
    assert old['cohort']=={'scans':180,'subjects':78}
    output={}
    for key,candidate in proof['class_error_description'].items():
        seed,arm=key.split('_',1);control_key=seed+('_subject' if arm.startswith('subject') else '_native')
        delta=differences(candidate,old['summary'][control_key]);delta['control_key']=control_key
        assert delta['point_hit_delta']==result['comparisons'][key]['versus_own_sampling_control']['point_delta']
        output[key]=delta
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'study':args.study,'result_sha256':sha(rec/'RESULT.json'),'actual_mask_verification_sha256':sha(rec/'DEV_MASK_VERIFICATION.json'),'control_error_result_sha256':sha(base/'RESULT.json'),'comparisons':output,'new_forwards':0,'descriptive_only':True,'primary_gates_and_parameters_unchanged':True,'confidence_intervals':None,'recall_scope':'pooled voxels within same dev180/78, not patient Dice'}
    write_new(rec/'ERROR_CHANGE_DESCRIPTION.json',report)
    lines=['# 新数据处理究竟改变了哪些错误？','', '只使用已经保存并独立核对的mask统计，没有新增模型前向。每个候选和自身同采样对照比较，不用native复合增益替代方法自身效应。以下是同dev180/78总体体素描述，不是患者Dice、置信区间或额外晋级门。','', '| 候选 | label2召回变化（百分点） | label3召回变化（百分点） | label4误画体素变化 | label4错误点变化 | label0错误点变化 |','|---|---:|---:|---:|---:|---:|']
    for key,d in output.items():lines.append(f"| {key} | {100*d['class_recall']['2']['delta']:+.3f} | {100*d['class_recall']['3']['delta']:+.3f} | {d['false_positive_voxel_change']['4']:+d} | {d['point_miss_change_by_class']['4']:+d} | {d['point_miss_change_by_class']['0']:+d} |")
    lines+=['','误画体素或错误点变化为负表示减少；召回变化为正表示覆盖增加。二者可能不同向，不据某一格掩盖总体门失败；也不在看到本表后改角度、模糊强度、阈值或种子。','', '[数值与来源SHA](ERROR_CHANGE_DESCRIPTION.json)、[实际mask核验](DEV_MASK_VERIFICATION.json)、[正式主结果/门](RESULT.json)。']
    with (rec/'ERROR_CHANGE_DESCRIPTION.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
