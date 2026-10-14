"""Retrospective evidence correction, never a preregistration repair."""
from datetime import datetime,timezone
from collections import defaultdict
import json
import statistics
import numpy as np
from scripts import structured_background_protocol as p
from scripts.finalize_structured_background_aux import independent_ci


def main():
    out=p.ROOT/'archive/research_records/2026-09-10_resumed_eight_hour/POINT_RECORD_CORRECTION.json'
    if out.exists():raise FileExistsError(out)
    _,dev=p.check_roles();index=json.loads((p.CACHE/'index.json').read_text());mapping={c:i for i,c in enumerate(index['case_ids'])}
    cache=json.loads((p.RECORD/'AUX_CACHE_RECEIPT.json').read_text());p.checked(p.CACHE/'labels.u8',cache['source_labels_sha256'])
    labels=np.memmap(p.CACHE/'labels.u8',mode='r',dtype=np.uint8,shape=(972,128,128,128))
    source=p.ROOT/'archive/research_records/2026-09-10_subject_fixed_flip8/RAW_DEVELOPMENT.json';base=json.loads(source.read_text())
    records={}
    for method,variant,bseed in [('point_consensus','vote',2026091121),('mask_centroid_point','centroid',2026091131)]:
        root=p.ROOT/f'archive/research_records/2026-09-11_{method}'
        raw=json.loads((root/'RAW_DEVELOPMENT.json').read_text());pre=json.loads((root/'DEV_PREACCESS.json').read_text())
        p.checked(root/'DEV_PREACCESS.json',raw['dev_preaccess_sha256'])
        bound={}
        for name,digest in pre['files'].items():
            actual=p.sha(p.ROOT/name);bound[name]={'expected':digest,'actual':actual,'matches':actual==digest}
        try:old=json.loads((root/'VERIFICATION_RECEIPT.json').read_text());valid=True
        except json.JSONDecodeError:old=None;valid=False
        per={}
        for i,seed in enumerate(p.SEEDS):
            m=raw['results'][str(seed)][variant];rows=m['per_case'];groups=defaultdict(list)
            assert [(r['case_id'],r['subject_id']) for r in rows]==[(r['case_id'],r['subject_id']) for r in dev]
            flip=base['results'][str(seed)]['flip8'];identity=base['results'][str(seed)]['identity'];equal=[]
            for row,other in zip(rows,flip['per_case'],strict=True):
                truth=labels[mapping[row['case_id']]].reshape(-1)
                assert row['target_voxels']==int(truth.sum())
                assert 0<=row['intersection']<=min(row['predicted_voxels'],row['target_voxels'])
                assert row['dice']==2*row['intersection']/(row['predicted_voxels']+row['target_voxels'])
                assert 0<=row['point_flat_index']<len(truth)
                assert bool(truth[row['point_flat_index']])==row['point_hit']
                groups[row['subject_id']].append(row['dice'])
                equal.append(all(row[k]==other[k] for k in ['case_id','subject_id','dice','intersection','predicted_voxels','target_voxels']))
            assert abs(m['scan_dice']-statistics.fmean(r['dice'] for r in rows))<1e-12
            assert abs(m['subject_dice']-statistics.fmean(statistics.fmean(v) for v in groups.values()))<1e-12
            assert m['point_hits']==sum(r['point_hit'] for r in rows)
            intervals=independent_ci(rows,bseed+i)
            assert all(np.allclose(m[k],v,atol=1e-12,rtol=0) for k,v in intervals.items())
            per[str(seed)]={'scan_dice':m['scan_dice'],'subject_dice':m['subject_dice'],'point_hits':m['point_hits'],'saved_counts_equal_flip8':all(equal),'point_vs_identity':m['point_hits']-identity['point_hits'],'point_vs_flip8':m['point_hits']-flip['point_hits'],'preregistered_native_floor_passed':m['point_hits']>=[169,162][i],'preregistered_identity_gate_passed':m['point_hits']>=identity['point_hits'],'stored_point_truth_checked':180,'independent_absolute_ci_checked':True}
        records[method]={'raw_created_utc':raw['created_utc'],'directory_date_does_not_match_execution_date':True,'original_verification_json_valid':valid,'original_receipt_establishes_independent_verification':False,'raw_sha256':p.sha(root/'RAW_DEVELOPMENT.json'),'result_sha256':p.sha(root/'RESULT.json'),'original_verification_sha256':p.sha(root/'VERIFICATION_RECEIPT.json'),'preaccess_sha256':p.sha(root/'DEV_PREACCESS.json'),'bound_file_current_matches':bound,'metrics':per,'new_model_forwards':0,'historical_new_forwards':raw['new_forwards'],'preregistered_joint_passed':False,'limitations':['raw logits/prediction masks not preserved: no independent point-generator or voxelwise-mask reconstruction','posthoc audit cannot replace missing pre-execution smoke','point definition differs from primary raw-logit-argmax leaderboard','consensus uses stacked mean rather than frozen ordered accumulation; equal counts do not establish byte equality'] if method=='point_consensus' else ['masks not preserved: no independent centroid reconstruction','posthoc audit cannot replace missing pre-execution smoke','point definition differs from primary raw-logit-argmax leaderboard'],'gate_discrepancy':'consensus code omits native floor and adds flip8-point gate' if method=='point_consensus' else 'code adds flip8-point gate; raw negative unchanged','no_original_files_modified':True}
    report={'status':'retrospective_saved_record_checks_completed_not_full_inference_verification','created_utc':datetime.now(timezone.utc).isoformat(),'records':records,'old_window_extra_forwards_not_part_of_original_8h':5760,'new_forwards':0,'access':{'reused_dev_labels':True,'test':False,'outer':False,'confirmation':False},'code_sha256':p.sha(__file__),'labels_sha256':cache['source_labels_sha256'],'cache_index_sha256':p.sha(p.CACHE/'index.json')}
    p.write_new(out,report)
    md=out.with_suffix('.md')
    with md.open('x') as f:f.write('# 旧点研究证据更正\n\n两项实际执行日期为Sept10，不是目录名Sept11；额外5760开发前向不计入旧八小时5040。原收据与结果均原样保留。共720条已保存点命中和计数算术、绝对患者簇置信区间重新核对；不重跑推理，因此不声称独立重建投票/质心/掩膜。\n\n点共识原验证不是合法JSON；掩膜中心点原验证只有哈希，不能认作独立核验。前者原代码遗漏native点下限并改用stack平均，两者代码增加原预注册未列的Flip8点门。两者在真正已注册的identity点门依然失败：共识seed12少6例，质心两seed各少38例。没有因此准入测试，也不证明所有点研究不可能。\n\n新核验不补造事前检查，不修订旧门，不改变榜单。详见[逐项机器更正](POINT_RECORD_CORRECTION.json)。\n')
    print(json.dumps({'status':report['status'],'points':{k:v['metrics'] for k,v in records.items()}},indent=2))


if __name__=='__main__':main()
