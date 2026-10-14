"""Independent saved-count and patient-cluster audit of the fixed8-view study."""
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import numpy as np
from scripts import structured_background_protocol as p
from scripts.finalize_structured_background_aux import independent_ci

RECORD=p.ROOT/'archive/research_records/2026-09-10_subject_fixed_flip8'
SOURCE=p.ROOT/'archive/research_records/2026-09-07_subject_balanced_sampling'


def main():
    output=RECORD/'RESULT.json'
    if output.exists():raise FileExistsError(output)
    raw_path=RECORD/'RAW_DEVELOPMENT.json';raw=json.loads(raw_path.read_text());pre=RECORD/'DEV_PREACCESS.json';p.checked(pre,raw['dev_preaccess_sha256']);receipt=json.loads(pre.read_text())
    for name,digest in receipt['files'].items():p.checked(p.ROOT/name,digest)
    p.checked(SOURCE/'RAW_DEVELOPMENT.json',receipt['source_raw_sha256']);source=json.loads((SOURCE/'RAW_DEVELOPMENT.json').read_text())
    metrics={};gates={};checks={}
    for i,seed in enumerate(p.SEEDS):
        key=str(seed);pair=raw['results'][key]
        assert pair['identity']['per_case']==source['results'][key]['candidate']['per_case']
        for name,summary in pair.items():
            grouped=defaultdict(list)
            for row in summary['per_case']:
                assert abs(row['dice']-2*row['intersection']/(row['predicted_voxels']+row['target_voxels']))<1e-12
                grouped[row['subject_id']].append(row['dice'])
            assert len(summary['per_case'])==180 and len(grouped)==78
            assert abs(summary['scan_dice']-statistics.fmean(r['dice'] for r in summary['per_case']))<1e-12
            assert abs(summary['subject_dice']-statistics.fmean(statistics.fmean(v) for v in grouped.values()))<1e-12
            assert summary['point_hits']==sum(r['point_hit'] for r in summary['per_case'])
            ci=independent_ci(summary['per_case'],2026091121+i);assert all(np.allclose(v,summary[k],rtol=0,atol=1e-12) for k,v in ci.items())
        delta=[]
        for a,b in zip(pair['identity']['per_case'],pair['flip8']['per_case'],strict=True):
            assert (a['case_id'],a['subject_id'],a['target_voxels'])==(b['case_id'],b['subject_id'],b['target_voxels']);delta.append({**a,'dice':b['dice']-a['dice']})
        c=raw['comparisons'][key];ci=independent_ci(delta,2026091121+i)
        assert all(np.allclose(v,c[k],rtol=0,atol=1e-12) for k,v in ci.items())
        assert abs(c['scan_delta']-(pair['flip8']['scan_dice']-pair['identity']['scan_dice']))<1e-12
        assert abs(c['subject_delta']-(pair['flip8']['subject_dice']-pair['identity']['subject_dice']))<1e-12
        assert c['point_delta']==pair['flip8']['point_hits']-pair['identity']['point_hits']
        g={'scan_positive':c['scan_delta']>0,'subject_positive':c['subject_delta']>0,'own_point_safety':c['point_delta']>=-1,'scan_ci_safety':ci['scan_cluster_95ci'][0]>=-.003,'subject_ci_safety':ci['subject_95ci'][0]>=-.003,'native_point_floor':pair['flip8']['point_hits']>=[169,162][i]}
        assert g==raw['gates'][key];gates[key]=g;metrics[key]={name:{k:v for k,v in summary.items() if k!='per_case'} for name,summary in pair.items()}
        for value in receipt['models'][key].values():p.checked(value['path'],value['sha256'])
        checks[key]={'identity_per_case_exact':True,'metric_and_paired_ci_recomputed':True,'frozen_gates_recomputed':True,'checkpoint_report_hashes_exact':True}
    avg=statistics.fmean(c['scan_delta'] for c in raw['comparisons'].values());passed=all(all(g.values()) for g in gates.values()) and avg>=.003
    assert avg==raw['average_scan_delta'] and passed==raw['all_gates_passed'] and raw['forwards']==2880
    result={'schema':'aflow.subject_fixed_flip8.result','created_utc':datetime.now(timezone.utc).isoformat(),'status':'completed_development_supported' if passed else 'completed_development_gate_failed','metrics':metrics,'comparisons':raw['comparisons'],'gates':gates,'average_scan_delta':avg,'all_gates_passed':passed,'models':receipt['models'],'raw_sha256':p.sha(raw_path),'dev_preaccess_sha256':p.sha(pre),'access':raw['access'],'cost':{'new_training':0,'development_forwards':2880,'seconds':raw['seconds'],'peak_bytes':raw['peak_bytes'],'inference_forwards':8},'fixed_test_seed_if_eligible':2026090711,'next_decision':'freeze_one_fixed_seed11_test_panel_if_time_allows' if passed else 'close_fixed_subject_flip8_no_view_weight_threshold_seed_search'}
    p.write_new(output,result);p.write_new(RECORD/'VERIFICATION_RECEIPT.json',{'status':'passed','result_sha256':p.sha(output),'raw_sha256':p.sha(raw_path),'checks':checks,'new_model_forwards':0,'verifier_sha256':p.sha(Path(__file__))})
    lines=['# 患者模型固定八镜像：开发结果','',f"状态：{result['status']}。",'',
           '同两个固定患者采样checkpoint，只把1次原输出改成同一模型8镜像还原raw-logit等权平均；128³/阈值0/argmax/原hard GT不变，fit684/313、dev180/78，无训练或多模型融合。历史1×采样配方的失败结论保留，本新应用单独验证。','',
           '| seed | 输出 | Scan Dice | Subject Dice | Point /180 |','|---|---|---:|---:|---:|']
    for seed,values in metrics.items():
        for name,m in values.items():lines.append(f"| {seed} | {name} | {m['scan_dice']:.6f} | {m['subject_dice']:.6f} | {m['point_hits']} |")
    for seed,c in raw['comparisons'].items():lines.append(f"\n- {seed} Flip8−identity：Scan {c['scan_delta']:+.6f} CI {c['scan_cluster_95ci']}；Subject {c['subject_delta']:+.6f} CI {c['subject_95ci']}；Point {c['point_delta']:+d}。")
    lines+=['','10k患者簇百分位95%CI，两个seed共用78人，未多比较校正；本提案基于已揭示开发和历史推理证据，不是新盲确认。', '',f"成本：0训练、2880forwards/{raw['seconds']:.2f}s、peak{raw['peak_bytes']/1024**3:.3f}GiB，部署同4.70M/8×而非1×。",'',result['next_decision'],'',
            '当前没有test/oldconfirmation/fixed validation/outer/external新访问；不能把dev分数与旧test270榜0.7572696616942132直接比较。', '',
            '[七问](IDEA.md)、[预注册](PREREG.md)、[raw](RAW_DEVELOPMENT.json)、[开发前收据](DEV_PREACCESS.json)、[机器结果](RESULT.json)、[验证](VERIFICATION_RECEIPT.json)。', '',f"Raw SHA `{p.sha(raw_path)}`；RESULT SHA `{p.sha(output)}`。"]
    with (RECORD/'RESULT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'status':result['status'],'metrics':metrics,'gates':gates,'all_gates_passed':passed},indent=2),flush=True)


if __name__=='__main__':main()
