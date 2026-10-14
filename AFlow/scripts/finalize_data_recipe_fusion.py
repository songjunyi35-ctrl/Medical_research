"""Independent fixed-fusion readout audit; preserves constituent negative results."""
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import numpy as np
from scripts import structured_background_protocol as p
from scripts.finalize_structured_background_aux import independent_ci

RECORD=p.ROOT/'archive/research_records/2026-09-10_data_recipe_ensemble'


def main():
    output=RECORD/'RESULT.json'
    if output.exists():raise FileExistsError(output)
    raw_path=RECORD/'RAW_DEVELOPMENT.json';raw=json.loads(raw_path.read_text());pre=RECORD/'DEV_PREACCESS.json';p.checked(pre,raw['dev_preaccess_sha256']);receipt=json.loads(pre.read_text())
    for name,digest in receipt['files'].items():p.checked(p.ROOT/name,digest)
    source={}
    for name,value in receipt['source_raws'].items():p.checked(value['path'],value['sha256']);source[name]=json.loads(Path(value['path']).read_text())
    native=json.loads((p.RECORD/'RAW_DEVELOPMENT.json').read_text())
    metrics={};gates={};checks={}
    for ordinal,seed in enumerate(p.SEEDS):
        key=str(seed);results=raw['results'][key]
        for name in ('sampling','occupancy'):
            assert results[name]['per_case']==source[name]['results'][key]['candidate']['per_case']
            model=receipt['models'][f'{seed}_{name}'];p.checked(model['checkpoint']['path'],model['checkpoint']['sha256']);p.checked(model['train_report']['path'],model['train_report']['sha256'])
        for name,summary in results.items():
            rows=summary['per_case'];grouped=defaultdict(list)
            for row in rows:
                assert abs(row['dice']-2*row['intersection']/(row['predicted_voxels']+row['target_voxels']))<1e-12
                grouped[row['subject_id']].append(row['dice'])
            assert len(rows)==180 and len(grouped)==78
            assert abs(summary['scan_dice']-statistics.fmean(r['dice'] for r in rows))<1e-12
            assert abs(summary['subject_dice']-statistics.fmean(statistics.fmean(v) for v in grouped.values()))<1e-12
            assert summary['point_hits']==sum(r['point_hit'] for r in rows)
            ci=independent_ci(rows,2026091111+ordinal);assert all(np.allclose(v,summary[k],rtol=0,atol=1e-12) for k,v in ci.items())
        ensemble=results['ensemble'];contrasts=raw['comparisons'][key]
        for name in ('sampling','occupancy','native'):
            control=results[name] if name!='native' else native['results'][f'seed{seed}_control']
            delta=[]
            for a,b in zip(control['per_case'],ensemble['per_case'],strict=True):
                assert (a['case_id'],a['subject_id'],a['target_voxels'])==(b['case_id'],b['subject_id'],b['target_voxels'])
                delta.append({**a,'dice':b['dice']-a['dice']})
            c=contrasts[name];ci=independent_ci(delta,2026091111+ordinal)
            assert all(np.allclose(v,c[k],rtol=0,atol=1e-12) for k,v in ci.items())
            assert abs(c['scan_delta']-(ensemble['scan_dice']-control['scan_dice']))<1e-12 and abs(c['subject_delta']-(ensemble['subject_dice']-control['subject_dice']))<1e-12
            assert c['point_delta']==ensemble['point_hits']-control['point_hits']
        member=[results[n] for n in ('sampling','occupancy')]
        g={'scan_above_both_members':ensemble['scan_dice']>max(m['scan_dice'] for m in member),'subject_above_both_members':ensemble['subject_dice']>max(m['subject_dice'] for m in member),'best_member_point_safety':ensemble['point_hits']>=max(m['point_hits'] for m in member)-1,'native_subject_gain':contrasts['native']['subject_delta']>=.003,'native_scan_positive':contrasts['native']['scan_delta']>0,'all_member_scan_ci_safety':all(contrasts[n]['scan_cluster_95ci'][0]>=-.003 for n in ('sampling','occupancy')),'all_member_subject_ci_safety':all(contrasts[n]['subject_95ci'][0]>=-.003 for n in ('sampling','occupancy'))}
        assert g==raw['gates'][key];gates[key]=g;metrics[key]={name:{k:v for k,v in summary.items() if k!='per_case'} for name,summary in results.items()}
        checks[key]={'member_saved_rows_exact':True,'ensemble_count_arithmetic_exact':True,'absolute_paired_ci_and_gate_recomputed':True,'checkpoint_report_hashes_exact':True,'fusion_logits_not_independently_replayed':True}
    average=statistics.fmean(c['native']['scan_delta'] for c in raw['comparisons'].values());passed=all(all(g.values()) for g in gates.values()) and average>=.003
    assert abs(average-raw['average_native_scan_delta'])<1e-12 and passed==raw['all_gates_passed'] and raw['new_forwards']==720
    result={'schema':'aflow.data_recipe_ensemble.result','created_utc':datetime.now(timezone.utc).isoformat(),'status':'completed_development_supported' if passed else 'completed_development_gate_failed','metrics':metrics,'comparisons':raw['comparisons'],'gates':gates,'average_native_scan_delta':average,'all_gates_passed':passed,'members':receipt['models'],'fusion_formula':'0.5 sampling raw logits +0.5 occupancy raw logits','raw_sha256':p.sha(raw_path),'dev_preaccess_sha256':p.sha(pre),'access':raw['access'],'cost':{'new_training':0,'new_development_forwards':720,'seconds':raw['seconds'],'peak_gpu_bytes':raw['peak_bytes'],'inference_forwards':2,'inference_forwards_with_fixed_flip8':16},'decision':'candidate_requires_separate_fixed_seed11_test_preaccess_within_remaining_time' if passed else 'close_fixed_equal_fusion_no_weight_member_seed_search','constituent_negative_decisions_preserved':True}
    p.write_new(output,result);p.write_new(RECORD/'VERIFICATION_RECEIPT.json',{'status':'passed','result_sha256':p.sha(output),'raw_sha256':p.sha(raw_path),'checks':checks,'new_forwards':0,'verifier_sha256':p.sha(Path(__file__))})
    lines=['# 不同数据配方等权融合：开发结果','',f"状态：{result['status']}。",'',
           '仅将同seed两个固定168模型的raw logits等权平均；阈值0、平均分数最高点、原hard GT不变。源模型分别由患者均衡和体积占比标签训练，均仅fit684/313、无教师；两个原单模型失败结论保留。此提案产生于已揭示dev之后，复用180/78、非独立确认。','',
           '| seed | 输出 | Scan Dice | Subject Dice | Point /180 |','|---|---|---:|---:|---:|']
    for seed,values in metrics.items():
        for name,m in values.items():lines.append(f"| {seed} | {name} | {m['scan_dice']:.6f} | {m['subject_dice']:.6f} | {m['point_hits']} |")
    for seed,values in raw['comparisons'].items():
        for name,c in values.items():lines.append(f"\n- {seed} ensemble−{name}：Scan {c['scan_delta']:+.6f} CI {c['scan_cluster_95ci']}；Subject {c['subject_delta']:+.6f} CI {c['subject_95ci']}；Point {c['point_delta']:+d}。")
    lines+=['','10k患者簇百分位95%CI，未多比较校正；不能用一个模型Dice和另一个Point拼成绩。所有成员原输出逐例parity通过，融合是否兼得由冻结联合门判断。', '',
            f"新训练0；dev720成员forwards/{raw['seconds']:.2f}s、峰值{raw['peak_bytes']/1024**3:.3f}GiB。部署两模型/2×（如固定八镜像为16×），不是1×提分；没有新checkpoint。",'',result['decision'],'',
            '没有新增test/oldconfirmation/fixed validation/outer/external。旧test270/119整体0.7572696616942132、1×0.7520031755617648不与此开发分数混比。', '',
            '[七问](IDEA.md)、[预注册](PREREG.md)、[raw](RAW_DEVELOPMENT.json)、[开发前冻结](DEV_PREACCESS.json)、[机器结果](RESULT.json)、[独立验证](VERIFICATION_RECEIPT.json)。', '',f"Raw SHA `{p.sha(raw_path)}`；RESULT SHA `{p.sha(output)}`。"]
    for name,value in receipt['models'].items():lines.append(f"\n- [{name} checkpoint](../../../{Path(value['checkpoint']['path']).relative_to(p.ROOT)}) SHA `{value['checkpoint']['sha256']}`。")
    with (RECORD/'RESULT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'status':result['status'],'metrics':metrics,'gates':gates,'all_gates_passed':passed},indent=2),flush=True)


if __name__=='__main__':main()
