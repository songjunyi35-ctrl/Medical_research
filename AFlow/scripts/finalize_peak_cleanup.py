"""Independent arithmetic/CI/point-preservation audit and complete negative archival."""
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import numpy as np
from scripts import peak_cleanup_protocol as p
from scripts.finalize_structured_background_aux import independent_ci


def main():
    output=p.RECORD/'RESULT.json'
    if output.exists():raise FileExistsError(output)
    p.source_controls()
    raw_path=p.RECORD/'RAW_DEVELOPMENT.json';raw=json.loads(raw_path.read_text())
    pre=p.RECORD/'DEV_PREACCESS.json';p.checked(pre,raw['dev_preaccess_sha256']);receipt=json.loads(pre.read_text())
    for name,digest in receipt['files'].items():p.checked(p.ROOT/name,digest)
    source=receipt['source_raw'];p.checked(source['path'],source['sha256']);old=json.loads(Path(source['path']).read_text())
    metrics={};comparisons={};gates={};mechanism={};checks={}
    for i,seed in enumerate(p.SEEDS):
        key=str(seed);values=raw['results'][key]
        assert values['base']['per_case']==old['results'][f'seed{seed}_control']['per_case']
        for arm in ('base','clean'):
            summary=values[arm];rows=summary['per_case'];groups=defaultdict(list)
            for r in rows:
                assert abs(r['dice']-2*r['intersection']/(r['predicted_voxels']+r['target_voxels']))<1e-12
                groups[r['subject_id']].append(r['dice'])
            assert len(rows)==180 and len(groups)==78
            assert abs(summary['scan_dice']-statistics.fmean(r['dice'] for r in rows))<1e-12
            assert abs(summary['subject_dice']-statistics.fmean(statistics.fmean(v) for v in groups.values()))<1e-12
            assert summary['point_hits']==sum(r['point_hit'] for r in rows)
            intervals=independent_ci(rows,2026091061+i)
            assert all(np.allclose(v,summary[k],rtol=0,atol=1e-12) for k,v in intervals.items())
        delta=[];tp=fp=basefp=0;affected=0
        for a,b,diag in zip(values['base']['per_case'],values['clean']['per_case'],values['component_diagnostics'],strict=True):
            assert a['case_id']==b['case_id']==diag['case_id']
            assert (a['subject_id'],a['target_voxels'],a['point_flat_index'],a['point_hit'])==(b['subject_id'],b['target_voxels'],b['point_flat_index'],b['point_hit'])
            assert a['predicted_voxels']-b['predicted_voxels']==diag['removed_voxels']
            assert a['intersection']-b['intersection']==diag['removed_true_positive_voxels']
            assert diag['removed_voxels']==diag['removed_true_positive_voxels']+diag['removed_false_positive_voxels']
            assert diag['components_after']<=diag['components_before'] and diag['point_flat_index']==a['point_flat_index']
            tp+=diag['removed_true_positive_voxels'];fp+=diag['removed_false_positive_voxels'];basefp+=a['predicted_voxels']-a['intersection'];affected+=int(diag['removed_voxels']>0)
            delta.append({**a,'dice':b['dice']-a['dice']})
        c={'scan_delta':values['clean']['scan_dice']-values['base']['scan_dice'],'subject_delta':values['clean']['subject_dice']-values['base']['subject_dice'],'point_delta':values['clean']['point_hits']-values['base']['point_hits'],**independent_ci(delta,2026091061+i)}
        assert all(np.allclose(v,raw['comparisons'][key][k],rtol=0,atol=1e-12) for k,v in c.items())
        c.update(improved=sum(r['dice']>0 for r in delta),worse=sum(r['dice']<0 for r in delta),tied=sum(r['dice']==0 for r in delta))
        assert c==raw['comparisons'][key]
        g={'subject_gain':c['subject_delta']>=.003,'scan_positive':c['scan_delta']>0,'point_exact':c['point_delta']==0,'subject_ci_safety':c['subject_95ci'][0]>=-.003,'scan_cluster_ci_safety':c['scan_cluster_95ci'][0]>=-.003}
        assert g==raw['gates'][key]
        comparisons[key]=c;gates[key]=g;metrics[key]={a:{k:v for k,v in values[a].items() if k!='per_case'} for a in ('base','clean')}
        mechanism[key]={'affected_cases':affected,'removed_true_positive_voxels':tp,'removed_false_positive_voxels':fp,'fraction_of_baseline_false_positive_voxels_removed':fp/basefp if basefp else 0,'original_false_positive_voxels':basefp,'point_locations_preserved':180}
        checks[key]={'historical_baseline_per_case_exact':True,'saved_overlap_and_removed_tp_fp_counts_consistent':True,'point_locations_exact':True,'absolute_and_paired_ci_recomputed':True,'gates_recomputed':True,'predicted_masks_not_independently_replayed':True}
    average=statistics.fmean(c['scan_delta'] for c in comparisons.values());passed=all(all(g.values()) for g in gates.values()) and average>=.003
    assert abs(average-raw['average_scan_delta'])<1e-12 and passed==raw['all_gates_passed']
    assert raw['new_forwards']==360 and not any(v for k,v in raw['access'].items() if k!='reused_development180')
    result={'schema':'aflow.peak_protected_cleanup.result','created_utc':datetime.now(timezone.utc).isoformat(),'status':'completed_development_supported' if passed else 'completed_development_gate_failed','metrics':metrics,'comparisons':comparisons,'gates':gates,'average_scan_delta':average,'all_gates_passed':passed,'mechanism':mechanism,'checkpoints':receipt['controls'],'deployment_rule':receipt['rule'],'raw_sha256':p.sha(raw_path),'dev_preaccess_sha256':p.sha(pre),'source_raw_sha256':source['sha256'],'cache_index_sha256':p.sha(p.CACHE/'index.json'),'access':raw['access'],'cost':{'new_training':0,'new_development_forwards':360,'seconds':raw['seconds'],'peak_gpu_allocated_bytes':raw['peak_gpu_allocated_bytes'],'inference_forwards_per_case':1,'cpu_connected_component_postprocessing':True},'decision':'retain_independent_development_candidate_pending_fixed_branch_admission' if passed else 'close_fixed_cleanup_rule_no_size_fraction_connectivity_search'}
    p.write_new(output,result)
    p.write_new(p.RECORD/'VERIFICATION_RECEIPT.json',{'status':'passed','result_sha256':p.sha(output),'raw_sha256':p.sha(raw_path),'preaccess_sha256':p.sha(pre),'checks':checks,'new_forwards':0,'verifier_sha256':p.sha(Path(__file__))})
    lines=['# 保护峰值的小孤岛清理：开发结果','',f"状态：{result['status']}。",'',
           '只增加固定26邻接小组件清理（max(8,最大块1%)，并保护最大块/最高logit点），其余checkpoint/阈值/输入/模型不变。两个旧native模型都只由fit684训练；同一dev180/78已复用，非盲、非外部、不是两份独立cohort。无新训练。','',
           '| seed | 输出 | Scan Dice | Subject Dice | Point /180 |','|---|---|---:|---:|---:|']
    for seed,values in metrics.items():
        for arm,m in values.items():lines.append(f"| {seed} | {arm} | {m['scan_dice']:.6f} | {m['subject_dice']:.6f} | {m['point_hits']} |")
    for seed,c in comparisons.items():
        lines.append(f"\n{seed}：Scan差{c['scan_delta']:+.6f}，患者簇95%CI {c['scan_cluster_95ci']}；Subject差{c['subject_delta']:+.6f}，CI {c['subject_95ci']}；Point差{c['point_delta']:+d}，180点位置全相同。改善/恶化/相同={c['improved']}/{c['worse']}/{c['tied']}。")
        m=mechanism[seed];lines.append(f"清理涉及{m['affected_cases']}例，删真阳性{m['removed_true_positive_voxels']}体素、假阳性{m['removed_false_positive_voxels']}体素；仅占该模型原假阳性体素的{m['fraction_of_baseline_false_positive_voxels_removed']*100:.3f}%。")
    lines+=['','区间为固定seed的10,000次患者簇百分位95%CI，未校正多比较；Point不变是算子设计及逐例验证，不代表Dice必升。任何失败门不按比例/连接性/阈值救回。','',
            f"成本：新训练0，开发360forwards/{raw['seconds']:.2f}s，额外峰值{raw['peak_gpu_allocated_bytes']/1024**3:.3f}GiB；部署模型仍1×，但多CPU组件计数和传输延迟，无新影像cache/新checkpoint。",'',
            result['decision'], '',
            '未访问弱形变新checkpoint、test270、fixed validation、outer、confirmation或external。弱形变训练/门保持原样，不与本规则捆绑。旧榜overall仍0.757269662（Full972KD168八镜像8×）、1×Scan0.752003176（GCSAM184），同test270/119，不能用开发数值直接比较。', '',
            '[七问](IDEA.md)、[预注册](PREREG.md)、[raw](RAW_DEVELOPMENT.json)、[preaccess](DEV_PREACCESS.json)、[机器结果](RESULT.json)、[验证](VERIFICATION_RECEIPT.json)。', '',f'Raw SHA `{p.sha(raw_path)}`；RESULT JSON SHA `{p.sha(output)}`。']
    for seed,c in receipt['controls'].items():lines.append(f"\n- [{seed}原模型checkpoint](../../../{Path(c['checkpoint']['path']).relative_to(p.ROOT)})；清理需额外固定规则，不存在新训练权重。")
    with (p.RECORD/'RESULT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'status':result['status'],'metrics':metrics,'comparisons':comparisons,'mechanism':mechanism},indent=2),flush=True)


if __name__=='__main__':main()
