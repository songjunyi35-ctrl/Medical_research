"""Independent verification/archival of soft-target training, hard-GT evaluation."""
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import numpy as np
from scripts import fractional_occupancy_protocol as p
from scripts.finalize_structured_background_aux import independent_ci


def main():
    p.check_freeze();output=p.RECORD/'RESULT.json'
    if output.exists():raise FileExistsError(output)
    raw_path=p.RECORD/'RAW_DEVELOPMENT.json';raw=json.loads(raw_path.read_text());pre=p.RECORD/'DEV_PREACCESS.json'
    p.checked(pre,raw['dev_preaccess_sha256']);receipt=json.loads(pre.read_text())
    for name,digest in receipt['files'].items():p.checked(p.ROOT/name,digest)
    source_path=p.source.RECORD/'RAW_DEVELOPMENT.json';p.checked(source_path,receipt['source_raw_sha256']);source=json.loads(source_path.read_text())
    cache=json.loads((p.RECORD/'CACHE_RECEIPT.json').read_text());p.checked(Path(cache['target_path']),cache['target_sha256'])
    p.checked(p.CACHE/'labels.u8',receipt['hard_labels_sha256']);index=json.loads((p.CACHE/'index.json').read_text());slots={c:i for i,c in enumerate(index['case_ids'])}
    labels=np.memmap(p.CACHE/'labels.u8',mode='r',dtype=np.uint8,shape=(972,128,128,128))
    _,dev=p.check_roles();truth={r['case_id']:int(labels[slots[r['case_id']]].sum()) for r in dev}
    metrics={};comparisons={};gates={};checkpoints={};checks={}
    for i,seed in enumerate(p.SEEDS):
        key=str(seed);pair=raw['results'][key];control=pair['reused_control'];candidate=pair['candidate']
        assert control==source['results'][f'seed{seed}_control']
        rows=candidate['per_case'];groups=defaultdict(list)
        assert [(r['case_id'],r['subject_id']) for r in rows]==[(r['case_id'],r['subject_id']) for r in dev]
        for row in rows:
            assert row['target_voxels']==truth[row['case_id']]
            assert abs(row['dice']-2*row['intersection']/(row['predicted_voxels']+row['target_voxels']))<1e-12
            assert bool(labels[slots[row['case_id']]].reshape(-1)[row['point_flat_index']])==row['point_hit']
            groups[row['subject_id']].append(row['dice'])
        assert len(rows)==180 and len(groups)==78
        absolute=independent_ci(rows,2026091081+i)
        assert abs(candidate['scan_dice']-statistics.fmean(r['dice'] for r in rows))<1e-12
        assert abs(candidate['subject_dice']-statistics.fmean(statistics.fmean(v) for v in groups.values()))<1e-12
        assert candidate['point_hits']==sum(r['point_hit'] for r in rows)
        assert all(np.allclose(v,candidate[k],rtol=0,atol=1e-12) for k,v in absolute.items())
        delta=[]
        for a,b in zip(control['per_case'],rows,strict=True):
            assert (a['case_id'],a['subject_id'],a['target_voxels'])==(b['case_id'],b['subject_id'],b['target_voxels'])
            delta.append({**a,'dice':b['dice']-a['dice']})
        interval=independent_ci(delta,2026091081+i);c=raw['comparisons'][key]
        assert all(np.allclose(v,c[k],rtol=0,atol=1e-12) for k,v in interval.items())
        assert abs(c['scan_delta']-(candidate['scan_dice']-control['scan_dice']))<1e-12
        assert abs(c['subject_delta']-(candidate['subject_dice']-control['subject_dice']))<1e-12
        assert c['point_delta']==candidate['point_hits']-control['point_hits']
        g={'subject_gain':c['subject_delta']>=.003,'scan_positive':c['scan_delta']>0,'point_safety':c['point_delta']>=-1,'subject_ci_safety':interval['subject_95ci'][0]>=-.003,'scan_cluster_ci_safety':interval['scan_cluster_95ci'][0]>=-.003}
        assert g==raw['gates'][key];gates[key]=g;comparisons[key]=c
        report_info=receipt['reports'][key];p.checked(report_info['report_path'],report_info['report_sha256']);report=json.loads(Path(report_info['report_path']).read_text())
        ckpt=Path(report['checkpoint']['path']);p.checked(ckpt,report['checkpoint']['sha256'])
        checkpoints[key]={'path':str(ckpt.relative_to(p.ROOT)),'sha256':p.sha(ckpt),'train_report_sha256':p.sha(report_info['report_path']),'seconds':report['elapsed_seconds'],'peak_bytes':report['peak_gpu_memory_bytes']}
        metrics[key]={arm:{k:v for k,v in value.items() if k!='per_case'} for arm,value in pair.items()}
        checks[key]={'unchanged_hard_gt_counts':True,'saved_point_gt_hit_verified':True,'argmax_not_independently_recomputed':True,'absolute_paired_cluster_ci_recomputed':True,'gates_recomputed':True,'checkpoint_sha_exact':True}
    average=statistics.fmean(c['scan_delta'] for c in comparisons.values());passed=all(all(g.values()) for g in gates.values()) and average>=.003
    assert abs(average-raw['average_scan_delta'])<1e-12 and passed==raw['all_gates_passed'] and raw['new_forwards']==360
    assert not any(v for k,v in raw['access'].items() if k!='reused_development180')
    result={'schema':'aflow.fractional_occupancy.result','created_utc':datetime.now(timezone.utc).isoformat(),'status':'completed_reused_development_supported' if passed else 'completed_reused_development_gate_failed','metrics':metrics,'comparisons':comparisons,'gates':gates,'average_scan_delta':average,'all_gates_passed':passed,'checkpoints':checkpoints,'raw_sha256':p.sha(raw_path),'dev_preaccess_sha256':p.sha(pre),'cache_receipt_sha256':p.sha(p.RECORD/'CACHE_RECEIPT.json'),'soft_target_sha256':p.sha(Path(cache['target_path'])),'cache_index_sha256':p.sha(p.CACHE/'index.json'),'evaluation_target':'unchanged_original_hard_GT','access':raw['access'],'verification':checks,'fixed_test_seed_if_eligible':2026090711,'next_decision':'eligible_fixed_seed11_test_requires_separate_preaccess_and_session_budget' if passed else 'close_exact_occupancy_target_original_loss_pairing_no_kernel_threshold_seed_search'}
    p.write_new(output,result);p.write_new(p.RECORD/'VERIFICATION_RECEIPT.json',{'status':'passed','result_sha256':p.sha(output),'raw_sha256':p.sha(raw_path),'preaccess_sha256':p.sha(pre),'checks':checks,'new_forwards':0,'reused_development_label_reread':True,'verifier_sha256':p.sha(Path(__file__))})
    lines=['# 精确体积占比训练标签：原硬GT开发结果','',f"状态：{result['status']}。",'',
           '唯一改变为训练WT target的体积占比表示，float64计算/float16缓存/float32训练；图像、native抽样、初始化、原增强、168轮、主损失、阈值0和1×输出不变，无教师。开发仍原硬128³ GT，不按新target评分。两seed共用同一180扫描/78患者reused development，非外部或独立确认。','',
           '| seed | arm | Scan Dice | Subject Dice | Point /180 |','|---|---|---:|---:|---:|']
    for seed,values in metrics.items():
        for arm,m in values.items():lines.append(f"| {seed} | {arm} | {m['scan_dice']:.6f} | {m['subject_dice']:.6f} | {m['point_hits']} |")
    for seed,c in comparisons.items():lines.append(f"\n{seed}：Scan差{c['scan_delta']:+.6f}，簇95%CI {c['scan_cluster_95ci']}；Subject差{c['subject_delta']:+.6f}，CI {c['subject_95ci']}；Point差{c['point_delta']:+d}。")
    lines+=['','10,000次患者簇百分位95%CI，两个固定seed、同一78人，未多比较校正；不取较好seed或训练epoch。质量守恒是工程性质，不自动证明分割提升或概率校准。', '',
            f"新增两训练{sum(v['seconds'] for v in checkpoints.values())/3600:.3f}GPU小时、峰值{max(v['peak_bytes'] for v in checkpoints.values())/1024**3:.3f}GiB；开发360forwards/{raw['seconds']:.2f}s。临时soft target cache2.672GiB，原影像不复制；部署4.70M/1×。",'',
            result['next_decision'],'',
            '目前无新增test/outer/confirmation/fixed validation/external性能；旧test270/119 overall Full972KD168 Flip8 Scan0.7572696616942132，1×Scan GCSAM0.7520031755617648，不能和此dev180混比。来源镜像/官方字节parity未证明限制仍保留。','',
            '[七问](IDEA.md)、[预注册](PREREG.md)、[cache](CACHE_RECEIPT.json)、[raw](RAW_DEVELOPMENT.json)、[preaccess](DEV_PREACCESS.json)、[机器结果](RESULT.json)、[验证](VERIFICATION_RECEIPT.json)。','',f'Raw SHA `{p.sha(raw_path)}`；RESULT JSON SHA `{p.sha(output)}`。']
    for seed,c in checkpoints.items():lines.append(f"\n- [{seed} checkpoint](../../../{c['path']}) SHA `{c['sha256']}`。")
    with (p.RECORD/'RESULT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'status':result['status'],'metrics':metrics,'comparisons':comparisons,'gates':gates},indent=2),flush=True)


if __name__=='__main__':main()
