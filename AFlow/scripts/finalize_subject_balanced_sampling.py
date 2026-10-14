"""Independently verify the saved sampler-study readout and archive it."""
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import numpy as np
from scripts import subject_balanced_protocol as p
from scripts.finalize_structured_background_aux import independent_ci


def main():
    p.check_freeze();out=p.RECORD/'RESULT.json'
    if out.exists():raise FileExistsError(out)
    raw_path=p.RECORD/'RAW_DEVELOPMENT.json';raw=json.loads(raw_path.read_text())
    source_path=p.SOURCE_RECORD/'RAW_DEVELOPMENT.json';p.checked(source_path,raw['source_raw_sha256']);source=json.loads(source_path.read_text())
    p.checked(p.RECORD/'DEV_PREACCESS.json',raw['dev_preaccess_sha256'])
    checks={};metrics={};checkpoints={};all_gates={}
    for i,seed in enumerate(p.SEEDS):
        pair=raw['results'][str(seed)];control=pair['reused_control'];candidate=pair['candidate']
        assert control==source['results'][f'seed{seed}_control']
        rows=candidate['per_case'];groups=defaultdict(list)
        for r in rows:
            assert abs(r['dice']-2*r['intersection']/(r['predicted_voxels']+r['target_voxels']))<1e-12
            groups[r['subject_id']].append(r['dice'])
        assert len(rows)==180 and len(groups)==78
        absolute=independent_ci(rows,2026091830+i)
        assert abs(candidate['scan_dice']-statistics.fmean(r['dice'] for r in rows))<1e-12
        assert abs(candidate['subject_dice']-statistics.fmean(statistics.fmean(v) for v in groups.values()))<1e-12
        assert candidate['point_hits']==sum(r['point_hit'] for r in rows)
        assert all(np.allclose(absolute[k],candidate[k],rtol=0,atol=1e-12) for k in absolute)
        delta=[]
        for a,b in zip(control['per_case'],rows,strict=True):
            assert (a['case_id'],a['subject_id'],a['target_voxels'])==(b['case_id'],b['subject_id'],b['target_voxels'])
            delta.append({**a,'dice':b['dice']-a['dice']})
        interval=independent_ci(delta,2026090830+i*100);c=raw['comparisons'][str(seed)]
        assert all(np.allclose(interval[k],c[k],rtol=0,atol=1e-12) for k in interval)
        assert abs(c['scan_delta']-(candidate['scan_dice']-control['scan_dice']))<1e-12
        assert abs(c['subject_delta']-(candidate['subject_dice']-control['subject_dice']))<1e-12
        assert c['point_delta']==candidate['point_hits']-control['point_hits']
        gates={'subject_gain':c['subject_delta']>=.003,'scan_positive':c['scan_delta']>0,'point_safety':c['point_delta']>=-1,'subject_ci_safety':interval['subject_95ci'][0]>=-.003,'scan_cluster_ci_safety':interval['scan_cluster_95ci'][0]>=-.003}
        assert gates==raw['gates'][str(seed)];all_gates[str(seed)]=gates
        report_path=p.WORK/f'seed{seed}_subject_balanced/TRAIN_DONE.json';report=json.loads(report_path.read_text())
        ckpt=Path(report['checkpoint']['path']);p.checked(ckpt,report['checkpoint']['sha256'])
        checkpoints[str(seed)]={'path':str(ckpt.relative_to(p.ROOT)),'sha256':p.sha(ckpt),'train_report_sha256':p.sha(report_path),'seconds':report['elapsed_seconds'],'peak_bytes':report['peak_gpu_memory_bytes']}
        checks[str(seed)]={'control_saved_rows_exact':True,'candidate_overlap_mean_point_and_absolute_ci_exact':True,'paired_ci_and_gates_exact':True,'checkpoint_sha_exact':True}
        metrics[str(seed)]={arm:{k:v for k,v in summary.items() if k!='per_case'} for arm,summary in pair.items()}
    average=statistics.fmean(c['scan_delta'] for c in raw['comparisons'].values())
    assert abs(average-raw['average_scan_delta'])<1e-12
    supported=all(all(g.values()) for g in all_gates.values()) and average>=.003
    assert supported==raw['all_gates_passed'] and raw['new_forwards']==360
    assert all(not value for key,value in raw['access'].items() if key!='reused_development180')
    result={'schema':'aflow.subject_balanced_sampling.result','created_utc':datetime.now(timezone.utc).isoformat(),'status':'completed_reused_development_supported' if supported else 'completed_reused_development_gate_failed','metrics':metrics,'comparisons':raw['comparisons'],'gates':all_gates,'all_gates_passed':supported,'average_scan_delta':average,'checkpoints':checkpoints,'raw_sha256':p.sha(raw_path),'source_control_raw_sha256':p.sha(source_path),'dev_preaccess_sha256':p.sha(p.RECORD/'DEV_PREACCESS.json'),'access':raw['access'],'verification':checks,'next_decision':'retain development candidate; no automatic protected-cohort access' if supported else 'close exact patient-equal policy; no temperature/mixture/epoch/seed search'}
    p.write_new(out,result)
    p.write_new(p.RECORD/'VERIFICATION_RECEIPT.json',{'status':'passed','result_sha256':p.sha(out),'raw_sha256':p.sha(raw_path),'checks':checks,'new_forwards':0,'verifier_sha256':p.sha(Path(__file__))})
    lines=['# 患者等概率采样：复用开发结果','',f"状态：{'开发门全部通过' if supported else '开发门失败，关闭当前策略'}。",'',
        '本实验只将每个扫描的原生体积权重按患者归一化；其余网络、初值、主损失、辅助系数0、168轮、两seed和1×输出与预先固定native control一致。对照复用前一实验的固定结果，新候选从零训练。两seed共用同一78名复用开发患者，不是独立外部确认。','',
        '| seed | arm | Scan Dice | Subject Dice | Point /180 |','|---|---|---:|---:|---:|']
    for seed,values in metrics.items():
        for arm,m in values.items():lines.append(f"| {seed} | {arm} | {m['scan_dice']:.6f} | {m['subject_dice']:.6f} | {m['point_hits']} |")
    lines+=['','患者聚类95%区间保留同患者全部扫描；Subject先患者内平均，Scan使用重采样患者贡献的总扫描均值。']
    for seed,c in raw['comparisons'].items():lines.append(f"\n- {seed}：Scan差{c['scan_delta']:+.6f}，CI {c['scan_cluster_95ci']}；Subject差{c['subject_delta']:+.6f}，CI {c['subject_95ci']}；Point差{c['point_delta']:+d}。")
    lines+=['',f"新增两条训练合计{sum(c['seconds'] for c in checkpoints.values())/3600:.3f} GPU运行小时，峰值{max(c['peak_bytes'] for c in checkpoints.values())/1024**3:.3f} GiB；新增dev360 forwards，未重复control推理。缓存复用，部署仍4.70M/1×。",'',
        '患者权重均匀是确定的采样性质，不能自动推出准确率提高。是否支持当前直觉由上述冻结开发门决定；失败不证明所有患者级采样无效。', '', result['next_decision'], '',
        '未访问fixed validation、outer、confirmation、internal test270或外部新模型性能。旧内部总体仍Full KD168 Flip8（8×）0.757270/0.741254/265；1×最高Scan GCSAM0.752003，Subject/Point Full0.737327/262；这些历史test分数不与本dev180直接比较。Kaggle→官方Synapse parity限制继承。', '',
        '[七问](IDEA.md)、[预注册](PREREG.md)、[设计冻结](DESIGN_FREEZE.json)、[机器结果](RESULT.json)、[raw](RAW_DEVELOPMENT.json)、[开发前收据](DEV_PREACCESS.json)、[验证](VERIFICATION_RECEIPT.json)。', '', f"Raw SHA `{p.sha(raw_path)}`；RESULT JSON SHA `{p.sha(out)}`。"]
    for seed,c in checkpoints.items():lines.append(f"\n- [{seed} checkpoint](../../../{c['path']})：SHA `{c['sha256']}`。")
    with (p.RECORD/'RESULT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'status':result['status'],'comparisons':result['comparisons'],'gates':result['gates']},indent=2))


if __name__=='__main__':main()
