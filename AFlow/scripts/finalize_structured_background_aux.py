"""Verify saved development counts and archive the fixed two-seed experiment."""
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import numpy as np
from scripts import structured_background_protocol as p


def independent_ci(rows,seed):
    subjects=sorted({r['subject_id'] for r in rows})
    values=[[r['dice'] for r in rows if r['subject_id']==s] for s in subjects]
    totals=np.asarray([sum(v) for v in values]); sizes=np.asarray([len(v) for v in values])
    rng=np.random.default_rng(seed)
    samples=rng.integers(0,len(subjects),(10000,len(subjects)))
    return {'subject_95ci':np.quantile((totals/sizes)[samples].mean(1),[.025,.975]).tolist(),
            'scan_cluster_95ci':np.quantile(totals[samples].sum(1)/sizes[samples].sum(1),[.025,.975]).tolist()}


def main():
    p.check_freeze()
    raw_path=p.RECORD/'RAW_DEVELOPMENT.json'
    raw=json.loads(raw_path.read_text())
    if (p.RECORD/'RESULT.json').exists():
        raise FileExistsError('result already archived')
    pre=p.RECORD/'DEV_PREACCESS.json'
    p.checked(pre,raw['dev_preaccess_sha256'])
    _,dev=p.check_roles();expected=[r['case_id'] for r in dev]
    checks={};summary={};checkpoint_hashes={}
    for ordinal,(key,model) in enumerate(raw['results'].items()):
        # Ordinal used by evaluation is seed then arm (control, candidate), not JSON sort order.
        seed_index=p.SEEDS.index(int(key.split('_')[0][4:]));arm=key.split('_')[1]
        ordinal=seed_index*2+p.ARMS.index(arm)
        rows=model['per_case']
        assert [r['case_id'] for r in rows]==expected
        assert all(abs(r['dice']-2*r['intersection']/(r['predicted_voxels']+r['target_voxels']))<1e-12 for r in rows)
        groups=defaultdict(list)
        for r in rows:groups[r['subject_id']].append(r['dice'])
        intervals=independent_ci(rows,2026090730+1000+ordinal)
        checks[key]={'scan_exact':abs(statistics.fmean(r['dice'] for r in rows)-model['scan_dice'])<1e-12,'subject_exact':abs(statistics.fmean(statistics.fmean(v) for v in groups.values())-model['subject_dice'])<1e-12,'point_exact':sum(r['point_hit'] for r in rows)==model['point_hits'],'ci_exact':all(np.allclose(intervals[k],model[k],rtol=0,atol=1e-12) for k in intervals)}
        report=json.loads((p.WORK/key/'TRAIN_DONE.json').read_text())
        path=Path(report['checkpoint']['path']);p.checked(path,report['checkpoint']['sha256'])
        checkpoint_hashes[key]={'path':str(path.relative_to(p.ROOT)),'sha256':p.sha(path),'train_report_sha256':p.sha(p.WORK/key/'TRAIN_DONE.json')}
        summary[key]={k:v for k,v in model.items() if k!='per_case'}
    recomputed_gates={}
    for i,seed in enumerate(p.SEEDS):
        control=raw['results'][f'seed{seed}_control'];candidate=raw['results'][f'seed{seed}_candidate']
        delta=[{**a,'dice':b['dice']-a['dice']} for a,b in zip(control['per_case'],candidate['per_case'],strict=True)]
        intervals=independent_ci(delta,2026090730+i*100)
        c=raw['comparisons'][str(seed)]
        assert all(np.allclose(intervals[k],c[k],rtol=0,atol=1e-12) for k in intervals)
        sd=candidate['subject_dice']-control['subject_dice'];scan=candidate['scan_dice']-control['scan_dice'];points=candidate['point_hits']-control['point_hits']
        assert abs(sd-c['subject_delta'])<1e-12 and abs(scan-c['scan_delta'])<1e-12 and points==c['point_delta']
        recomputed_gates[str(seed)]={'subject_gain':sd>=.003,'scan_positive':scan>0,'point_safety':points>=-1,'subject_ci_safety':intervals['subject_95ci'][0]>=-.003,'scan_cluster_ci_safety':intervals['scan_cluster_95ci'][0]>=-.003}
    assert recomputed_gates==raw['gates'] and all(all(g.values()) for g in checks.values())
    assert raw['forwards']==720
    supported=all(all(g.values()) for g in recomputed_gates.values()) and raw['average_scan_delta']>=.003
    assert supported==raw['all_gates_passed']
    assert all(not v for k,v in raw['access'].items() if k!='reused_development180')
    result={'schema':'aflow.structured_background_aux.result','created_utc':datetime.now(timezone.utc).isoformat(),'status':'completed_reused_development_supported' if supported else 'completed_reused_development_gate_failed','metrics':summary,'comparisons':raw['comparisons'],'gates':raw['gates'],'all_gates_passed':supported,'checkpoint_hashes':checkpoint_hashes,'raw_sha256':p.sha(raw_path),'dev_preaccess_sha256':p.sha(pre),'fit_sha256':p.FIT_SHA,'dev_sha256':p.DEV_SHA,'access':raw['access'],'independent_verification':checks,'interpretation':'two optimization seeds on the same historically reused patient split; not independent external confirmation','next_decision':'no automatic test access; retain candidate for later preregistered investigation' if supported else 'close weight0.1/current recipe without epoch/seed/weight search'}
    p.write_new(p.RECORD/'RESULT.json',result)
    p.write_new(p.RECORD/'VERIFICATION_RECEIPT.json',{'status':'passed','result_sha256':p.sha(p.RECORD/'RESULT.json'),'raw_sha256':p.sha(raw_path),'checks':checks,'paired_gates_recomputed':True,'checkpoint_hashes':checkpoint_hashes,'new_forwards':0,'verifier_sha256':p.sha(Path(__file__))})
    lines=['# 训练期切除腔辅助监督：复用开发结果','',f"状态：{'开发门全部通过' if supported else '开发门失败，关闭当前配方'}。",'',
           '本实验只测试额外训练监督是否改善WT；两臂均从零、无教师、同结构、同fit和计算预算，唯一因素为辅助权重0→0.1。开发集180 scans/78人来自用户批准的复用开发池，两个seed共享同一批患者，不能称两次独立泛化验证。','',
           '| seed/arm | Scan Dice | Subject Dice | Point /180 |', '|---|---:|---:|---:|']
    for key,m in summary.items():lines.append(f"| {key} | {m['scan_dice']:.6f} | {m['subject_dice']:.6f} | {m['point_hits']} |")
    lines+=['','配对95%区间按患者重采样，每位患者全部扫描一起进入Scan统计；Subject先患者内平均。']
    for seed,c in raw['comparisons'].items():lines.append(f"\n- Seed {seed}：Scan差{c['scan_delta']:+.6f}，cluster CI {c['scan_cluster_95ci']}；Subject差{c['subject_delta']:+.6f}，CI {c['subject_95ci']}；Point差{c['point_delta']:+d}。")
    training=[json.loads((p.WORK/key/'TRAIN_DONE.json').read_text()) for key in summary]
    lines+=['','## 成本、解释和下一步','',f"四条训练合计{sum(r['elapsed_seconds'] for r in training)/3600:.3f} GPU运行小时，峰值{max(r['peak_gpu_memory_bytes'] for r in training)/1024**3:.3f} GiB。辅助uint8缓存约1.34GiB；部署裁除辅助输出，4.70M参数、1×前向。开发评测720 forwards。",'',
            '结果只能说明当前0.1辅助监督在该复用开发协议的表现。辅助任务学得更好不自动意味着WT更好；不能由此更改已揭示外部队列后处理或重跑test。', '',
            result['next_decision'], '',
            '历史内部总体点估计仍为Full KD168固定Flip8（8×）Scan0.757270/Subject0.741254/Point265；内部1×最高Scan为GCSAM0.752003、最高Subject/Point为Full0.737327/262。这些旧test结果与本开发集不直接比较。', '',
            '本实验没有访问fixed validation108、outer、confirmation、internal test270或任何外部模型性能。训练源Kaggle→官方Synapse字节parity限制继承。','',
            '## 证据','', '[七问](IDEA.md)、[预注册](PREREG.md)、[机器结果](RESULT.json)、[raw](RAW_DEVELOPMENT.json)、[开发前收据](DEV_PREACCESS.json)、[独立重算](VERIFICATION_RECEIPT.json)。', '',
            f"Raw SHA256 `{p.sha(raw_path)}`；RESULT JSON SHA256 `{p.sha(p.RECORD/'RESULT.json')}`。"]
    for key,rec in checkpoint_hashes.items():lines.append(f"\n- [{key} checkpoint](../../../{rec['path']})：SHA256 `{rec['sha256']}`。")
    with (p.RECORD/'RESULT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'status':result['status'],'comparisons':result['comparisons'],'gates':result['gates']},indent=2))


if __name__=='__main__':
    main()
