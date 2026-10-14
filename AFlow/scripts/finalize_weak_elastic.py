"""Independent saved-count, clustered-CI, gate, hash and development-point audit."""
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import numpy as np
from scripts import weak_elastic_protocol as p
from scripts.finalize_structured_background_aux import independent_ci


def main():
    p.check_freeze();out=p.RECORD/'RESULT.json'
    if out.exists():raise FileExistsError(out)
    raw_path=p.RECORD/'RAW_DEVELOPMENT.json';raw=json.loads(raw_path.read_text())
    pre=p.RECORD/'DEV_PREACCESS.json';p.checked(pre,raw['dev_preaccess_sha256']);receipt=json.loads(pre.read_text())
    for name,digest in receipt['files'].items():p.checked(p.ROOT/name,digest)
    replay_path=p.RECORD/'TRAINING_REPLAY_FINAL.json';replay=json.loads(replay_path.read_text());assert replay['status']=='passed_full168' and len(replay['results'])==4
    sources={}
    for key,value in receipt['source_raws'].items():
        p.checked(Path(value['path']),value['sha256']);sources[key]=json.loads(Path(value['path']).read_text())
    cache=json.loads((p.RECORD/'AUX_CACHE_RECEIPT.json').read_text());p.checked(p.CACHE/'labels.u8',cache['source_labels_sha256'])
    index=json.loads((p.CACHE/'index.json').read_text());mapping={c:i for i,c in enumerate(index['case_ids'])}
    labels=np.memmap(p.CACHE/'labels.u8',mode='r',dtype=np.uint8,shape=(972,128,128,128))
    _,dev=p.check_roles();truth={r['case_id']:int(labels[mapping[r['case_id']]].sum()) for r in dev}
    metrics={};comparisons={};gates={};checkpoints={};checks={}
    for i,seed in enumerate(p.SEEDS):
        native=sources['native']['results'][f'seed{seed}_control'];subject=sources['subject']['results'][str(seed)]['candidate']
        for arm in p.ARMS:
            key=f'{seed}_{arm}';candidate=raw['results'][key];rows=candidate['per_case'];groups=defaultdict(list)
            assert [(r['case_id'],r['subject_id']) for r in rows]==[(r['case_id'],r['subject_id']) for r in dev]
            for r in rows:
                assert r['target_voxels']==truth[r['case_id']]
                assert 0<=r['intersection']<=min(r['predicted_voxels'],r['target_voxels'])
                assert abs(r['dice']-2*r['intersection']/(r['predicted_voxels']+r['target_voxels']))<1e-12
                assert bool(labels[mapping[r['case_id']]].reshape(-1)[r['point_flat_index']])==r['point_hit']
                groups[r['subject_id']].append(r['dice'])
            assert len(rows)==180 and len(groups)==78
            absolute=independent_ci(rows,2026091011+i)
            assert abs(candidate['scan_dice']-statistics.fmean(r['dice'] for r in rows))<1e-12
            assert abs(candidate['subject_dice']-statistics.fmean(statistics.fmean(v) for v in groups.values()))<1e-12
            assert candidate['point_hits']==sum(r['point_hit'] for r in rows)
            assert all(np.allclose(absolute[k],candidate[k],rtol=0,atol=1e-12) for k in absolute)
            comparisons[key]={}
            for contrast,control in [('versus_native',native),('versus_own_sampling_control',subject if arm=='subject_elastic' else native)]:
                delta=[]
                for a,b in zip(control['per_case'],rows,strict=True):
                    assert (a['case_id'],a['subject_id'],a['target_voxels'])==(b['case_id'],b['subject_id'],b['target_voxels'])
                    delta.append({**a,'dice':b['dice']-a['dice']})
                interval=independent_ci(delta,2026091011+i);saved=raw['comparisons'][key][contrast]
                c={'scan_delta':candidate['scan_dice']-control['scan_dice'],'subject_delta':candidate['subject_dice']-control['subject_dice'],'point_delta':candidate['point_hits']-control['point_hits'],**interval}
                assert all(np.allclose(v,saved[k],rtol=0,atol=1e-12) for k,v in c.items())
                assert saved['improved']==sum(r['dice']>0 for r in delta) and saved['worse']==sum(r['dice']<0 for r in delta) and saved['tied']==sum(r['dice']==0 for r in delta)
                comparisons[key][contrast]=saved
            c=comparisons[key]['versus_native'];own=comparisons[key]['versus_own_sampling_control']
            g={'subject_gain':c['subject_delta']>=.003,'scan_positive':c['scan_delta']>0,'point_safety':c['point_delta']>=-1,'subject_ci_safety':c['subject_95ci'][0]>=-.003,'scan_cluster_ci_safety':c['scan_cluster_95ci'][0]>=-.003}
            if arm=='subject_elastic':g.update(own_subject_positive=own['subject_delta']>0,own_scan_positive=own['scan_delta']>0,own_point_nonnegative=own['point_delta']>=0)
            assert g==raw['gates'][key];gates[key]=g
            report_info=receipt['checkpoints_reports'][key]['train_report'];path=Path(report_info['path']);p.checked(path,report_info['sha256']);report=json.loads(path.read_text());ckpt=Path(report['checkpoint']['path']);p.checked(ckpt,report['checkpoint']['sha256'])
            checkpoints[key]={'path':str(ckpt.relative_to(p.ROOT)),'sha256':p.sha(ckpt),'train_report_sha256':p.sha(path),'seconds':report['elapsed_seconds'],'peak_bytes':report['peak_gpu_memory_bytes']}
            metrics[key]={k:v for k,v in candidate.items() if k!='per_case'}
            checks[key]={'patient_order_and_saved_count_arithmetic':True,'saved_point_ground_truth_hit_verified':True,'argmax_not_independently_recomputed':True,'absolute_and_paired_ci':True,'frozen_gates':True,'checkpoint_sha':True,'full168_metadata_replay':True}
    arm_gates={}
    for arm in p.ARMS:
        avg=statistics.fmean(comparisons[f'{seed}_{arm}']['versus_native']['scan_delta'] for seed in p.SEEDS)
        allseed=all(all(gates[f'{seed}_{arm}'].values()) for seed in p.SEEDS)
        arm_gates[arm]={'all_seed_gates':allseed,'average_scan_delta':avg,'average_scan_gain_gate':avg>=.003,'eligible':allseed and avg>=.003}
    eligible=[a for a in p.ARMS if arm_gates[a]['eligible']]
    chosen=max(eligible,key=lambda arm:statistics.fmean(metrics[f'{s}_{arm}']['subject_dice'] for s in p.SEEDS)) if eligible else None
    assert arm_gates==raw['arm_gates'] and chosen==raw['selected_arm'] and raw['new_forwards']==720
    assert not any(raw['access'][key] for key in ['test270','fixed_validation108','outer','confirmation','external'])
    result={'schema':'aflow.weak_elastic.result','created_utc':datetime.now(timezone.utc).isoformat(),'status':'completed_reused_development_supported' if chosen else 'completed_reused_development_gate_failed','metrics':metrics,'comparisons':comparisons,'gates':gates,'arm_gates':arm_gates,'selected_arm':chosen,'checkpoints':checkpoints,'raw_sha256':p.sha(raw_path),'dev_preaccess_sha256':p.sha(pre),'training_replay_sha256':p.sha(replay_path),'cache_index_sha256':p.sha(p.CACHE/'index.json'),'labels_sha256':p.sha(p.CACHE/'labels.u8'),'access':raw['access'],'verification':checks,'next_decision':'conditional_full864_refit_requires_own_admission' if chosen else 'close_both_fixed_weak_elastic_recipes_no_strength_seed_epoch_search'}
    p.write_new(out,result)
    p.write_new(p.RECORD/'VERIFICATION_RECEIPT.json',{'status':'passed','result_sha256':p.sha(out),'raw_sha256':p.sha(raw_path),'preaccess_sha256':p.sha(pre),'checks':checks,'new_forwards':0,'reused_development_label_reread':True,'verifier_sha256':p.sha(Path(__file__))})
    lines=['# 弱弹性形变双抽样层：复用开发结果','',f"状态：{result['status']}；唯一入选层：{chosen or '无'}。",'',
           '每个固定抽样层仅追加p0.2/最大2输入体素的平滑形变，原模型、初值、168轮、基础增强、主损失、1×输出不变。两个seed共用同一180扫描/78患者reused development，不是两份独立确认。无教师；历史四个无形变对照复用已揭示记录。','',
           '| seed/新增臂 | Scan Dice | Subject Dice | Point /180 |','|---|---:|---:|---:|']
    for key,m in metrics.items():lines.append(f"| {key} | {m['scan_dice']:.6f} | {m['subject_dice']:.6f} | {m['point_hits']} |")
    lines+=['','原native control11/12：Scan0.746762/0.744177、Subject0.733213/0.729318、Point170/163；原subject control11/12：Scan0.753475/0.755906、Subject0.740633/0.742725、Point167/168。','',
            '两Dice采用10,000次患者簇配对百分位95%CI；探索性两候选筛选，未校正多比较。Point列为计数差，不据平均值掩盖单seed失败。']
    for key,contrasts in comparisons.items():
        for name,c in contrasts.items():lines.append(f"\n- {key} {name}：Scan {c['scan_delta']:+.6f} CI {c['scan_cluster_95ci']}；Subject {c['subject_delta']:+.6f} CI {c['subject_95ci']}；Point {c['point_delta']:+d}。")
    lines+=['',f"新增四训练{sum(c['seconds'] for c in checkpoints.values())/3600:.3f} GPU小时，峰值{max(c['peak_bytes'] for c in checkpoints.values())/1024**3:.3f}GiB；开发720 forwards/{raw['seconds']:.2f}s；无新影像cache，模型仍4.70M/1×。",'',
            '工程证明实现的形变/随机序列如预期，不自动证明泛化获益。冻结联合门决定本配方是否支持晋级；失败不证明所有空间增强无效。详细机制解释与后续决定在INTERPRETATION.md完成后记录。','',
            '没有新增test/outer/confirmation/fixed validation/external性能；旧test270/119 overall仍Full972KD168 Flip8 Scan0.757269662，1×Scan仍GCSAM0.752003176。它们不与本dev180分数直接比较。训练源Kaggle→官方字节parity未知限制保留。','',
            '[七问](IDEA.md)、[预注册](PREREG.md)、[raw](RAW_DEVELOPMENT.json)、[preaccess](DEV_PREACCESS.json)、[机器结果](RESULT.json)、[验证](VERIFICATION_RECEIPT.json)、[全训练重放](TRAINING_REPLAY_FINAL.json)。','',f'Raw SHA `{p.sha(raw_path)}`；RESULT JSON SHA `{p.sha(out)}`。']
    for key,c in checkpoints.items():lines.append(f"\n- [{key} checkpoint](../../../{c['path']})：SHA `{c['sha256']}`。")
    with (p.RECORD/'RESULT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'status':result['status'],'metrics':metrics,'arm_gates':arm_gates,'selected_arm':chosen},indent=2),flush=True)


if __name__=='__main__':main()
