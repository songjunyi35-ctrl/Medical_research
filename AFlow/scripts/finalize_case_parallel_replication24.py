"""Verify the new-patient workload replication and retain exact boundaries."""
from datetime import datetime,timezone
import json
from pathlib import Path
from scripts import structured_background_protocol as p
from scripts.publish_research_card import publish

RECORD=p.ROOT/'archive/research_records/2026-09-11_case_parallel_replication24'
PRIOR=p.ROOT/'archive/research_records/2026-09-11_case_parallel_preprocessing'


def main():
    if (RECORD/'RESULT.json').exists():raise FileExistsError('replication already finalized')
    raw=json.loads((RECORD/'RAW_BENCHMARK.json').read_text())
    pre=json.loads((RECORD/'PREACCESS.json').read_text())
    old=json.loads((PRIOR/'PREACCESS.json').read_text())
    derive=json.loads((RECORD/'DERIVATION.json').read_text())
    assert raw['status']=='completed' and raw['case_forwards']==pre['expected_forwards']==792
    assert raw['preaccess_sha256']==p.sha(RECORD/'PREACCESS.json')
    for name,digest in pre['files'].items():p.checked(p.ROOT/name,digest)
    p.checked(p.ROOT/derive['source'],derive['source_sha256'])
    p.checked(p.ROOT/derive['derived'],derive['derived_sha256'])
    p.checked(pre['model']['checkpoint'],pre['model']['sha256'])
    assert pre['model']==old['model']
    p.checked(p.CACHE/'index.json',pre['index_sha256'])
    ids=[r['case_id'] for r in pre['sources']]
    subjects={r['subject_id'] for r in pre['sources']};old_subjects={r['subject_id'] for r in old['sources']}
    assert len(ids)==len(set(ids))==24 and len(subjects)==12 and len(old_subjects)==4
    assert not subjects&old_subjects and set(pre['excluded_prior_subjects'])==old_subjects
    fit=p.rows(p.FIT)
    assert ids==[r['case_id'] for r in fit if r['subject_id'] not in old_subjects][:24]
    assert [(r['repeat'],r['workers']) for r in raw['runs']]==[(0,1),(0,4),(1,4),(1,1)]
    fps={};runs=[]
    for r in raw['runs']:
        path=RECORD/f'RUN_{r["repeat"]}_{r["workers"]}.json'
        assert json.loads(path.read_text())==r
        assert [v['case_id'] for v in r['rows']]==ids
        assert len({v['worker_pid'] for v in r['rows']})==r['workers']
        assert r['including_shutdown_seconds']>=r['batch_wall_seconds']>=r['rows'][-1]['completed_since_start']
        assert abs(24/r['batch_wall_seconds']-r['cases_per_second'])<1e-12
        for v in r['rows']:
            assert v['input_exact'] and v['output_exact'] and v['completed_since_start']>=v['received_since_start']>0
            fps.setdefault(v['case_id'],[]).append((v['logit_sha256'],v['mask_sha256'],v['point']))
        runs.append({'repeat':r['repeat'],'workers':r['workers'],'batch_seconds':r['batch_wall_seconds'],
            'first_output_seconds':r['rows'][0]['completed_since_start'],
            'including_shutdown_seconds':r['including_shutdown_seconds'],'cases_per_minute':60*r['cases_per_second'],
            'worker_prepare_cpu_seconds':r['worker_prepare_cpu_seconds'],
            'sampled_rss_sum_gib':r['max_sampled_parent_plus_children_rss_bytes']/1024**3,
            'single_worker_peak_rss_max_gib':max(v['worker_peak_rss_bytes'] for v in r['rows'])/1024**3,
            'run_sha256':p.sha(path)})
    assert len(fps)==24 and all(len(v)==4 and len(set(v))==1 for v in fps.values())
    speedups={str(i):next(r['batch_seconds'] for r in runs if r['repeat']==i and r['workers']==1)/
                     next(r['batch_seconds'] for r in runs if r['repeat']==i and r['workers']==4) for i in range(2)}
    assert speedups==raw['speedups'] and all(v>=1.5 for v in speedups.values())==raw['all_gates_passed']
    result={'status':'completed','created_utc':datetime.now(timezone.utc).isoformat(),
        'all_gates_passed':raw['all_gates_passed'],'speedups':speedups,'runs':runs,
        'scans':24,'subjects':12,'prior_patient_overlap':0,'input_exact':96,'output_fingerprint_exact':96,
        'case_forwards':792,'new_training_runs':0,'new_checkpoints':0,'new_large_cache':False,
        'seconds':raw['seconds'],'access':raw['access'],'model':pre['model'],
        'raw_sha256':p.sha(RECORD/'RAW_BENCHMARK.json'),'preaccess_sha256':p.sha(RECORD/'PREACCESS.json'),
        'derivation_sha256':p.sha(RECORD/'DERIVATION.json'),
        'prior_result_sha256':p.sha(PRIOR/'RESULT.json'),
        'interpretation':'different fit patients and larger batch; engineering replication, not clinical external validation',
        'decision':'retain 4-worker bounded batch option; no per-case latency or accuracy improvement claim' if raw['all_gates_passed'] else 'close this fixed workload replication, no worker-grid retuning'}
    p.write_new(RECORD/'RESULT.json',result)
    lines=['# 不同fit患者上的24扫描批量复制','',
        f"状态：固定两轮复制完成，全部门通过={raw['all_gates_passed']}。24扫描/12患者，与上轮8扫描/4患者的患者交集为0。",'',
        '## 问题、直觉与控制','',
        '上轮小批量吞吐2.852/2.950倍，不能直接当其他工作负载的保证。此次排除旧患者并增加到24扫描，验证同一个有界队列在持续补充任务时的行为。每轮只比较CPU进程1→4；每进程仍单线程原N4，pending8、固定native seed11模型完整八视图、输入/融合/阈值/点均不变。没有重开已失败镜像剪枝。','',
        '| 轮次 | CPU进程 | 24例批量秒 | 首例秒 | 例/分钟 | 预处理CPU秒 | 采样RSS和GiB |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in runs:lines.append(f"| {r['repeat']} | {r['workers']} | {r['batch_seconds']:.3f} | {r['first_output_seconds']:.3f} | {r['cases_per_minute']:.3f} | {r['worker_prepare_cpu_seconds']:.3f} | {r['sampled_rss_sum_gib']:.3f} |")
    lines.extend(['',f"两轮批量加速{speedups['0']:.3f}/{speedups['1']:.3f}倍，冻结门均为≥1.5。",'',
        '96/96输入重建与原cache逐字节一致，24病例四运行完整logit SHA、mask SHA和最高点均相同。存量记录重算顺序/计数/倍率通过。这里只检查预测相等，不读取GT，因此没有新的Dice或Point-hit精度验证。','',
        '## 科研解释与成本','',
        ('在不同fit患者和更大批量下复现了工程吞吐收益，支持保留4进程有界在线批量选项；仍不支持单病例同比加速、CPU计算量减少或医学泛化。' if raw['all_gates_passed'] else '复制没有稳定通过原速度门；旧小批量结果仍保留，但不能扩大为普适收益，不按新结果搜索进程/队列参数。'),'',
        '与旧探针相比同时改变患者负载和批量大小，不能把两次倍率差归因于其中单一因素。四进程使用更多CPU与运行时内存；RSS和含共享页，是输出时刻采样，不是唯一物理内存或严格峰值。','',
        'wall包括spawn启动、文件读取、N4/裁剪/归一化、IPC、排队、GPU八视图至CPU输出及运行间审计开销；关闭进程池另记，不含模型加载、原生NIfTI导出、临床其他流程。先SHA读取暖文件页，不称冷I/O。两轮不是独立患者统计，不给医学CI。','',
        f"0训练、新权重、大型cache；96次N4，768正式模型前向+24 warmup=792，墙钟{raw['seconds']:.1f}秒。仅fit图像，无标签/开发/test/旧outer/confirmation/外部访问。",'',
        '现有reused internal test270/119总体Scan0.757269662、单次前向Scan0.752003176未改变；这些是旧精度榜，不是本24例的分数。不同fit患者不等于外部精度验证。','',
        '## 证据','',
        '[七问](IDEA.md)、[代码派生收据](DERIVATION.json)、[源代码差异](DERIVATION.patch)、[固定队列与预访问](PREACCESS.json)、[四次原始运行](RAW_BENCHMARK.json)、[机器结果](RESULT.json)、[核验](VERIFICATION.json)。','',
        '[前一小批量结果](../2026-09-11_case_parallel_preprocessing/RESULT.md)、[当前交接](../../../AFLOW_BRATS_HANDOFF_CURRENT.md)、[方法目录](../../../AFLOW_BRATS_METHOD_CATALOG.md)、[排行榜](../../../AFLOW_BRATS_LEADERBOARD.md)。', '',
        f"[未改变的checkpoint](../../../{Path(pre['model']['checkpoint']).relative_to(p.ROOT)}) SHA `{pre['model']['sha256']}`。"])
    with (RECORD/'RESULT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    p.write_new(RECORD/'VERIFICATION.json',{'status':'passed','result_sha256':p.sha(RECORD/'RESULT.json'),
        'raw_sha256':p.sha(RECORD/'RAW_BENCHMARK.json'),'markdown_sha256':p.sha(RECORD/'RESULT.md'),
        'preaccess_sha256':p.sha(RECORD/'PREACCESS.json'),'derivation_sha256':p.sha(RECORD/'DERIVATION.json'),
        'patient_overlap_with_prior':0,'24_cases_four_identical_fingerprints':True,
        'scope':'saved hash/order/count/latency arithmetic audit, not independent runtime reruns',
        'verifier_sha256':p.sha(Path(__file__))})
    before=(RECORD/'CARD.md').read_bytes()
    with (RECORD/'CARD.md.BEFORE_RESULT').open('xb') as f:f.write(before)
    card='## 24扫描不同fit患者批量复制完成：固定门通过='+str(raw['all_gates_passed'])+'\n\n'
    card+='动机：旧8例工程探针不足以保证其他工作负载；排除旧4患者，按清单取24扫描/12患者。只比较CPU进程1→4，单线程N4/pending8/固定native模型八视图/输入阈值与点不变，旧SHA绑定代码保留。\n\n'
    card+=f"两轮批量加速{speedups['0']:.3f}/{speedups['1']:.3f}倍；96/96输入byte exact，24病例四次完整logit/mask/point一致。{'支持更大批量下的工程复制' if raw['all_gates_passed'] else '未稳定通过更大负载的原门'}，不称单病例延迟同比改善、CPU量减少或精度提升。首例时延与CPU/RSS代价完整报告。\n\n"
    card+='0训练/新checkpoint/大cache，792模型前向、96次N4；仅fit图像，无标签/开发/test/保护队列访问，旧test270总体/1× Scan仍0.757269662/0.752003176。两个不同fit队列不是外部临床验证；不按结果改进程网格。\n\n'
    card+='[七问](archive/research_records/2026-09-11_case_parallel_replication24/IDEA.md)、[完整结果](archive/research_records/2026-09-11_case_parallel_replication24/RESULT.md)、[机器结果](archive/research_records/2026-09-11_case_parallel_replication24/RESULT.json)、[独立账目核验](archive/research_records/2026-09-11_case_parallel_replication24/VERIFICATION.json)。\n'
    assert (RECORD/'CARD.md').read_bytes()==before
    (RECORD/'CARD.md').write_text(card);publish(str(RECORD.relative_to(p.ROOT)))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
