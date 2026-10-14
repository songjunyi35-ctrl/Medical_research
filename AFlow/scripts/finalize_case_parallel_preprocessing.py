"""Recount and archive scheduling-only benchmark without changing scientific endpoints."""
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
from scripts import structured_background_protocol as p
from scripts.publish_research_card import publish

RECORD=p.ROOT/'archive/research_records/2026-09-11_case_parallel_preprocessing'


def main():
    if (RECORD/'RESULT.json').exists():raise FileExistsError('scheduling result already archived')
    raw=json.loads((RECORD/'RAW_BENCHMARK.json').read_text())
    pre=json.loads((RECORD/'PREACCESS.json').read_text())
    assert raw['status']=='completed' and raw['case_forwards']==pre['expected_forwards']==280
    assert raw['preaccess_sha256']==p.sha(RECORD/'PREACCESS.json')
    for name,digest in pre['files'].items():p.checked(p.ROOT/name,digest)
    p.checked(pre['model']['checkpoint'],pre['model']['sha256'])
    p.checked(p.CACHE/'index.json',pre['index_sha256'])
    assert len(raw['runs'])==4 and len(pre['sources'])==8
    ids=[r['case_id'] for r in pre['sources']]
    fingerprints={};summary=[]
    for run in raw['runs']:
        saved=json.loads((RECORD/f'RUN_{run["repeat"]}_{run["workers"]}.json').read_text())
        assert run==saved and [r['case_id'] for r in run['rows']]==ids
        assert len({r['worker_pid'] for r in run['rows']})==run['workers']
        assert run['including_shutdown_seconds']>=run['batch_wall_seconds']>=run['rows'][-1]['completed_since_start']
        assert abs(run['cases_per_second']-8/run['batch_wall_seconds'])<1e-12
        for r in run['rows']:
            assert r['input_exact'] and r['output_exact']
            assert r['completed_since_start']>=r['received_since_start']>0
            fp=(r['logit_sha256'],r['mask_sha256'],r['point'])
            fingerprints.setdefault(r['case_id'],[]).append(fp)
        summary.append({'repeat':run['repeat'],'workers':run['workers'],
            'batch_seconds':run['batch_wall_seconds'],'with_shutdown_seconds':run['including_shutdown_seconds'],
            'first_completed_seconds':run['rows'][0]['completed_since_start'],
            'cases_per_minute':60*run['cases_per_second'],
            'worker_prepare_cpu_seconds':run['worker_prepare_cpu_seconds'],
            'sampled_rss_sum_gib':run['max_sampled_parent_plus_children_rss_bytes']/1024**3,
            'single_worker_peak_rss_max_gib':max(r['worker_peak_rss_bytes'] for r in run['rows'])/1024**3})
    assert all(len(v)==4 and len(set(v))==1 for v in fingerprints.values())
    speedups={str(i):next(r['batch_seconds'] for r in summary if r['repeat']==i and r['workers']==1)/
              next(r['batch_seconds'] for r in summary if r['repeat']==i and r['workers']==4) for i in range(2)}
    assert speedups==raw['speedups'] and all(v>=1.5 for v in speedups.values())==raw['all_gates_passed']
    subjects=len({r['subject_id'] for r in pre['sources']})
    result={'status':'completed','created_utc':datetime.now(timezone.utc).isoformat(),
        'all_gates_passed':raw['all_gates_passed'],'speedups':speedups,'runs':summary,
        'scans':8,'subjects':subjects,'input_exact':32,'output_fingerprint_exact':32,
        'case_forwards':280,'training_runs':0,'seconds':raw['seconds'],'access':raw['access'],
        'raw_sha256':p.sha(RECORD/'RAW_BENCHMARK.json'),'preaccess_sha256':p.sha(RECORD/'PREACCESS.json'),
        'model':pre['model'],'new_checkpoint':False,'large_cache_created':False,
        'interpretation':'bulk scheduling with four CPU processes, not less per-case work or single-request acceleration',
        'decision':'retain bounded 4-worker bulk deployment option; validate representative workloads before default' if raw['all_gates_passed'] else 'close fixed 8-case scheduling probe; no worker-grid retuning'}
    p.write_new(RECORD/'RESULT.json',result)
    lines=['# 病例间并行预处理：在线批量工作流结果','',
           f"状态：完成；两轮固定吞吐门通过={raw['all_gates_passed']}。8扫描/{subjects}患者，只作无标签工程探针。",'',
           '## 动机与做法','',
           '此前单病例文件到输出约12秒，约97%在预处理。此次不改N4、不删镜像，固定同一native seed11模型完整八视图；只把单线程CPU工作进程从1增加到4，以有界队列按原病例顺序交给GPU。旧多进程缓存构建已有同类并行思想，这不是新算法或精度贡献。','',
           '| 轮次 | CPU进程 | 8例批量秒 | 首例完成秒 | 例/分钟 | 预处理CPU秒之和 | 采样RSS和GiB |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['repeat']} | {r['workers']} | {r['batch_seconds']:.3f} | {r['first_completed_seconds']:.3f} | {r['cases_per_minute']:.3f} | {r['worker_prepare_cpu_seconds']:.3f} | {r['sampled_rss_sum_gib']:.3f} |")
    lines.extend(['',f"两轮批量加速分别{speedups['0']:.3f}×、{speedups['1']:.3f}×；原门均需≥1.5。",'',
        '计时包含spawn池启动、源文件读取、原N4/归一化/缩放、进程传输、排队和GPU八视图至CPU输出；包含运行间的完整性采样开销。pool shutdown另记，不含权重加载/原生NIfTI导出/临床其他步骤。先SHA校验暖文件页，不称冷I/O。','',
        '## 是否保持数值和精度','',
        '32/32重建输入与冻结128缓存逐字节一致；同8例四次执行的完整float32 logit SHA、掩膜SHA和最高点坐标一致。因此这里没有改变这些已测输入的预测；没有新GT访问或Dice/Point-hit评估，不能宣称指标提升、普遍保证或新患者精度确认。','',
        '## 代价与解释','',
        '吞吐提升来自增加CPU并发、重叠不同病例的准备，不是每个病例需要更少CPU计算。每进程仍1线程，但增加了3个工作进程及运行时内存；首例响应时间和批量完成时间要分别看。RSS列是父进程与子进程采样RSS之和，会重复计共享页，既不是真实唯一物理内存，也不是持续采样的严格峰值。','',
        ('结果支持在已有多病例队列且CPU资源允许时使用有界并行部署，不能据此把它设为所有单病例请求的默认。' if raw['all_gates_passed'] else '该固定小批量流程没有稳定过吞吐门，不因数值一致就称工程提速成功；不扩worker网格救结果。'),'',
        f"0新训练、权重或大型cache；32次N4，256正式模型前向+24 warmup=280，总墙钟{raw['seconds']:.1f}秒。旧镜像剪枝内部双模型确认失败保持原样，不因本工程正负改变。",'',
        '## 指标/访问边界','',
        '没有标签、开发/test/旧outer/confirmation/外部新访问。现有reused test270/119总体与单次前向Scan仍0.757269662/0.752003176；不是本8例工程分数。本实验不新增精度CI，不能拿吞吐重复轮次当独立患者统计。','',
        '[七问](IDEA.md)、[预访问冻结](PREACCESS.json)、[原始四运行](RAW_BENCHMARK.json)、[机器汇总](RESULT.json)、[核验](VERIFICATION.json)。','',
        '[当前交接](../../../AFLOW_BRATS_HANDOFF_CURRENT.md)、[方法目录](../../../AFLOW_BRATS_METHOD_CATALOG.md)、[排行榜](../../../AFLOW_BRATS_LEADERBOARD.md)。', '',
        f"[固定checkpoint](../../../{Path(pre['model']['checkpoint']).relative_to(p.ROOT)}) SHA `{pre['model']['sha256']}`。"])
    with (RECORD/'RESULT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    p.write_new(RECORD/'VERIFICATION.json',{'status':'passed','result_sha256':p.sha(RECORD/'RESULT.json'),
        'raw_sha256':p.sha(RECORD/'RAW_BENCHMARK.json'),'markdown_sha256':p.sha(RECORD/'RESULT.md'),
        'preaccess_sha256':p.sha(RECORD/'PREACCESS.json'),'cases_four_identical_fingerprints':8,
        'run_receipt_sha256':{f'RUN_{r["repeat"]}_{r["workers"]}.json':p.sha(RECORD/f'RUN_{r["repeat"]}_{r["workers"]}.json') for r in summary},
        'scope':'saved hash/order/count/timing arithmetic audit; timings and input parity not independently rerun',
        'verifier_sha256':p.sha(Path(__file__))})
    before=(RECORD/'CARD.md').read_bytes()
    with (RECORD/'CARD.md.BEFORE_RESULT').open('xb') as f:f.write(before)
    card='## 病例间并行预处理完成：两轮吞吐门通过='+str(raw['all_gates_passed'])+'\n\n'
    card+='动机：约97%文件到输出时间在预处理。只将单线程CPU进程1→4，固定N4/输入/模型完整八视图/阈值/最高点/有界pending8与病例顺序；不恢复失败剪枝。\n\n'
    card+=f"fit8扫描/{subjects}患者，两轮批量加速{speedups['0']:.3f}/{speedups['1']:.3f}倍；32/32输入与旧cache字节一致，同8例四次完整logit/mask SHA及点完全一致。只证固定小批量调度，不称Dice提升、单病例同比加速或新算法。\n\n"
    card+='增加3个CPU进程和运行时内存；完整首例时延/批量耗时/CPU秒/采样RSS见报告。0训练/新权重/大型cache，280前向、32次N4；没有标签/开发/test/保护队列新访问。旧test270总体/1× Scan仍0.757269662/0.752003176。保留明确批量选项，不能无条件默认所有单病例请求。\n\n'
    card+='[七问](archive/research_records/2026-09-11_case_parallel_preprocessing/IDEA.md)、[结果与资源代价](archive/research_records/2026-09-11_case_parallel_preprocessing/RESULT.md)、[机器结果](archive/research_records/2026-09-11_case_parallel_preprocessing/RESULT.json)、[核验](archive/research_records/2026-09-11_case_parallel_preprocessing/VERIFICATION.json)。\n'
    assert (RECORD/'CARD.md').read_bytes()==before
    (RECORD/'CARD.md').write_text(card);publish(str(RECORD.relative_to(p.ROOT)))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
