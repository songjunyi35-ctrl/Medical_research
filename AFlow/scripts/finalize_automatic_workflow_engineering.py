"""Independently recount benchmark rows and publish the bounded engineering result."""
import json
from pathlib import Path
import statistics
from datetime import datetime, timezone
from scripts import structured_background_protocol as p
from scripts.publish_research_card import publish

RECORD = p.ROOT/'archive/research_records/2026-09-11_automatic_workflow_engineering'


def main():
    raw = json.loads((RECORD/'RAW_BENCHMARK.json').read_text())
    pre = json.loads((RECORD/'PREACCESS.json').read_text())
    assert raw['status'] == 'completed'
    assert raw['preaccess_sha256'] == p.sha(RECORD/'PREACCESS.json')
    for name, digest in pre['files'].items():
        p.checked(p.ROOT/name, digest)
    for report in pre['models']:
        p.checked(report['checkpoint'], report['sha256'])
    assert len(raw['rows']) == 384 and len(raw['parity']) == 16
    assert all(r['max_abs_error'] == 0 for r in raw['parity'])
    assert set(raw['access']) == {'fit_images','labels','development','test','outer','confirmation','external'}
    assert raw['access']['fit_images'] and sum(raw['access'].values()) == 1
    assert raw['case_forwards'] == 1654 and not raw['accuracy_promotion']
    unique = {(r['seed'],r['case_id'],r['repeat'],r['workflow']) for r in raw['rows']}
    assert len(unique) == 384
    assert {r['case_id'] for r in raw['rows']} == set(pre['cases'])
    assert sum(r['case_forwards'] for r in raw['rows']) == 1440
    for seed, summary in raw['summary'].items():
        subset = [r for r in raw['rows'] if str(r['seed']) == seed]
        lookup = {(r['case_id'],r['repeat'],r['workflow']):r['seconds'] for r in subset}
        for name, stats in summary.items():
            rows = [r for r in subset if r['workflow'] == name]
            assert len(rows) == 48
            assert all(r['seconds'] > 0 and r['case_forwards'] == len(pre['views'][name]) for r in rows)
            mean = sum(r['seconds'] for r in rows)/48
            median = statistics.median(r['seconds'] for r in rows)
            ratios = [lookup[(r['case_id'],r['repeat'],'full8')]/r['seconds'] for r in rows]
            assert abs(mean-stats['mean_seconds']) < 1e-12
            assert median == stats['median_seconds']
            assert statistics.median(ratios) == stats['paired_median_speedup_vs_full8']
        assert raw['gates'][seed] == {
            'identity_2x': summary['identity']['paired_median_speedup_vs_full8'] >= 2,
            'four_1_2x': summary['single_axis4']['paired_median_speedup_vs_full8'] >= 1.2}
    assert raw['all_gates_passed'] == all(all(g.values()) for g in raw['gates'].values())
    result = {k:v for k,v in raw.items() if k not in ('rows','parity')}
    result.update({'raw_sha256': p.sha(RECORD/'RAW_BENCHMARK.json'),
                   'interpretation': 'engineering execution evidence only; no accuracy/clinical superiority established',
                   'next_priority': 'separately preregister accuracy-constrained workflow search on reused development; no test admission'})
    p.write_new(RECORD/'RESULT.json', result)
    lines = ['# 自动模型固定工作流：工程闭环结果', '',
             '状态：工程执行完成；不是工作流搜索完整科研闭环，也不是精度晋级。', '',
             '## 做了什么', '',
             '只改变镜像集合，固定两个fit684/313、epoch168自动模型；训练侧前24图像，无标签。1/2/4/8视图均真实按需执行；16个在线/原子子集检查逐体素exact，历史identity/full8与原子融合也exact。没有训练、开发或test性能访问。', '',
             '| Seed | 工作流 | 平均秒/例 | 配对中位加速（相对full8） |',
             '|---|---|---:|---:|']
    for seed, summary in raw['summary'].items():
        for name, stats in summary.items():
            lines.append(f"| {seed} | {name} | {stats['mean_seconds']:.6f} | {stats['paired_median_speedup_vs_full8']:.3f}× |")
    lines.extend(['', f"工程速度门全部通过：{raw['all_gates_passed']}。GPU {raw['hardware']}；推理阶段墙钟{raw['gpu_phase_seconds']:.2f}秒；共1654病例前向，其中1440计时前向、90预热、124一致性检查。", '',
                  f"峰值allocated显存：{max(m['peak_allocated_bytes'] for m in raw['models'])/1024**3:.3f} GiB。0新训练、0新checkpoint、无持久大型预测缓存。", '',
                  '计时包含主机图像tensor到GPU、镜像前向融合、阈值/最高点、掩膜和点返回CPU；不包含文件读取、NIfTI预处理或模型加载。因此只能报告当前硬件的推理阶段收益，不叫端到端收益。重复测速不是独立患者，不给临床精度CI。', '',
                  '## 直觉支持与限制', '',
                  '实现确实只做选中的视图，数值结果与完整原子中的相同子集一致。速度门通过支持减少执行次数能节省本实现的推理耗时；没有标签，无法知道剪枝后Dice和Point是否安全，不能推荐部署或宣称超过排行榜。两个seed是同训练/开发来源，不是外部复制。', '',
                  '## 下一优先级', '',
                  '正式搜索前另冻协议：固定两个native模型和复用dev180/78，按患者分离搜索/选择/评估；明确重复开发而非新盲测，保留identity/full8参照；比较有限预算搜索与随机/贪心，并用完整有限空间作搜索质量参照。只在精度、点与实际成本联合门通过时才讨论后续确认，不自动开启test。旧subject-flip8负结果不复活，不用旧外部结果选视图。', '',
                  '本阶段完成的是该主线的执行和测速基础，不声称已经完成搜索算法比较、开发准确率验证或最终确认。', '',
                  '## 当前同队列纪录（仅历史背景）', '',
                  '复用非盲internal test270/119总体Scan0.757269662、Subject0.741253585、Point265/270（8×、一个checkpoint）；单次前向Scan0.752003176、Subject0.737266420、Point252/270。不是官方隐藏榜；本工程未改变排名。总体Scan绝对95%CI[0.739089,0.774152]，Subject[0.714109,0.766152]；不能将历史CI套在本轮测速上。', '',
                  '## 证据', '',
                  '- [七问/冻结边界](IDEA.md)、[前访问收据](PREACCESS.json)、[原始测速](RAW_BENCHMARK.json)、[机器结果](RESULT.json)、[验证](VERIFICATION.json)',
                  '- [当前交接](../../../AFLOW_BRATS_HANDOFF_CURRENT.md)、[目录](../../../AFLOW_BRATS_METHOD_CATALOG.md)、[排行榜](../../../AFLOW_BRATS_LEADERBOARD.md)',
                  '- [官方镜像推理实现](https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/nnunetv2/inference/predict_from_raw_data.py)仅支持镜像融合实现依据，不支持精度剪枝收益。'])
    for report in pre['models']:
        relative = Path(report['checkpoint']).relative_to(p.ROOT)
        lines.append(f"- [固定seed{report['seed']} checkpoint](../../../{relative})，SHA `{report['sha256']}`")
    with (RECORD/'RESULT.md').open('x') as f:
        f.write('\n'.join(lines)+'\n')
    receipt = {'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),
               'raw_sha256':p.sha(RECORD/'RAW_BENCHMARK.json'), 'result_sha256':p.sha(RECORD/'RESULT.json'),
               'markdown_sha256':p.sha(RECORD/'RESULT.md'),'preaccess_sha256':p.sha(RECORD/'PREACCESS.json'),
               'verifier_sha256':p.sha(Path(__file__)), 'raw_rows_recounted':384,
               'scope':'independent count/summary/hash check; GPU tensor parity was asserted by original benchmark, not replayed here',
               'checkpoints':pre['models']}
    p.write_new(RECORD/'VERIFICATION.json', receipt)
    old = (RECORD/'CARD.md').read_bytes()
    with (RECORD/'CARD.md.BEFORE_RESULT').open('xb') as f:
        f.write(old)
    card = '\n'.join(lines[:4])+ '\n\n'
    card += '原预注册七问保持不变。仅改变视图集合；无训练、标签、开发/test/outer/confirmation/外部性能访问。\n\n'
    for seed, summary in raw['summary'].items():
        card += f"Seed {seed}：1×/2×/4×相对full8配对中位加速为" + '/'.join(f"{summary[n]['paired_median_speedup_vs_full8']:.3f}倍" for n in ('identity','opposite2','single_axis4'))+'。\n\n'
    card += f"16项GPU子集parity精确一致，实际1654病例前向；工程速度门通过={raw['all_gates_passed']}。当前硬件推理阶段{raw['gpu_phase_seconds']:.1f}秒，不包含预处理/文件加载，不是端到端加速。无新checkpoint或大型cache。支持按需执行确实省计算，不证明剪枝精度安全。下一步另冻精度约束搜索协议，尚未准入测试。\n\n"
    card += '[七问](archive/research_records/2026-09-11_automatic_workflow_engineering/IDEA.md)、[完整结果](archive/research_records/2026-09-11_automatic_workflow_engineering/RESULT.md)、[机器结果](archive/research_records/2026-09-11_automatic_workflow_engineering/RESULT.json)、[验证](archive/research_records/2026-09-11_automatic_workflow_engineering/VERIFICATION.json)。\n'
    assert (RECORD/'CARD.md').read_bytes() == old
    (RECORD/'CARD.md').write_text(card)
    publish(str(RECORD.relative_to(p.ROOT)))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
