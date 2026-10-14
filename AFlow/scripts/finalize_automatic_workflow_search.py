"""Archive the frozen development/online result, retaining every controller."""
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
from sammed3d.automatic_workflow_search import scoped_rows,search_controller,select_workflow
from scripts import structured_background_protocol as p
from scripts.publish_research_card import publish

RECORD=p.ROOT/'archive/research_records/2026-09-11_automatic_workflow_search'


def main():
    if (RECORD/'RESULT.json').exists():raise FileExistsError('result already finalized')
    analysis=json.loads((RECORD/'ANALYSIS.json').read_text())
    raw=json.loads((RECORD/'RAW_DEVELOPMENT.json').read_text())
    score_verify=json.loads((RECORD/'SCORE_VERIFICATION.json').read_text())
    online=json.loads((RECORD/'ONLINE_BENCHMARK.json').read_text())
    opre=json.loads((RECORD/'ONLINE_PREACCESS.json').read_text())
    pre=json.loads((RECORD/'DEV_PREACCESS.json').read_text())
    assert score_verify['status']=='passed' and score_verify['candidate_case_scores_verified']==46080
    assert p.sha(RECORD/'RAW_DEVELOPMENT.json')==analysis['raw_sha256']==score_verify['raw_sha256']
    assert p.sha(RECORD/'ANALYSIS.json')==opre['analysis_sha256']
    assert p.sha(RECORD/'ONLINE_PREACCESS.json')==online['online_preaccess_sha256']
    for name,digest in pre['files'].items():p.checked(p.ROOT/name,digest)
    for report in pre['models']:p.checked(report['checkpoint'],report['sha256'])
    selection_checks=[]
    for ordinal,seed in enumerate(p.SEEDS):
        info=raw['results'][str(seed)];p.checked(info['path'],info['sha256'])
        table={int(k):v for k,v in json.loads(Path(info['path']).read_text())['rows'].items()}
        for receipt in analysis['selection_receipts']:
            if receipt['seed']!=seed:continue
            fold=receipt['fold'];name=receipt['controller']
            a=set(receipt['search_subjects']);b=set(receipt['selector_subjects']);c=set(receipt['assessment_subjects'])
            assert len(a)==len(b)==len(c)==26 and not(a&b or b&c or a&c)
            replay=search_controller(name,scoped_rows(table,a),2026091132+ordinal*100+fold)
            # JSON normalizes integer keys to strings; compare the serialized representation.
            assert json.loads(json.dumps(replay))==receipt['search']
            pool=replay['selector_pool']
            choice=select_workflow(pool,scoped_rows({k:table[k] for k in pool},b))
            assert json.loads(json.dumps(choice))==receipt['selection']
            saved=json.loads((RECORD/f'SELECTION_{seed}_{fold}_{name}.json').read_text())
            assert saved==receipt
            selection_checks.append({'seed':seed,'fold':fold,'controller':name,'exact_replay':True})
        for name,policy in analysis['results'][str(seed)]['policies'].items():
            case_map={}
            for r in analysis['selection_receipts']:
                if r['seed']==seed and r['controller']==name:
                    for row in table[r['selection']['selected']]:
                        if row['subject_id'] in r['assessment_subjects']:
                            assert row['case_id'] not in case_map
                            case_map[row['case_id']]=row
            assert [case_map[k] for k in pre['case_ids']]==policy['summary']['per_case']
    assert len(selection_checks)==24
    assert len(online['rows'])==384 and len(online['parity'])==16
    assert all(r['max_abs_error']==0 for r in online['parity'])
    assert online['case_forwards']==opre['expected_case_forwards']
    assert not online['labels_opened'] and not online['test_access']
    for seed in opre['policies']:
        timings={(r['case_id'],r['repeat'],r['workflow']):r['seconds'] for r in online['rows'] if r['seed']==seed}
        assert len(timings)==192
        ratios=[]
        for case in opre['fit_cases']:
            for repeat in range(2):
                weighted=sum(w*timings[(case,repeat,name)] for name,w in opre['assessment_scan_weights'][seed].items())
                ratios.append(timings[(case,repeat,'full8')]/weighted)
        assert statistics.median(ratios)==online['summary'][seed]['paired_median_speedup']
        assert (statistics.median(ratios)>=1.5)==online['summary'][seed]['speedup_at_least1_5']
    passed=analysis['primary_passed'] and online['all_gates_passed']
    evidence=['IDEA.md','METADATA_CORRECTION.md','SMOKE_PREACCESS.json','SMOKE.json','DEV_PREACCESS.json',
              'RAW_DEVELOPMENT.json','ANALYSIS_PREACCESS.json','ANALYSIS.json','SCORE_VERIFICATION.json',
              'ONLINE_PREACCESS.json','ONLINE_BENCHMARK.json']
    evidence.extend(f'SELECTION_{r["seed"]}_{r["fold"]}_{r["controller"]}.json' for r in analysis['selection_receipts'])
    hashes={name:p.sha(RECORD/name) for name in evidence}
    body={'status':'completed','created_utc':datetime.now(timezone.utc).isoformat(),
          'primary_accuracy_cost_gates_passed':analysis['primary_passed'],
          'online_gates_passed':online['all_gates_passed'],'development_engineering_loop_passed':passed,
          'results':analysis['results'],'selection_receipts':analysis['selection_receipts'],
          'online_summary':online['summary'],'evidence_sha256':hashes,'checkpoints':pre['models'],
          'atom_caches':raw['cache'],'cost':{'training_runs':0,'smoke_case_forwards':496,
              'development_case_forwards':2880,'online_fit_case_forwards':online['case_forwards'],
              'development_collection_seconds':raw['seconds'],'online_seconds':online['seconds'],
              'development_peak_bytes':raw['peak_bytes'],'atom_bytes':sum(c['bytes'] for c in raw['cache'].values())},
          'access':{'reused_development180_78':True,'fit_images_and_smoke_labels':True,
                    'test':False,'protected_outer':False,'confirmation':False,'fixed_validation':False,'external':False},
          'decision':'bounded automatic-workflow development and engineering evidence complete; no final deployment workflow or test confirmation admitted',
          'limitations':['conditional OOF on repeatedly reused patients','no formal noninferiority claim',
              'six fold-specific workflows, not a case-adaptive router','exhaustive search uses shortlist, not globally optimal selector performance',
              'no demonstrated UCT advantage','128 candidate evaluation cache was fully generated; no search-stage GPU saving claimed']}
    p.write_new(RECORD/'RESULT.json',body)
    lines=['# 自动工作流搜索与剪枝：开发—执行闭环结果','',
        f'状态：本次固定开发/工程协议完成，全部主门通过={passed}。不是最终部署或新盲测确认。','',
        '## 一句话结论与动机','',
        '针对当前八镜像重复推理的成本，在固定自动模型上搜索含原图的镜像子集。结果支持用可控的小幅Dice损失换取显著推理节省；不支持“精度完全不降”，也不支持树搜索优于简单方法。','',
        '只改变镜像子集/预定控制器；两个native fit684/313固定168轮模型、权重、128栅格、bf16前向、float32有序均值、阈值0与argmax点保持不变。无训练、教师、SAM或后处理。','',
        '## 主结果：同一复用开发180扫描/78患者','',
        '| 模型seed | 策略 | Scan Dice | Subject Dice | Point /180 | 平均前向 |',
        '|---|---|---:|---:|---:|---:|']
    for seed,record in analysis['results'].items():
        for title,summary,cost in [('identity',record['fixed']['0'],1),('full8',record['fixed']['127'],8),
             ('主方法OOF',record['policies']['exhaustive']['summary'],record['policies']['exhaustive']['mean_forwards'])]:
            lines.append(f"| {seed} | {title} | {summary['scan_dice']:.9f} | {summary['subject_dice']:.9f} | {summary['point_hits']} | {cost:.3f} |")
    lines.extend(['','## 不能隐藏的精度代价',''])
    for seed,record in analysis['results'].items():
        contrast=record['policies']['exhaustive']['versus_full8'];identity=record['policies']['exhaustive']['versus_identity']
        lines.append(f"- Seed {seed}相对full8：Scan差{contrast['scan_delta']:+.9f}，患者簇95%CI {contrast['scan_cluster_95ci']}；Subject差{contrast['subject_delta']:+.9f}，CI {contrast['subject_95ci']}；Point差{contrast['point_delta']:+d}。相对identity：Scan/Subject差{identity['scan_delta']:+.9f}/{identity['subject_delta']:+.9f}，Point{identity['point_delta']:+d}。")
    lines.extend(['','部分相对full8区间整体为负：这是小幅精度损失，不是无损压缩。通过的是预注册工程容差门，不是正式统计非劣。相对identity的两seed Dice区间为正，但这里是条件于已选策略的反复开发证据；两训练seed不是独立队列，CI未覆盖完整选择不确定性或多重探索。','',
        '## 搜索对照与机制','',
        '| Seed | 控制器 | Scan | Subject | Point | 平均前向 | 开发联合门 |','|---|---|---:|---:|---:|---:|---|'])
    for seed,record in analysis['results'].items():
        for name,policy in record['policies'].items():
            s=policy['summary'];lines.append(f"| {seed} | {name} | {s['scan_dice']:.6f} | {s['subject_dice']:.6f} | {s['point_hits']} | {policy['mean_forwards']:.3f} | {policy['passed']} |")
    lines.extend(['','穷举为预定主方法；每折全128次查询，其他控制器各16次唯一查询、同8个selector名额且强制两个参照。UCT实际14–15次树模拟就耗尽16唯一查询（含两个参照），不能声称512次模拟或充分证明树搜索无效。随机seed11与UCT seed12分别失败于Point门；贪心两seed通过。这里没有证明复杂搜索器更好。','',
        '128候选仅穷举search reward，之后仍按冻结短名单进入selector；不称“全局最优精度—成本解”。所有控制器共享完整原子评价缓存，实际并没有在本次搜索阶段少跑GPU。节省发生在选定工作流的真实部署执行。','',
        '## 在线复核与成本',''])
    for seed,s in online['summary'].items():
        lines.append(f"- Seed {seed}：按assessment扫描数加权的折策略平均{ s['assessment_scan_weighted_selected_mean_seconds']:.6f}秒/例；相对full8配对中位加速{s['paired_median_speedup']:.3f}倍。")
    lines.extend(['',f"硬件{online['hardware']}；16项fit病例在线/原子融合逐体素完全一致。峰值开发显存{raw['peak_bytes']/1024**3:.3f}GiB。正式开发{raw['seconds']:.1f}秒、2880模型前向；fit smoke496前向；在线测速{online['seconds']:.1f}秒、{online['case_forwards']}前向。原子缓存22.5GiB在/tmp保留，0新训练/权重。",'',
        '时延包含图像tensor H2D、镜像模型/融合、掩膜和点D2H，不含NIfTI预处理、文件读取、权重加载。六个折策略按开发组病例数加权，只是部署成本估计，不代表已实现按病例自适应路由；没有选定一个可立即上线的最终工作流。','',
        '## 科研闭环完成范围与下一步','',
        '本次已完成：预注册→患者分组→合成与GPU smoke→全部候选开发评分→控制器/selector→折外分析→独立重算全部46080条评分→实际按需在线复核与测速→归档。','',
        '尚未完成：单一最终工作流的冻结与独立确认、端到端预处理耗时、搜索控制器跨随机种子的稳定性。这些不能被本次通过替代。下一安全工作优先补搜索随机性敏感度与受限机制解释；不重启模型训练或偷偷打开test。若以后进入test，必须另有明确冻结和适用的访问准入，且保持reused/internal标签。','',
        '同一历史reused internal test270/119：总体Scan0.757269662、Subject0.741253585、Point265/270（8×、一个checkpoint）；单次前向Scan0.752003176、Subject0.737266420、Point252/270。总体Scan/Subject绝对95%CI为[0.739089,0.774152]/[0.714109,0.766152]。本轮未改变榜，也不能拿开发180数值相减宣称超榜。','',
        '## 访问与证据','',
        '新访问仅授权复用开发180/78与fit smoke；0 test/旧outer/confirmation/fixed validation/外部访问。原子与checkpoint SHA已核查。测试源官方字节parity等历史限制不因本实验消失。','',
        '- [七问](IDEA.md)、[预访问元数据更正](METADATA_CORRECTION.md)、[开发前收据](DEV_PREACCESS.json)、[原始索引](RAW_DEVELOPMENT.json)、[分析](ANALYSIS.json)',
        '- [全部评分独立验证](SCORE_VERIFICATION.json)、[在线测速](ONLINE_BENCHMARK.json)、[结果JSON](RESULT.json)、[归档验证](VERIFICATION.json)',
        '- [当前交接](../../../AFLOW_BRATS_HANDOFF_CURRENT.md)、[方法目录](../../../AFLOW_BRATS_METHOD_CATALOG.md)、[排行榜](../../../AFLOW_BRATS_LEADERBOARD.md)'])
    for report in pre['models']:
        path=Path(report['checkpoint']).relative_to(p.ROOT)
        lines.append(f"- [固定seed{report['seed']}权重](../../../{path})，SHA `{report['sha256']}`")
    with (RECORD/'RESULT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    p.write_new(RECORD/'VERIFICATION.json',{'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),
        'result_sha256':p.sha(RECORD/'RESULT.json'),'markdown_sha256':p.sha(RECORD/'RESULT.md'),
        'evidence_sha256':hashes,'selection_exact_replay':selection_checks,'online_rows_recounted':384,
        'all_46080_scores_independently_verified':True,'verifier_sha256':p.sha(Path(__file__)),
        'limitation':'selector replay shares frozen controller code; raw score recount uses independent NumPy implementation'})
    before=(RECORD/'CARD.md').read_bytes()
    with (RECORD/'CARD.md.BEFORE_RESULT').open('xb') as f:f.write(before)
    card='## 自动工作流搜索与剪枝：开发—执行闭环完成，主门通过\n\n'
    card+='一句话：以预先允许的小幅Dice损失减少重复镜像。只改视图子集；模型、权重、128网格、融合、阈值0和argmax固定。两个native模型、复用dev180/78患者三组轮换搜索/选择/评估，穷举主参照及随机/贪心/UCT等预算对照。\n\n'
    for seed,r in analysis['results'].items():
        v=r['policies']['exhaustive'];s=v['summary'];d=v['versus_full8']
        card+=f"Seed {seed}：OOF Scan/Subject={s['scan_dice']:.9f}/{s['subject_dice']:.9f}，Point{s['point_hits']}/180，平均{v['mean_forwards']:.1f}×。相对full8 Scan差{d['scan_delta']:+.6f}、Subject差{d['subject_delta']:+.6f}、Point{d['point_delta']:+d}，实测阶段加速{online['summary'][seed]['paired_median_speedup']:.3f}倍。\n\n"
    card+='支持受限精度—成本权衡，不是无损或显著不降；部分full8差值CI整体为负。随机seed11和UCT seed12点门失败，贪心两seed通过，不证明树搜索优越。原定主方法两seed全部精度/点/成本门与在线门通过，不换主方法救结果。\n\n'
    card+='完整46080候选病例评分由独立NumPy重算exact，16项所选在线/原子parity exact；零训练，开发2880前向、fit smoke496、在线'+str(online['case_forwards'])+'前向，/tmp原子22.5GiB保留。无test/保护outer/confirmation/固定validation/外部访问；没有单一最终部署工作流或新盲测确认。下一步补搜索随机性敏感度，不恢复相似增强训练。\n\n'
    card+='[七问](archive/research_records/2026-09-11_automatic_workflow_search/IDEA.md)、[完整结果与CI](archive/research_records/2026-09-11_automatic_workflow_search/RESULT.md)、[机器结果](archive/research_records/2026-09-11_automatic_workflow_search/RESULT.json)、[独立验证](archive/research_records/2026-09-11_automatic_workflow_search/VERIFICATION.json)。\n'
    assert (RECORD/'CARD.md').read_bytes()==before
    (RECORD/'CARD.md').write_text(card);publish(str(RECORD.relative_to(p.ROOT)))
    print(json.dumps({'development_engineering_loop_passed':passed,'online':online['summary'],
                      'result_sha256':p.sha(RECORD/'RESULT.json')},indent=2))


if __name__=='__main__':main()
