"""Fixed twenty-seed sensitivity; no model, image, label or atom-cache access."""
import json
from datetime import datetime,timezone
from pathlib import Path
import statistics
import time
from sammed3d.automatic_workflow_search import scoped_rows,search_controller,select_workflow,reward
from scripts.evaluate_structured_background_aux import summarize,compare
from scripts import structured_background_protocol as p
from scripts.publish_research_card import publish

RECORD=p.ROOT/'archive/research_records/2026-09-11_workflow_search_seed_sensitivity'
SOURCE=p.ROOT/'archive/research_records/2026-09-11_automatic_workflow_search'


def main():
    if (RECORD/'PREACCESS.json').exists():raise FileExistsError('fixed grid already started')
    inputs=['RESULT.json','ANALYSIS.json','RAW_DEVELOPMENT.json','DEV_PREACCESS.json',
            'RAW_SEED2026090711.json','RAW_SEED2026090712.json','VERIFICATION.json']
    hashes={name:p.sha(SOURCE/name) for name in inputs}
    p.write_new(RECORD/'PREACCESS.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':hashes,'idea_sha256':p.sha(RECORD/'IDEA.md'),
        'script_sha256':p.sha(Path(__file__)),'grid':list(range(20)),
        'seed_formula':'2026091150 +1000*r +100*model_ordinal +fold',
        'source_performance_previously_seen':True,'new_forwards':0,'test_access':False})
    main_result=json.loads((SOURCE/'RESULT.json').read_text())
    source_verify=json.loads((SOURCE/'VERIFICATION.json').read_text())
    assert source_verify['result_sha256']==hashes['RESULT.json'] and source_verify['status']=='passed'
    pre=json.loads((SOURCE/'DEV_PREACCESS.json').read_text())
    for name,digest in pre['files'].items():p.checked(p.ROOT/name,digest)
    groups=pre['patient_groups'];order=pre['case_ids'];all_runs=[];started=time.monotonic()
    for ordinal,seed in enumerate(p.SEEDS):
        data=json.loads((SOURCE/f'RAW_SEED{seed}.json').read_text())
        table={int(k):v for k,v in data['rows'].items()}
        fixed=main_result['results'][str(seed)]['fixed']
        search_scopes=[scoped_rows(table,groups[(f+1)%3]) for f in range(3)]
        maxima=[max(reward(v,k) for k,v in scope.items()) for scope in search_scopes]
        for r in range(20):
            for controller in ('random','uct'):
                rows=[];costs=[];receipts=[]
                for fold in range(3):
                    search=search_controller(controller,search_scopes[fold],2026091150+1000*r+100*ordinal+fold)
                    pool=search['selector_pool']
                    selection=select_workflow(pool,scoped_rows({k:table[k] for k in pool},groups[(fold+2)%3]))
                    receipt={'fold':fold,'search':search,'selection':selection,
                             'reward_regret':maxima[fold]-max(search['search_rewards'].values())}
                    receipts.append(receipt)
                    # Selection is complete before assessment row extraction.
                    selected=scoped_rows({selection['selected']:table[selection['selected']]},groups[fold])[selection['selected']]
                    rows.extend(selected);costs.extend([len(selection['views'])]*len(selected))
                by_id={row['case_id']:row for row in rows};assert len(by_id)==180
                summary=summarize([by_id[k] for k in order],2026091141+ordinal)
                delta=compare(fixed['127'],summary,2026091141+ordinal)
                gates={'cost':statistics.fmean(costs)<=4,'scan':delta['scan_delta']>=-.003,
                       'subject':delta['subject_delta']>=-.003,'scan_ci':delta['scan_cluster_95ci'][0]>=-.005,
                       'subject_ci':delta['subject_95ci'][0]>=-.005,
                       'point':summary['point_hits']>=max(fixed['0']['point_hits'],fixed['127']['point_hits'])-1}
                run={'model_seed':seed,'grid_index':r,'controller':controller,'summary':summary,
                     'versus_full8':delta,'mean_forwards':statistics.fmean(costs),'gates':gates,
                     'passed':all(gates.values()),'folds':receipts}
                all_runs.append(run)
            print(json.dumps({'model_seed':seed,'grid_completed':r+1,'seconds':time.monotonic()-started}),flush=True)
    assert len(all_runs)==80
    p.write_new(RECORD/'RAW_REPLAYS.json',{'runs':all_runs,'source_sha256':hashes,'preaccess_sha256':p.sha(RECORD/'PREACCESS.json')})
    def span(values):return {'min':min(values),'median':statistics.median(values),'max':max(values)}
    aggregate={}
    for controller in ('random','uct'):
        model_summary={}
        for seed in p.SEEDS:
            runs=[v for v in all_runs if v['controller']==controller and v['model_seed']==seed]
            assert len(runs)==20
            model_summary[str(seed)]={'passed':sum(v['passed'] for v in runs),'denominator':20,
                'scan_delta':span([v['versus_full8']['scan_delta'] for v in runs]),
                'subject_delta':span([v['versus_full8']['subject_delta'] for v in runs]),
                'point_hits':span([v['summary']['point_hits'] for v in runs]),
                'mean_forwards':span([v['mean_forwards'] for v in runs]),
                'reward_regret':span([f['reward_regret'] for v in runs for f in v['folds']])}
        joint=sum(all(v['passed'] for v in all_runs if v['controller']==controller and v['grid_index']==r) for r in range(20))
        aggregate[controller]={'models':model_summary,'both_models_passed':joint,'denominator':20}
    # Independent accounting of success indicators and finite-grid extrema.
    reload=json.loads((RECORD/'RAW_REPLAYS.json').read_text())
    for run in reload['runs']:
        assert run['passed']==all(run['gates'].values())
        assert len(run['summary']['per_case'])==180
        assert all(f['search']['unique_queries']==16 for f in run['folds'])
    for name,digest in hashes.items():p.checked(SOURCE/name,digest)
    result={'status':'completed','created_utc':datetime.now(timezone.utc).isoformat(),
        'aggregate':aggregate,'runs':80,'selection_traces':240,'new_model_forwards':0,
        'seconds':time.monotonic()-started,'raw_sha256':p.sha(RECORD/'RAW_REPLAYS.json'),
        'preaccess_sha256':p.sha(RECORD/'PREACCESS.json'),
        'scope':'post-primary-result sensitivity on same 180 scans/78 patients; no seed selection or new validation',
        'main_result_unchanged_sha256':hashes['RESULT.json'],'test_access':False}
    p.write_new(RECORD/'RESULT.json',result)
    lines=['# 搜索随机性敏感度：20种子全部完成','',
           '这是主实验结果后的补充分析；同一批78患者，不是20次独立验证。未新增训练、模型前向、影像/标签数组读取或任何测试队列访问。','',
           '| 控制器 | 模型seed | 通过次数/20 | Point范围 | 平均前向范围 |','|---|---|---:|---|---|']
    for name,a in aggregate.items():
        for seed,m in a['models'].items():
            lines.append(f"| {name} | {seed} | {m['passed']}/20 | {m['point_hits']} | {m['mean_forwards']} |")
        lines.extend(['',f"{name}双模型同时通过：{a['both_models_passed']}/20。",''])
    lines.extend(['## 解释','',
        '通过频次和范围衡量固定数据/固定预算下的随机种子敏感度，不是临床置信概率或算法显著检验。不给最佳seed、不替换原主方法、不从本表选择最终workflow。确定性的穷举/贪心没有虚构20次独立重复。','',
        '对树搜索的合理结论仍然是：这套16唯一查询的小空间实现没有建立稳定优越性；不是证明所有MCTS无效。论文主贡献应放在精度约束的自动推理剪枝、数据角色隔离与真实按需执行上。','',
        '只改变random/UCT随机种子；模型、患者分组、候选空间、查询数、选择规则和统计门均不变。主实验原始随机结果和通过判断保持原样。','',
        f"80份OOF结果、240条折选择轨迹完整保存，CPU耗时{result['seconds']:.1f}秒；0 GPU、新checkpoint或大型缓存。",'',
        '[七问](IDEA.md)、[冻结](PREACCESS.json)、[完整重放](RAW_REPLAYS.json)、[机器结果](RESULT.json)、[核验](VERIFICATION.json)、[主实验](../2026-09-11_automatic_workflow_search/RESULT.md)。'])
    with (RECORD/'RESULT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    p.write_new(RECORD/'VERIFICATION.json',{'status':'passed','result_sha256':p.sha(RECORD/'RESULT.json'),
        'raw_sha256':p.sha(RECORD/'RAW_REPLAYS.json'),'preaccess_sha256':p.sha(RECORD/'PREACCESS.json'),
        'markdown_sha256':p.sha(RECORD/'RESULT.md'),'source_sha256_unchanged':hashes,
        'scope':'grid completeness, query budgets, success conjunction and unchanged source hashes; no claim of separate patient replication'})
    before=(RECORD/'CARD.md').read_bytes()
    with (RECORD/'CARD.md.BEFORE_RESULT').open('xb') as f:f.write(before)
    card='## 搜索随机性敏感度：20种子已完成，主实验保持不变\n\n'
    card+='动机：单个搜索种子不够判断算法。仅改变random/UCT随机种子，固定模型、128子集、患者三组、16查询/8候选与联合门；同78患者重复分析，不是新盲测。\n\n'
    for name,a in aggregate.items():
        card+=f"{name}：模型11/12各通过{a['models']['2026090711']['passed']}/20、{a['models']['2026090712']['passed']}/20，双模型同时通过{a['both_models_passed']}/20。\n\n"
    card+='完整80结果/240折轨迹均保留，不择最好seed，不改穷举主实验；频次不是显著性检验，不支持稳定树搜索优越性。0训练/GPU/模型前向、新影像标签或保护队列访问，无新大cache。下一步论文主张保持精度—成本权衡，最终workflow/确认另冻，不能冒称全论文已完成。\n\n'
    card+='[七问](archive/research_records/2026-09-11_workflow_search_seed_sensitivity/IDEA.md)、[结果](archive/research_records/2026-09-11_workflow_search_seed_sensitivity/RESULT.md)、[机器汇总](archive/research_records/2026-09-11_workflow_search_seed_sensitivity/RESULT.json)、[核验](archive/research_records/2026-09-11_workflow_search_seed_sensitivity/VERIFICATION.json)。\n'
    assert (RECORD/'CARD.md').read_bytes()==before
    (RECORD/'CARD.md').write_text(card);publish(str(RECORD.relative_to(p.ROOT)))
    print(json.dumps(aggregate,indent=2),flush=True)


if __name__=='__main__':main()
