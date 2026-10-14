"""Archive all six fixed test outputs and update both canonical rankings safely."""
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
from scripts import structured_background_protocol as p
from scripts.archive_resumed_test270_panel import atomic_text,parse_existing_table,render_leaderboard
from scripts.verify_aflow_brats_leaderboard import validate_leaderboard
from scripts.publish_research_card import publish

RECORD=p.ROOT/'archive/research_records/2026-09-11_automatic_workflow_confirmation'


def main():
    if (RECORD/'RESULT.json').exists():raise FileExistsError('confirmation already finalized; inspect staged receipts instead of rerunning')
    raw=json.loads((RECORD/'RAW_TEST270.json').read_text())
    verified=json.loads((RECORD/'VERIFICATION.json').read_text())
    pre=json.loads((RECORD/'TEST_PREACCESS.json').read_text())
    assert verified['status']=='passed' and verified['raw_sha256']==p.sha(RECORD/'RAW_TEST270.json')
    assert verified['preaccess_sha256']==p.sha(RECORD/'TEST_PREACCESS.json')
    rows=[]
    for ordinal,model in enumerate(pre['models']):
        seed=model['seed'];p.checked(model['checkpoint'],model['sha256'])
        for kind in ('identity','pruned','full8'):
            name=f'seed{seed}_{kind}';s=raw['results'][name]
            views=[0] if kind=='identity' else list(range(8)) if kind=='full8' else model['medoid']['views']
            rows.append({'method':f'fit684_native168_workflow_{seed}_{kind}',
                'display_name':f'Fit684 native168 seed{seed} {kind}',
                'scan_dice':s['scan_dice'],'subject_dice':s['subject_dice'],
                'point_hit':s['point_hits']/270,'point_hits':s['point_hits'],'point_denominator':270,
                'locator_forwards':len(views),'cohort':{'scans':270,'subjects':119},
                'checkpoint_path':str(Path(model['checkpoint']).relative_to(p.ROOT)),
                'checkpoint_sha256':model['sha256'],'resolution':128,'threshold':0.0,
                'tta':len(views)>1,'tta_formula':f'canonical views {views}; inverse-reflected ordered float32 equal raw-logit mean',
                'sam':False,'postprocessing':False,'views':views,
                'scan_dice_95ci':s['scan_cluster_95ci'],'subject_dice_95ci':s['subject_95ci'],
                'absolute_ci_seed':pre['bootstrap_seeds'][ordinal],
                'evidence_status':'user_authorized_workflow_closure; reused_internal_test; nonblind; not_external; development_derived_medoid; no_test_retuning',
                'report_path':str((RECORD/'RAW_TEST270.json').relative_to(p.ROOT)),
                'report_sha256':p.sha(RECORD/'RAW_TEST270.json'),
                'comparability_note':'same historical 128 grid/rasterization; GPU float32 Dice; fit684 not full972; T1ce WT1,2,3 excludes4',
                'status':'fixed panel control' if kind!='pruned' else 'fixed pruned candidate; gates='+str(raw['gates'][str(seed)]),
                'test_tuned':False,'external':False})
    jp=p.ROOT/'AFLOW_BRATS_LEADERBOARD.json';mp=p.ROOT/'AFLOW_BRATS_LEADERBOARD.md'
    oldj=jp.read_text();oldm=mp.read_text();board=json.loads(oldj)
    oldrows=copy.deepcopy(board['test270_direct_locator_rank_by_scan'])
    parse_existing_table(oldm,oldrows)
    ids={r['method'] for r in rows};assert len(ids)==6 and not ids&{r['method'] for r in oldrows}
    ranked=board['test270_direct_locator_rank_by_scan']
    ranked.extend(copy.deepcopy(rows));ranked.sort(key=lambda r:r['scan_dice'],reverse=True)
    for rank,row in enumerate(ranked,1):row['rank']=rank
    board['updated_utc']='2026-09-11'
    board['latest_workflow_confirmation_20260911']={'report':str((RECORD/'RAW_TEST270.json').relative_to(p.ROOT)),
        'report_sha256':p.sha(RECORD/'RAW_TEST270.json'),'all_six_rows_recorded':True,
        'reused_nonblind':True,'test_used_for_selection':False,'both_models_gate_passed':raw['both_models_passed']}
    best=ranked[0];single=next(r for r in ranked if r.get('locator_forwards')==1)
    result={'status':'completed_fixed_internal_confirmation','created_utc':datetime.now(timezone.utc).isoformat(),
        'both_models_passed':raw['both_models_passed'],'metrics':{k:{f:v for f,v in s.items() if f!='per_case'} for k,s in raw['results'].items()},
        'comparisons':raw['comparisons'],'gates':raw['gates'],'models':pre['models'],'leaderboard_rows':rows,
        'current_overall_scan_leader':best,'current_one_forward_scan_leader':single,
        'new_panel_has_overall_scan_lead':best['method'] in ids,'new_panel_has_one_forward_scan_lead':single['method'] in ids,
        'raw_sha256':p.sha(RECORD/'RAW_TEST270.json'),'preaccess_sha256':p.sha(RECORD/'TEST_PREACCESS.json'),
        'verification_sha256':p.sha(RECORD/'VERIFICATION.json'),'mask_index_sha256':p.sha(RECORD/'MASK_INDEX.json'),
        'cost':{'training':0,'test_case_forwards':raw['case_forwards'],'test_seconds':raw['seconds'],'peak_bytes':raw['peak_bytes'],
                'logit_bytes':sum(info['bytes'] for info in raw['logits'].values()),
                'mask_bytes':sum(info['bytes'] for info in json.loads((RECORD/'MASK_INDEX.json').read_text())['files'].values())},
        'access':{'test270_reused':True,'protected_outer':False,'confirmation108':False,'fixed_validation108':False,'external':False},
        'closed_no_retuning':True,'interpretation':'fixed internal efficiency confirmation, not external blind or new search-algorithm superiority'}
    p.write_new(RECORD/'RESULT.json',result)
    lines=['# 固定自动工作流：一次性内部test270确认','',
           f"状态：六输出完成并独立核验；双模型固定精度—点—成本门通过={raw['both_models_passed']}。本面板关闭，不按结果重选视图或模型。",'',
           '## 动机、方法与控制','',
           '三折选择不等于最终部署策略。本阶段只根据开发折winner视图集合取中位代表，并列按成本/ID；规则在开发后、测试前冻结，不使用测试分数。Seed11固定[0,5]两视图；seed12固定[0,5,6]三视图。两个native fit684/313模型权重、epoch168、128网格、bf16/float32、有序均值、阈值0、最高点不变，无新训练/教师/SAM/后处理。', '',
           '同一270 scans/119 subjects，原始内部测试已反复揭示，不是新盲测、外部确认或公开SOTA。每模型identity/pruned/full8三行全部保留，不能挑测试赢家当最终模型。','',
           '| 输出 | Scan Dice | Subject Dice | Point /270 | 部署前向 |','|---|---:|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['display_name']} | {row['scan_dice']:.9f} | {row['subject_dice']:.9f} | {row['point_hits']} | {row['locator_forwards']} |")
    lines.extend(['','## 固定比较及不确定性',''])
    for seed,cs in raw['comparisons'].items():
        for kind,c in cs.items():lines.append(f"- {seed} {kind}：Scan差{c['scan_delta']:+.9f}，患者簇95%CI {c['scan_cluster_95ci']}；Subject差{c['subject_delta']:+.9f}，CI {c['subject_95ci']}；Point差{c['point_delta']:+d}。")
        lines.append(f"- {seed}门：{raw['gates'][seed]}。")
    lines.extend(['','区间按10,000次患者簇百分位bootstrap，固定seed2026091161/62；未作多重比较校正。通过工程容差不等于正式非劣，也不能将负区间说成完全不降。历史GPU float32 Dice保留，exact_count_dice独立核验误差≤1e−7。','',
        '## 直觉是否得到支持','',
        ('本次两模型在固定内部报告队列都通过预定精度/点/成本门，支持该受限剪枝流程从开发迁移到固定报告队列。仍不支持普适外部泛化或树搜索优越性。' if raw['both_models_passed'] else '至少一个模型未通过固定内部确认门，不能把开发成功升级成稳定确认；具体失败门保留，不改阈值或择seed补救。工程能省时的证据仍成立，但精度/点代价未稳定满足原要求。'),'',
        f"当前同test270总体Scan最高仍按全榜为{best['scan_dice']:.9f}（{best['method']}），单次前向Scan最高为{single['scan_dice']:.9f}（{single['method']}）。本面板总体/单次是否刷新：{result['new_panel_has_overall_scan_lead']}/{result['new_panel_has_one_forward_scan_lead']}；排名点估计不等于显著性。",'',
        '## 成本与复核','',
        f"零训练、新权重。实际测试{raw['case_forwards']}病例前向，包括额外在线pruned复核；{raw['seconds']:.1f}秒含结果归档，peak{raw['peak_bytes']/1024**3:.3f}GiB。540例在线pruned与原子融合逐体素exact。全部1620个保存输出的logit/掩膜/argmax/计数及区间独立核验。",'',
        '部署每模型只需2/3次前向；既有fit阶段实测约3.96/2.65倍加速，不包含预处理或文件读取，不能称端到端加速。测试同时计算三个对照的开销不等于部署成本。新增logits12.656GiB、packed masks405MiB位于/tmp，保留可复核证据，不静默删除。','',
        '旧outer、confirmation108、fixed validation108和外部队列无新访问。共同训练源Kaggle/官方字节parity未证明限制保留。','',
        '## 证据与后续边界','',
        '[七问](IDEA.md)、[最终策略冻结](FINAL_WORKFLOWS.json)、[测试前收据](TEST_PREACCESS.json)、[原始六输出](RAW_TEST270.json)、[独立验证](VERIFICATION.json)、[机器结果](RESULT.json)、[输出掩膜索引](MASK_INDEX.json)。', '',
        '[开发与在线主实验](../2026-09-11_automatic_workflow_search/RESULT.md)、[20种子敏感度](../2026-09-11_workflow_search_seed_sensitivity/RESULT.md)、[当前交接](../../../AFLOW_BRATS_HANDOFF_CURRENT.md)、[方法目录](../../../AFLOW_BRATS_METHOD_CATALOG.md)、[排行榜](../../../AFLOW_BRATS_LEADERBOARD.md)。','',
        '本固定内部确认阶段关闭，不据此继续视图/点/阈值搜索。下一步优先整理可答辩章节、因果边界和端到端工程成本；任何新外部确认另立协议，不能冒称当前全论文三项已全部完成。'])
    for m in pre['models']:lines.append(f"\n- [Seed{m['seed']} checkpoint](../../../{Path(m['checkpoint']).relative_to(p.ROOT)}) SHA `{m['sha256']}`。")
    with (RECORD/'RESULT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    newm=render_leaderboard(oldm,oldrows,ranked,ids,str((RECORD/'RESULT.md').relative_to(p.ROOT)))
    newm=newm.replace('更新时间：2026-09-10 UTC','更新时间：2026-09-11 UTC',1).replace('2026-09-10 当前同test270/119','2026-09-11 当前同test270/119',1).replace('本次所有四输出','本次所有六输出',1)
    newj=json.dumps(board,ensure_ascii=False,indent=2)+'\n'
    for name,text in [('LEADERBOARD_BEFORE.json',oldj),('LEADERBOARD_BEFORE.md',oldm),('LEADERBOARD_STAGED.json',newj),('LEADERBOARD_STAGED.md',newm)]:
        with (RECORD/name).open('x') as f:f.write(text)
    validate_leaderboard(RECORD/'LEADERBOARD_STAGED.json',RECORD/'LEADERBOARD_STAGED.md')
    parse_existing_table(newm,ranked)
    assert jp.read_text()==oldj and mp.read_text()==oldm,'concurrent leaderboard change'
    atomic_text(jp,newj);atomic_text(mp,newm)
    validate_leaderboard(jp,mp)
    p.write_new(RECORD/'LEADERBOARD_UPDATE.json',{'status':'both_updated_and_validated','prior_rows_preserved':len(oldrows),
        'new_methods':sorted(ids),'json_sha256':p.sha(jp),'markdown_sha256':p.sha(mp),
        'result_sha256':p.sha(RECORD/'RESULT.json'),'report_sha256':p.sha(RECORD/'RAW_TEST270.json')})
    before=(RECORD/'CARD.md').read_bytes()
    with (RECORD/'CARD.md.BEFORE_RESULT').open('xb') as f:f.write(before)
    card='## 自动工作流固定内部确认完成：双模型联合门通过='+str(raw['both_models_passed'])+'\n\n'
    card+='动机：将六折策略收束成两个固定部署workflow。仅按开发视图集合取中位代表：seed11[0,5]两视图，seed12[0,5,6]三视图；权重/输入/融合/阈值/点固定，无新训练。复用非盲test270/119六输出全部已入MD/JSON榜，不择test赢家改方案。\n\n'
    for m in pre['models']:
        seed=m['seed'];s=raw['results'][f'seed{seed}_pruned'];d=raw['comparisons'][str(seed)]['pruned_vs_full8']
        card+=f"Seed {seed} pruned：Scan/Subject={s['scan_dice']:.9f}/{s['subject_dice']:.9f}、Point{s['point_hits']}/270；对full8 Scan差{d['scan_delta']:+.6f}、Subject差{d['subject_delta']:+.6f}、Point{d['point_delta']:+d}。门{raw['gates'][str(seed)]}。\n\n"
    card+='直觉判断依全部原门：'+('支持固定内部精度—成本权衡，不是无损/正式非劣/外部泛化。' if raw['both_models_passed'] else '开发通过未稳定转为固定内部确认，不能择seed或改门救结果；工程加速不代表精度/点安全。')+'\n\n'
    card+=f"本次{raw['case_forwards']}前向、{raw['seconds']:.1f}秒含归档，0训练；/tmp新增logits12.656GiB+mask405MiB保留。540例在线parity和1620个保存输出独立核验；部署2/3×，既有fit阶段加速3.96/2.65倍，不含预处理。已新访问reused test，旧outer/confirmation/固定validation/外部未访问。本面板关闭，不测试回流调参。全榜总体/单次Scan最高{best['scan_dice']:.9f}/{single['scan_dice']:.9f}。\n\n"
    card+='[七问](archive/research_records/2026-09-11_automatic_workflow_confirmation/IDEA.md)、[完整结果与CI](archive/research_records/2026-09-11_automatic_workflow_confirmation/RESULT.md)、[原始报告](archive/research_records/2026-09-11_automatic_workflow_confirmation/RAW_TEST270.json)、[独立核验](archive/research_records/2026-09-11_automatic_workflow_confirmation/VERIFICATION.json)、[榜单更新](archive/research_records/2026-09-11_automatic_workflow_confirmation/LEADERBOARD_UPDATE.json)。\n'
    assert (RECORD/'CARD.md').read_bytes()==before
    (RECORD/'CARD.md').write_text(card);publish(str(RECORD.relative_to(p.ROOT)))
    print(json.dumps({'both_models_passed':raw['both_models_passed'],'gates':raw['gates'],
                      'six_metrics':result['metrics'],'overall_scan':best['scan_dice'],'single_scan':single['scan_dice']},indent=2))


if __name__=='__main__':main()
