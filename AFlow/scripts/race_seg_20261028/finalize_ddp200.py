"""Archive the user-selected single algorithm, with historical comparisons only."""
from scripts.race_seg_20261028.common import *


def main():
    import torch
    from scripts import segresnet_pretrained_native_common as native
    state=read(R/'race/STATUS.json')
    report_path=R/'race'/f"validation_e{state['best_epoch']:03}.json"
    report=read(report_path)
    saved=torch.load(OUT/'race/best.pth',map_location='cpu',weights_only=False)
    assert saved['epoch']==state['best_epoch'] and saved['arm']=='race'
    assert all(bool(v.isfinite().all()) for v in saved['model_state_dict'].values())
    baseline_path=ROOT/'archive/research_records/2026-10-27_segresnet_width_resume/RESULT.json'
    assert sha(baseline_path)==read(R/'DDP200_PLAN.json')['historical_result_sha256']
    baseline=read(baseline_path)['arms']['w32']['report']
    comparison=native.paired(baseline['per_case'],report['per_case'])
    complete=(R/'race/DONE.json').exists() and state['last_complete_epoch']==200 and state['updates']==24400
    passed=bool(complete and comparison['scan_delta']>=.005 and comparison['scan_delta_ci95'][0]>0
                and comparison['subject_delta']>=0 and comparison['point_hits_delta']>=0)
    result=dict(at=now(),state='complete' if complete else 'incomplete',states={'race':state},
          stopped_control=read(R/'control/STATUS.json'),training_recipe=read(R/'BUDGET_DECISION.json'),
          cohort='reused validation108/50',single_seed=True,test_access=False,
          comparison_race_minus_historical_w32=comparison,historical_improvement_gate=passed,
          original_matched_gate_evaluable=False,reason='User cancelled contemporaneous control',
          causal_algorithm_gain_established=False)
    write(R/'RESULT.json',result)
    write(R/'COMPLETION_AUDIT.json',dict(at=now(),state=result['state'],result_sha256=sha(R/'RESULT.json'),
          selected_models={'race':dict(epoch=state['best_epoch'],checkpoint_sha256=sha(OUT/'race/best.pth'),
               selected_report_sha256=sha(report_path),finite_cpu_reload=True)},
          historical_result_sha256=sha(baseline_path),execution_freeze_sha256=sha(R/'EXECUTION_FREEZE.json'),
          ddp_freeze_sha256=sha(R/'DDP200_FREEZE.json'),test_access=False))
    m=report['metrics'];b=baseline['metrics']
    body=f'''# RACE-Seg双卡单算法实验结果

状态：{result['state']}。算法组累计完成{state['last_complete_epoch']}轮、{state['updates']}保留更新，108扫描/50患者复用验证。用户取消同期对照，新对照仅完成4轮后停止；不将其与算法200轮作公平消融比较。

| 组 | 最佳轮 | Scan Dice | Subject Dice | Point |
|---|---:|---:|---:|---:|
| 历史32通道200轮参考 | 171 | {b['scan_dice']:.9f} | {b['subject_dice']:.9f} | {b['point_hits']}/108 |
| RACE-Seg | {state['best_epoch']} | {m['scan_dice']:.9f} | {m['subject_dice']:.9f} | {m['point_hits']}/108 |

相对历史参考Scan差{comparison['scan_delta']:+.9f}，描述性患者配对95%区间{comparison['scan_delta_ci95']}。历史提升门：{passed}。此比较包含种子、执行方式和新算法等差异，不能单独确认算法因果贡献；原同期对照门不可评估。

原972/108、单T1ce、原生1mm、有效batch8、每轮122更新、累计200轮上限及原200轮学习率不变。早期单卡完整模型/优化器接续双卡DDP；归约及dropout轨迹改变，不称逐位相同或等FLOPs。原173预算及取消的两组200迁移保留历史记录。本阶段无新270/outer/confirmation；未完成时不称完整200轮。没有自动续训、扩组或新测试。

[执行修订](RUNBOOK.md) · [机器结果](RESULT.json) · [当前状态](PIPELINE_STATUS.json)
'''
    (R/'RESULT.md').write_text(body)
    summary=f'RACE-Seg双卡单算法{result["state"]}，累计{state["last_complete_epoch"]}轮，最佳{state["best_epoch"]}轮108 Scan {m["scan_dice"]:.6f}。历史参考差{comparison["scan_delta"]:+.6f}，描述性区间{comparison["scan_delta_ci95"]}；历史提升门{passed}。同期对照已由用户取消，不能单独确认算法因果收益。无新270/outer/confirmation，本轮结束不自动恢复。[结果]({R.relative_to(ROOT)}/RESULT.md)。'
    for name in ('AFLOW_BRATS_RESULTS.md','AFLOW_BRATS_WEEKLY_REPORT.md','AFLOW_BRATS_METHOD_CATALOG.md','AFLOW_BRATS_HANDOFF_CURRENT.md'):
        path=ROOT/name;text=path.read_text();start=text.find('<!-- RACE_PREP_20261028 -->');end=text.find('<!-- END_RACE_PREP_20261028 -->')
        block=f'<!-- RACE_PREP_20261028 -->\n## 2026-10-28 RACE-Seg双卡算法终态\n\n{summary}\n<!-- END_RACE_PREP_20261028 -->'
        if start>=0 and end>=start:text=text[:start]+block+text[end+len('<!-- END_RACE_PREP_20261028 -->'):]
        else:text=text.rstrip()+'\n\n'+block+'\n'
        path.write_text(text)
    path=ROOT/'AGENTS.md';text=path.read_text();marker='<!-- RACE_DDP_FINAL_20261028 -->'
    if marker not in text:path.write_text(text.replace('# AFlow repository instructions','# AFlow repository instructions\n\n'+marker+'\n'+summary+'\n<!-- END_RACE_DDP_FINAL_20261028 -->',1))
    write(R/'PIPELINE_STATUS.json',dict(at=now(),state=result['state'],training_finished=True,
          active_arms=['race'],historical_improvement_gate=passed,original_matched_gate_evaluable=False,test_access=False))


if __name__=='__main__': main()
