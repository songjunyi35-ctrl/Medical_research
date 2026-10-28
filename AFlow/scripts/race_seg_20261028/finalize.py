"""Publish both matched outcomes, including failures; never runs a model."""
from scripts.race_seg_20261028.common import *


def main():
    import torch
    from scripts import segresnet_pretrained_native_common as native
    states={arm:read(R/arm/'STATUS.json') for arm in ARMS if (R/arm/'STATUS.json').exists()}
    selected={arm:read(R/arm/f"validation_e{s['best_epoch']:03}.json") for arm,s in states.items()}
    complete=all((R/arm/'DONE.json').exists() for arm in ARMS)
    comparison=None;passed=False
    if len(selected)==2:
        comparison=native.paired(selected['control']['per_case'],selected['race']['per_case'])
        passed=bool(complete and comparison['scan_delta']>=.005 and comparison['scan_delta_ci95'][0]>0
                    and comparison['subject_delta']>=0 and comparison['point_hits_delta']>=0)
    result={'at':now(),'state':'complete' if complete else 'incomplete','states':states,
            'comparison_race_minus_control':comparison,'clear_benefit_gate':passed,
            'cohort':'reused validation108/50','single_seed':True,'test_access':False,
            'training_recipe':read(R/'BUDGET_DECISION.json') if (R/'BUDGET_DECISION.json').exists() else None}
    write(R/'RESULT.json',result)
    audit={}
    for arm,s in states.items():
        checkpoint=OUT/arm/'best.pth'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['epoch']==s['best_epoch'] and saved['arm']==arm
        assert all(bool(v.isfinite().all()) for v in saved['model_state_dict'].values())
        report=R/arm/f"validation_e{s['best_epoch']:03}.json"
        assert len(selected[arm]['per_case'])==108 and len({x['subject_id'] for x in selected[arm]['per_case']})==50
        audit[arm]={'checkpoint_sha256':sha(checkpoint),'selected_report_sha256':sha(report),'epoch':s['best_epoch'],'finite_cpu_reload':True}
    write(R/'COMPLETION_AUDIT.json',{'at':now(),'state':result['state'],'selected_models':audit,
                                    'execution_freeze_sha256':sha(R/'EXECUTION_FREEZE.json'),
                                    'result_sha256':sha(R/'RESULT.json'),'test_access':False})
    lines=['# RACE-Seg首次匹配实验结果','',f"状态：{result['state']}。原972训练、108扫描/50患者复用验证；无新test270/outer/confirmation。",'',
           '| 组 | 完成轮 | 最佳轮 | Scan Dice | Subject Dice | Point |','|---|---:|---:|---:|---:|---:|']
    for arm,s in states.items():
        m=selected[arm]['metrics']
        lines.append(f"| {arm} | {s['last_complete_epoch']} | {s['best_epoch']} | {m['scan_dice']:.9f} | {m['subject_dice']:.9f} | {m['point_hits']}/108 |")
    lines+=['',f'明确收益门：{passed}。']
    if comparison:
        lines+=[f"race−control Scan差{comparison['scan_delta']:+.9f}，患者配对95%区间{comparison['scan_delta_ci95']}；Subject差{comparison['subject_delta']:+.9f}，Point差{comparison['point_hits_delta']:+d}。"]
    lines+=['','两组随机初始化、匹配骨干/有效batch/更新和增强；新增参照、类型对比与失效训练为组合因素，不单独归因某一组件。单seed和反复使用验证限制保留，历史270强参考不是本组新成绩。',
            '','工程原型或合成通过不能代替患者指标；负结果保持原样。未完成时不称完整200轮。',
            '','[方案](IDEA.md) · [机器结果](RESULT.json) · [当前状态](PIPELINE_STATUS.json)']
    (R/'RESULT.md').write_text('\n'.join(lines)+'\n')
    # New summaries are short and isolated from historical experiment records.
    entry=f"\n\n<!-- RACE_20261028_RESULT -->\n2026-10-28 RACE-Seg匹配实验{result['state']}，收益门{passed}。原972/108、单T1ce、单seed；无新test270/outer/confirmation。[结果]({R.relative_to(ROOT)}/RESULT.md)。\n<!-- END_RACE_20261028_RESULT -->\n"
    for name in ('AFLOW_BRATS_RESULTS.md','AFLOW_BRATS_WEEKLY_REPORT.md'):
        path=ROOT/name;text=path.read_text()
        if '<!-- RACE_20261028_RESULT -->' not in text:path.write_text(text.rstrip()+entry)
    summary=f"\n<!-- RACE_PREP_20261028 -->\n## 2026-10-28 RACE-Seg匹配实验 {result['state']}\n\n"
    for arm,s in states.items():
        m=selected[arm]['metrics']
        summary+=f"{arm}完成{s['last_complete_epoch']}轮，最佳{s['best_epoch']}轮108验证Scan {m['scan_dice']:.6f}、Subject {m['subject_dice']:.6f}、Point {m['point_hits']}/108。 "
    summary+=f"明确收益门{passed}。单seed、复用108开发；无新270/outer/confirmation。本轮结束，不自动恢复或扩组。[结果]({R.relative_to(ROOT)}/RESULT.md)。\n<!-- END_RACE_PREP_20261028 -->\n"
    for name in ('AFLOW_BRATS_HANDOFF_CURRENT.md','AFLOW_BRATS_METHOD_CATALOG.md'):
        path=ROOT/name;text=path.read_text()
        start=text.find('<!-- RACE_PREP_20261028 -->');end=text.find('<!-- END_RACE_PREP_20261028 -->')
        if start>=0 and end>=start:
            text=text[:start]+summary.strip()+text[end+len('<!-- END_RACE_PREP_20261028 -->'):]
        else:text=text.rstrip()+summary
        path.write_text(text)
    path=ROOT/'AGENTS.md';text=path.read_text();start=text.find('<!-- RACE_AUTH_20261028 -->');end=text.find('<!-- END_RACE_AUTH_20261028 -->')
    if start>=0 and end>=start:
        block=text[start:end]
        block=block.replace('当前worker1152470两卡空闲核验通过，源码实现与原生整脑GPU合成门通过，缓存准备进行中，尚无新患者模型训练结果。',
                            f"当前本轮已结束，状态{result['state']}，明确收益门{passed}；最新结果见本轮RESULT.md，不自动恢复或新增测试。")
        text=text[:start]+block+text[end:];path.write_text(text)
    write(R/'PIPELINE_STATUS.json',{'at':now(),'state':result['state'],'training_finished':True,'clear_benefit_gate':passed,'test_access':False})

if __name__=='__main__': main()
