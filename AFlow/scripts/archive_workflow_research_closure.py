"""Evidence index and bounded thesis handoff; no additional scientific evaluations."""
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from scripts import structured_background_protocol as p
from scripts.publish_research_card import publish

RECORD=p.ROOT/'archive/research_records/2026-09-11_workflow_research_closure'
BASE=p.ROOT/'archive/research_records'
STUDIES=['2026-09-11_automatic_workflow_engineering','2026-09-11_automatic_workflow_search',
         '2026-09-11_workflow_search_seed_sensitivity','2026-09-11_automatic_workflow_confirmation',
         '2026-09-11_workflow_file_to_output_timing']


def main():
    if (RECORD/'RESULT.json').exists():raise FileExistsError('closure already archived')
    data={name:json.loads((BASE/name/'RESULT.json').read_text()) for name in STUDIES}
    assert all((BASE/name/'RESULT.md').is_file() for name in STUDIES)
    search=data[STUDIES[1]];confirmation=data[STUDIES[3]];timing=data[STUDIES[4]]
    assert search['development_engineering_loop_passed'] and not confirmation['both_models_passed']
    assert timing['status']=='completed' and timing['input_cache_exact_count']==16
    integrity=[]
    for name in STUDIES:
        directory=BASE/name
        for file in directory.glob('*.json'):
            if file.name.startswith('LEADERBOARD_') or file.name.startswith('CARD_PUBLICATION_'):continue
            json.loads(file.read_text())
            integrity.append({'path':str(file.relative_to(p.ROOT)),'sha256':p.sha(file)})
    board=json.loads((p.ROOT/'AFLOW_BRATS_LEADERBOARD.json').read_text())
    ranked=board['test270_direct_locator_rank_by_scan']
    assert len(ranked)==15
    best=ranked[0];single=next(r for r in ranked if r.get('locator_forwards')==1)
    linkage=[]
    for name in STUDIES+[RECORD.name]:
        for path in (BASE/name).glob('*.md'):
            if path.name=='CARD.md' or path.name.startswith('LEADERBOARD_'):continue
            for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
                if target.startswith(('https://','http://','#')):continue
                target=target.strip('<>').split('#',1)[0]
                if not target:continue
                resolved=(path.parent/target).resolve()
                assert resolved.exists(),f'broken archive link {path}: {target}'
                linkage.append({'document':str(path.relative_to(p.ROOT)),'target':target})
    result={'status':'bounded_workflow_research_chain_closed','created_utc':datetime.now(timezone.utc).isoformat(),
        'study_result_sha256':{name:p.sha(BASE/name/'RESULT.json') for name in STUDIES},
        'development_and_online_passed':True,'two_model_fixed_internal_confirmation_passed':False,
        'all_thesis_parts_completed':False,
        'forward_accounting':{'previous_turn_engineering':1654,'current_turn_search_smoke':496,
            'current_turn_development':2880,'current_turn_online_fit':1675,
            'current_turn_confirmation_smoke':168,'current_turn_test':5670,
            'current_turn_file_to_output':147,'current_turn_total':11036,'full_chain_total':12690},
        'new_training_runs':0,'new_checkpoints':0,
        'independent_score_recounts':46080,'independent_test_output_recounts':1620,
        'internal_overall_scan':best['scan_dice'],'internal_one_forward_scan':single['scan_dice'],
        'current_leader_methods':{'overall':best['method'],'one_forward':single['method']},
        'file_to_output':timing['summary'],'source_artifacts':integrity,
        'archive_links_checked':len(linkage),'test_access_scope':'one fixed reused test270/119 panel; no test-based retuning',
        'protected_outer_confirmation_fixedval_external_new_access':False,
        'next_decision':'do not reopen failed mirror-subset protocol; retain bounded efficiency chapter, prioritize paper claim audit and independent hypotheses without test feedback'}
    p.write_new(RECORD/'RESULT.json',result)
    lines=['# 工作流研究交接：实验链闭环，不等于所有假设成功','',
        '本轮从开发搜索持续推进到独立评分核查、真实在线执行、20种子敏感度、固定内部确认和文件到输出耗时，没有只停在测速或开发正结果。','',
        '## 三个关键结论','',
        '1. 自动系统的模型阶段剪枝可行：开发中8次降为2/3次前向，实测约3.96/2.65倍；代价是小幅Dice下降。',
        '2. 固定内部确认未稳定通过：seed11少命中5例、Scan损失约0.00327等门失败，seed12通过；不得择seed宣布总体成功。',
        '3. 总流程主要受预处理限制，模型阶段倍率不能搬到文件到输出口径。四例探针数字如下。','',
        '| 模型seed | 总加速配对中位数 | Full8预处理占比 |','|---|---:|---:|']
    for seed,s in timing['summary'].items():lines.append(f"| {seed} | {s['paired_total_speedup_median']:.4f}× | {s['full8_preprocessing_share']:.2%} |")
    lines.extend(['','总耗时仅为暖页NIfTI读取到128³CPU掩膜/点，不含模型加载/原生导出/临床其他过程；4扫描来自2位患者，是工程探针，不给群体CI。16/16重建输入与原cache字节一致。','',
        '## 论文三部分重新定位','',
        '| 部分（依据归档贡献候选） | 当前状态 | 合理下一步 |','|---|---|---|',
        '| 自动定位分割 | 已有系统及内部/公开外部证据；未证理论上限 | 整理结果，暂停重复弱增强配方 |',
        '| 训练与评估方法 | 长时程、蒸馏等已有结果；独立创新与因果归因仍需审视 | 整理明确对照，保留教师历史接触等限制 |',
        '| 工作流搜索剪枝 | 开发—工程—固定内部报告链已完成；稳定确认假设失败 | 可写受限效率/稳定性章节，不可写普遍无损压缩或树搜索优越 |','',
        '所以不能宣布三项论文工作已全部成功完成，也不应该为了不停止而回头调已经失败的test视图。实验闭环完成与科学假设成立是两件事。本次第三部分已经形成可审核章节初稿。','',
        '## 资源、验证与访问','',
        '本用户轮零训练、新权重，实际11036模型病例前向；前一轮工程1654另列，整个链12690。开发46080候选病例评分独立重算；测试1620个保存输出独立验证。新增主要/tmp材料为22.5GiB开发原子、12.656GiB测试logits、405MiB测试掩膜，全部保留，无删除。','',
        f"同一reused internal test270/119总体Scan最高{best['scan_dice']:.9f}，单次前向最高{single['scan_dice']:.9f}，未超榜。本次新增六行，榜单共15条自动测试方法。",'',
        '主开发180/78，内部测试270/119分别报告，不混排、不称新盲测；新测试仅固定六输出一次，后续只是保存结果核验。旧outer/confirmation108/fixed validation108/外部无新性能访问。完整CI见主实验和内部确认报告。','',
        '## 阅读入口','',
        '- [第三项工作章节初稿](CHAPTER_DRAFT.md)',
        '- [开发与在线主结果](../2026-09-11_automatic_workflow_search/RESULT.md)',
        '- [随机种子敏感度](../2026-09-11_workflow_search_seed_sensitivity/RESULT.md)',
        '- [固定内部确认及权重](../2026-09-11_automatic_workflow_confirmation/RESULT.md)',
        '- [总耗时边界](../2026-09-11_workflow_file_to_output_timing/RESULT.md)',
        '- [机器总账](RESULT.json)、[质量核验](QUALITY.json)',
        '- [当前交接](../../../AFLOW_BRATS_HANDOFF_CURRENT.md)、[方法目录](../../../AFLOW_BRATS_METHOD_CATALOG.md)、[排行榜](../../../AFLOW_BRATS_LEADERBOARD.md)'])
    with (RECORD/'RESULT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    card='## 工作流研究链已闭环：开发效率成立，固定内部确认未稳定通过\n\n'
    card+='本轮连续完成128子集搜索、公平控制器对照、独立评分核查、在线测速、20种子敏感度、固定test270及文件到输出耗时。只改预定镜像策略，模型/权重/输入/阈值/点保持控制，0新训练。开发2/3×前向、3.96/2.65倍阶段加速通过；test seed11丢5个点等门失败，seed12通过，不能择seed宣布成功。\n\n'
    for seed,s in timing['summary'].items():card+=f"文件到输出工程探针seed{seed}总加速{s['paired_total_speedup_median']:.4f}倍，预处理占{s['full8_preprocessing_share']:.2%}；不是同倍端到端加速。\n\n"
    card+='本轮11036模型前向（此前工程1654另计），开发46080评分和测试1620保存输出独立核验。新增一次reused internal test270/119六输出全部入榜，旧outer/confirmation/固定val/外部未访问。无新checkpoint，大型/tmp原子/logits/masks全部保留。全榜总体/1× Scan仍0.757269662/0.752003176。\n\n'
    card+='下一步收束论文主张及独立问题，不复活已关闭test视图搜索；“实验链完整”不等于“假设全部成功”，更不等于全论文三项已完成。已写第三项工作章节初稿。\n\n'
    card+='[完整交接与成本/边界](archive/research_records/2026-09-11_workflow_research_closure/RESULT.md)、[章节初稿](archive/research_records/2026-09-11_workflow_research_closure/CHAPTER_DRAFT.md)、[机器总账](archive/research_records/2026-09-11_workflow_research_closure/RESULT.json)。\n'
    with (RECORD/'CARD.md').open('x') as f:f.write(card)
    publish(str(RECORD.relative_to(p.ROOT)))
    validators={}
    for script in ('verify_aflow_brats_doc_layout.py','verify_aflow_brats_method_catalog.py','verify_aflow_brats_leaderboard.py'):
        run=subprocess.run([sys.executable,str(p.ROOT/'scripts'/script)],cwd=p.ROOT,capture_output=True,text=True)
        validators[script]={'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr}
        assert run.returncode==0,run.stdout+run.stderr
    p.write_new(RECORD/'QUALITY.json',{'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),
        'validators':validators,'archive_links_checked_before_closure_publish':len(linkage),
        'full_regression_observed_terminal':{'passed':973,'warnings':90,'seconds':39.37,
                                           'scope':'full tests run during fixed test confirmation; subsequent additions are engineering/report scripts'},
        'result_sha256':p.sha(RECORD/'RESULT.json'),'markdown_sha256':p.sha(RECORD/'RESULT.md'),
        'chapter_sha256':p.sha(RECORD/'CHAPTER_DRAFT.md'),
        'publisher_sha256':p.sha(Path(__file__))})
    print(json.dumps({'status':result['status'],'file_to_output':timing['summary'],'validators_passed':True},indent=2))


if __name__=='__main__':main()
