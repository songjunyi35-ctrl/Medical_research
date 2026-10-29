"""Head-node arithmetic, archival and text publication after both fixed tests.

No clinical arrays are transferred or opened here. Worker-local independent
saved-mask audits must already have passed and be bound to the fixed selection.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np

from .governance import ROOT, RECORD, atomic_json, guard, sha
from .audit_validation import validate, aggregate, paired

A = ROOT / 'archive/research_records/2026-10-29_race_v2_inference_a'
B = ROOT / 'archive/research_records/2026-10-29_race_v2_inference_b'
TRAIN = ROOT / 'archive/research_records/2026-10-29_race_v2_training'
OLD = ROOT / 'archive/research_records/2026-10-29_race_e164_inference_test270'


def read(path):
    return json.loads(Path(path).read_text())


def block(name, body):
    path = ROOT / name
    text = path.read_text()
    begin, end = '<!-- RACE_V2_STAGE_20261029 -->', '<!-- END_RACE_V2_STAGE_20261029 -->'
    assert text.count(begin) == text.count(end) == 1
    path.write_text(text[:text.index(begin)] + begin + '\n' + body + '\n' + end +
                    text[text.index(end) + len(end):])


def complete():
    selection_path = RECORD / 'FINAL_TEST_SELECTION.json'
    selection = read(selection_path)
    selection_sha = sha(selection_path)
    assert selection['registered_by'] == 'root' and len(selection['candidates']) == 2
    assert {c['kind'] for c in selection['candidates']} == {'single', 'fusion'}
    cohort = [json.loads(line) for line in Path(selection['cohort']['manifest_path']).read_text().splitlines() if line.strip()]
    assert len(cohort) == 270 and len({r['subject_id'] for r in cohort}) == 119
    outputs, metrics, evidence = {}, {}, {}
    paths = {'single': A / 'final270/RAW_SINGLE.json', 'fusion': B / 'final270/RAW_FUSION.json'}
    for kind, path in paths.items():
        raw = read(path)
        assert raw['selection_sha256'] == selection_sha and raw['candidate_id'] == kind
        audit_path = Path(raw['masks_audit_receipt_path'])
        assert sha(audit_path) == raw['masks_audit_receipt_sha256']
        audit = read(audit_path)
        assert audit['status'] == 'passed' and audit['scans'] == 270
        assert audit['selection_sha256'] == selection_sha and audit['all_masks_counts_points_exact']
        records = raw['per_case']
        validate(records, cohort)
        outputs[kind] = records
        metrics[kind] = aggregate(records)
        evidence[str(path.relative_to(ROOT))] = sha(path)
        evidence[str(audit_path.relative_to(ROOT))] = sha(audit_path)
        atomic_json(RECORD / f'RAW_TEST270_{kind}.json', dict(
            per_case=records, metrics=metrics[kind], selection_sha256=selection_sha,
            source_report_path=str(path), source_report_sha256=sha(path),
            mask_audit_path=str(audit_path), mask_audit_sha256=sha(audit_path), test_tuned=True))
    old_raw = read(OLD / 'RAW_TEST270_raw.json')['per_case']
    old_full = read(OLD / 'RAW_TEST270_enhanced.json')['per_case']
    comparisons = {f'{kind}_minus_race164_old_full': paired(records, old_full) for kind, records in outputs.items()}
    comparisons['fusion_minus_single'] = paired(outputs['fusion'], outputs['single'])
    comparisons['single_minus_race164_raw'] = paired(outputs['single'], old_raw)
    model_map = {model['model_id']: model for candidate in selection['candidates'] for model in candidate['models']}
    atom_runtime = {}
    for name, model in model_map.items():
        ready = read(model['atoms_ready_path'])
        assert ready['checkpoint_sha256'] == model['checkpoint_sha256']
        assert ready['case_forwards'] == 2160 and len(ready['atoms']) == 2160
        atom_runtime[name] = {key: ready[key] for key in ready if key in (
            'new_model_forwards', 'network_seconds', 'seconds', 'batch_calls', 'peak_reserved_gib', 'case_forwards')}
        atom_runtime[name]['new_model_forwards'] = 0 if model['atom_source'] == 'reuse_race164' else 2160
    training = read(TRAIN / 'RESULT.json')
    result = dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(), state='complete',
                  cohort=dict(scans=270, subjects=119), metrics=metrics, comparisons=comparisons,
                  scan_above080={kind: score['scan_dice'] > .8 for kind, score in metrics.items()},
                  model_runtime=atom_runtime, new_test_case_forwards=sum(row['new_model_forwards'] for row in atom_runtime.values()),
                  per_case_deployment_forwards=dict(single=8, fusion=16),
                  training_new_epochs=training['actual_new_epochs'], training_new_updates=training['actual_new_updates'],
                  selection_path=str(selection_path.relative_to(ROOT)), selection_sha256=selection_sha,
                  source_reports_sha256=evidence, masks_audited=540, raw_integer_arithmetic_verified=True,
                  inherited_test_tuned_history=True, current_test_parameter_search=False,
                  independent_confirmation=False, new_outer_confirmation_access=False,
                  interpretation='Single and pureSegResNet pair reported separately; fixed108-selected recipes; no causal matched control')
    innovation108 = read(RECORD / 'INFERENCE_INNOVATION108_RESULT.json')
    assert innovation108['state'] == 'complete'
    result['inference_innovation108'] = dict(
        path=str(RECORD / 'INFERENCE_INNOVATION108_RESULT.json'),
        sha256=sha(RECORD / 'INFERENCE_INNOVATION108_RESULT.json'),
        conditional_stability_only=True, stable_improvement_established=False)
    result['selected_deployment_algorithms'] = {
        item['kind']: item.get('single_recipe', item.get('fusion')) for item in selection['candidates']}
    atomic_json(RECORD / 'RESULT.json', result)
    lines = ['# RACE-v2新阶段最终270评测', '',
             '原270扫描／119患者，单T1ce、原生1mm原始裁剪栅格，排除填充但保留整个裁剪GT。模型与统一配方只由108选择，最终恰两个候选，无270反向调参；继承test-tuned历史，非独立盲测。', '',
             '| 类别 | Scan | Subject | Point | 部署前向／例 | Scan>0.8000 |', '|---|---:|---:|---:|---:|---|']
    for kind, label in [('single', '单模型'), ('fusion', '纯SegResNet双模型')]:
        m = metrics[kind]
        lines.append(f"| {label} | {m['scan_dice']:.9f} | {m['subject_dice']:.9f} | {m['point_hits']}/270 | {8 if kind=='single' else 16} | {m['scan_dice']>.8} |")
    lines += ['', f"训练新增{training['actual_new_epochs']}轮／{training['actual_new_updates']}更新，使用新AdamW，非无缝续训。最终模型与统一配方、108选择依据和源码SHA见[冻结选择](FINAL_TEST_SELECTION.json)。", '',
              f"双模型相对单模型Scan差{comparisons['fusion_minus_single']['scan_delta']:+.9f}，患者配对95%区间{comparisons['fusion_minus_single']['scan_ci95']}。各方法相对旧RACE完整配方的Scan／Subject配对区间、Point差和全部变差病例见[机器结果](RESULT.json)。10000患者整簇bootstrap，seed20261029；选择后描述区间未校正搜索。", '',
              f"旧RACE164八视图直接复用。新270网络前向合计{result['new_test_case_forwards']}；各模型载荷和runtime分列机器结果，两个最终配置共享同一模型缓存，不重复网络。540份保存掩膜由worker本地独立重载，TP／FP／FN、Dice、定位点和SHA全核对；总控再独立核对整数统计和患者区间，未转移临床数组。", '',
              '新增16项单模局部相关分歧和12项双模协方差收缩在108的均值小幅提高，但配对区间跨0，未确认稳定收益；首轮融合分折稳定收益也未确认，最终结果不抹去该限制；训练机制贡献未由同期纯模型对照单独识别。所有失败、dtype修正和冻结补记保留，临床数组向主节点备份被自动审批拒绝且未执行。', '',
              '[新增推理108完整结果](INFERENCE_INNOVATION108_RESULT.md) · [单模型逐例](RAW_TEST270_single.json) · [双模型逐例](RAW_TEST270_fusion.json) · [首阶段108](INITIAL_STAGE_RESULT.md) · [训练交接](../2026-10-29_race_v2_training/HANDOFF.json) · [执行边界](RUNBOOK.md)', '']
    (RECORD / 'RESULT.md').write_text('\n'.join(lines))
    board = read(ROOT / 'AFLOW_BRATS_LEADERBOARD.json')
    key = 'test270_tuned_native_rank_by_scan'
    stage_ids = {f'race_v2_stage_20261029_{kind}_fixed108' for kind in ('single', 'fusion')}
    before = {row['method']: {k: v for k, v in row.items() if k != 'rank'}
              for row in board[key] if row['method'] not in stage_ids}
    for candidate in selection['candidates']:
        kind = candidate['kind']
        identifier = f'race_v2_stage_20261029_{kind}_fixed108'
        forward = 8 if kind == 'single' else 16
        report = RECORD / f'RAW_TEST270_{kind}.json'
        row = dict(method=identifier, **metrics[kind], point_hit=metrics[kind]['point_hits']/270, point_denominator=270,
                   cohort=dict(scans=270, subjects=119), ensemble_size=len(candidate['models']),
                   models=candidate['models'], fixed_candidate=candidate, grid='native1mm original crop excludes padding',
                   score_grid='native', locator_forwards=forward,
                   forward_equivalents=float(forward*np.prod([160,208,168])/128**3),
                   forward_cost_note='input voxel equivalence; not FLOPs; RACE references incur extra operations',
                   report_path=str(report.relative_to(ROOT)), report_sha256=sha(report),
                   result_path=str((RECORD/'RESULT.json').relative_to(ROOT)), result_sha256=sha(RECORD/'RESULT.json'),
                   test_tuned=True, current_test_retuning=False, input_modality='T1ce', uses_GT_for_localization=False,
                   evidence_status='fixed108 selection; reused historically test-tuned270; no current test parameter search; not independent confirmation',
                   new_outer_confirmation_access=False)
        board[key] = [old for old in board[key] if old['method'] != identifier] + [row]
    board[key].sort(key=lambda row: (-row['scan_dice'], row['method']))
    for rank, row in enumerate(board[key], 1):
        row['rank'] = rank
    for identifier, original in before.items():
        current = next(row for row in board[key] if row['method'] == identifier)
        assert {k: v for k, v in current.items() if k != 'rank'} == original
    board['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    board['active_research_stage_20261029'].update(status='complete', new_test270_results=True,
                                                  result_path=str((RECORD/'RESULT.md').relative_to(ROOT)))
    atomic_json(ROOT / 'AFLOW_BRATS_LEADERBOARD.json', board)
    p = ROOT / 'AFLOW_BRATS_LEADERBOARD.md'
    text = p.read_text(); begin = text.index('<!-- TEST270_TUNED_NATIVE -->'); end = text.index('<!-- END_TEST270_TUNED_NATIVE -->', begin)
    section = text[begin:end]; position = section.index('| 排名 | 方法/快照 |')
    table = '| 排名 | 方法/快照 | Scan Dice | Subject Dice | Point | 128³前向等价量 | 证据 |\n|---:|---|---:|---:|---:|---:|---|\n'
    table += '\n'.join(f"| {x['rank']} | {x['method']} | {x['scan_dice']:.9f} | {x['subject_dice']:.9f} | {x['point_hits']}/270 | {x.get('forward_equivalents',0):.3f}× | [报告]({x['report_path']}) |" for x in board[key])+'\n'
    p.write_text(text[:begin]+section[:position]+table+text[end:])
    s, f = metrics['single'], metrics['fusion']
    link = str((RECORD/'RESULT.md').relative_to(ROOT))
    body = (f"## 2026-10-29新阶段最终270完成\n\n原270/119、单T1ce、原生1mm，单模型Scan{s['scan_dice']:.9f}／Subject{s['subject_dice']:.9f}／Point{s['point_hits']}/270；纯SegResNet双模型Scan{f['scan_dice']:.9f}／Subject{f['subject_dice']:.9f}／Point{f['point_hits']}/270。Scan>0.8000分别为{s['scan_dice']>.8}/{f['scan_dice']>.8}。新增训练{training['actual_new_epochs']}轮／{training['actual_new_updates']}更新，新优化器微调；最终两个配方先108冻结，继承test-tuned历史，非独立确认。没有新outer/confirmation，本轮停止，不自动扩搜/续训/重测。[完整结果]({link})。")
    for name in ('AGENTS.md', 'AFLOW_BRATS_HANDOFF_CURRENT.md', 'AFLOW_BRATS_RESULTS.md', 'AFLOW_BRATS_LEADERBOARD.md'):
        block(name, body)
    card = body + ('\n\n**一句话：** 学习参考意见的收益并固定验证选择的多视图与双模型配方。\n**出发点：** 旧RACE完整推理0.789263，目标超过0.8000。\n**具体做法：** 单T1ce、固定972/108/270与原生1mm，新优化器校准后108选最多两个固定270配置。\n'
                   f"**结果：** 本阶段270单模型{s['scan_dice']:.9f}，双模型{f['scan_dice']:.9f}；区间和变差病例见报告。\n**是否支持直觉：** 以本次同队列实测为准；首轮融合分折稳定优势未确认，未识别独立算法因果收益。\n**代价/下一步：** 单模8／双模16前向，缓存共享；本轮关闭，不扩搜。")
    block('AFLOW_BRATS_METHOD_CATALOG.md', card)
    checks = {}
    for name in ('verify_aflow_brats_doc_layout.py', 'verify_aflow_root_artifacts.py', 'verify_aflow_brats_method_catalog.py', 'verify_aflow_brats_leaderboard.py'):
        result_check = subprocess.run([sys.executable, str(ROOT/'scripts'/name)], cwd=ROOT, check=True, text=True, capture_output=True)
        checks[name] = json.loads(result_check.stdout)
    atomic_json(RECORD / 'COMPLETION_AUDIT.json', dict(state='passed', masks_audited=540,
                result_sha256=sha(RECORD/'RESULT.json'), report_sha256=sha(RECORD/'RESULT.md'),
                selection_sha256=selection_sha, checks=checks, no_new_neural_forwards_during_root_audit=True))
    atomic_json(RECORD / 'PIPELINE_STATUS.json', dict(state='complete', at=result['at'],
                result_path=str(RECORD/'RESULT.md'), scan_above080=result['scan_above080'],
                current_test_parameter_search=False, automatic_continuation=False))
    subprocess.run([sys.executable, '-m', 'scripts.race_v2_stage_20261029.publish_innovation', '--message',
                    'Archive fixed single SegResNet and pure SegResNet fusion results for RACE-v2 stage'], cwd=ROOT, check=True)


def main():
    while True:
        guard(1154121)
        if (RECORD/'PIPELINE_STATUS.json').exists() and read(RECORD/'PIPELINE_STATUS.json').get('state') == 'complete':
            return
        if ((RECORD/'FINAL_TEST_SELECTION.json').exists() and (A/'final270/RAW_SINGLE.json').exists()
                and (B/'final270/RAW_FUSION.json').exists()):
            complete()
            return
        time.sleep(30)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        atomic_json(RECORD/'FINAL_PUBLICATION_FAILURE.json', {'at_unix': time.time(), 'error': repr(error),
                    'saved_predictions_retained': True, 'no_automatic_neural_retry': True})
        raise
