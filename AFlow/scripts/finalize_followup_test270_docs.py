"""Close the two fixed tests from saved results only; never run model inference."""
import datetime
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / 'archive/research_records/2026-09-21_segresnet_followup_test270'
R = {k: ROOT / ('archive/research_records/2026-09-21_segresnet_' + v + '_test270')
     for k, v in [('native', 'pretrain_native_e166'), ('patient', 'patient_equal_e172')]}


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def paragraph(text, prefix, replacement):
    return re.sub(r'^' + re.escape(prefix) + r'[^\n]*', lambda _: replacement, text, count=1, flags=re.M)


def main():
    d = {}
    evidence = {}
    freeze = json.loads((S / 'FREEZE.json').read_text())
    for name, expected in freeze['files'].items():
        assert sha(ROOT / name) == expected, name
    cache = json.loads((S / 'NATIVE_CACHE_RECEIPT.json').read_text())
    cache_index = Path(cache['cache']) / 'index.json'
    assert sha(cache_index) == cache['index_sha256'], 'run audit on worker with native cache'
    reference = json.loads((S / 'PRETRAINED179_NATIVE_RAW.json').read_text())
    assert reference['max_grid128_dice_error'] <= 1e-6
    manifest = [json.loads(x) for x in (ROOT / 'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/sealed_test/test_cases_t1ce.jsonl').read_text().splitlines() if x.strip()]
    for k, r in R.items():
        assert (r / 'DONE.json').exists(), k
        d[k] = json.loads((r / 'RESULT.json').read_text())
        rows = json.loads((r / 'RAW_TEST270.json').read_text())['summary']['per_case']
        assert [(x['case_id'], x['subject_id']) for x in rows] == [(x['case_id'], x['subject_id']) for x in manifest]
        assert len(rows) == len({x['case_id'] for x in rows}) == 270
        assert len({x['subject_id'] for x in rows}) == 119
        assert abs(sum(x['locator_dsc'] for x in rows) / 270 - d[k]['scan_dice']) < 1e-12
        assert sum(x['automatic_point_hits_gt'] for x in rows) == d[k]['point_hits']
        for n, h in d[k]['evidence'].items():
            assert sha(ROOT / n) == h, n
        evidence[str((r / 'RESULT.json').relative_to(ROOT))] = sha(r / 'RESULT.json')
    n, p = d['native'], d['patient']
    cn = n['comparisons']['native173']
    cp = p['comparisons']['pretrained179']
    links = {k: str((r / 'RESULT.md').relative_to(ROOT)) for k, r in R.items()}
    brief = (f"本次两个固定候选各一次270扫描/119患者测试均已完成。组合166在native grid的Scan {n['scan_dice']:.6f}、"
             f"Subject {n['subject_dice']:.6f}、Point {n['point_hits']}/270；对原native173差{cn['scan_delta']:+.6f}、"
             f"95%CI[{cn['scan_delta_ci95'][0]:+.6f},{cn['scan_delta_ci95'][1]:+.6f}]。患者等权172在128³的Scan {p['scan_dice']:.6f}、"
             f"Subject {p['subject_dice']:.6f}、Point {p['point_hits']}/270；对预训练179差{cp['scan_delta']:+.6f}、"
             f"95%CI[{cp['scan_delta_ci95'][0]:+.6f},{cp['scan_delta_ci95'][1]:+.6f}]。"
             f"不同栅格不直接相减。验证门未过的事实保留；本次为用户授权、复用内部队列探索，无本批test调优或新outer/confirmation。"
             f"[组合测试]({links['native']}) · [患者等权测试]({links['patient']})。")

    # Target exact paragraphs/markers only, preserving unrelated concurrent work.
    path = ROOT / 'AFLOW_BRATS_HANDOFF_CURRENT.md'
    text = path.read_text()
    text = paragraph(text, '**最新测试授权：**', '**本次固定候选测试终态：** ' + brief)
    text = paragraph(text, '**训练与验证终态：**', '**训练与验证终态：** 两组均完成200轮/24400更新及108/50验证，组合选166、患者等权选172；均未过明确提升门。随后用户另行授权的270评测已完成，训练阶段无test标记仅描述当时阶段。此次测试结束，不自动恢复本批训练或重复评测；其他批次见其独立协议与状态。')
    text = paragraph(text, '固定基线已建立；八镜像test取消', '固定基线已建立；八镜像test取消、无N4停止、原始分辨率173完成的边界保留。本次组合166与患者等权172训练、固定测试和归档均完成，没有待自动恢复的本批任务；本次未动用test调优许可。其他新授权批次见独立状态，不能据本批终态停止其他任务。')
    text = paragraph(text, '用户随后明确授权组合训练并追加', '组合训练及单卡患者等权实验现已完成训练、验证和用户追加授权的固定270测试；旧提案与阶段日志保留为历史。')
    path.write_text(text)

    for filename in ['AFLOW_BRATS_HANDOFF_CURRENT.md', 'AFLOW_BRATS_WEEKLY_REPORT.md']:
        path = ROOT / filename
        text = path.read_text()
        for mark, k in [('PRETRAIN_NATIVE_COMBINATION', 'native'), ('PATIENT_EQUAL_LOSS', 'patient')]:
            line = f"训练/108验证阶段已完成200轮，选第{d[k]['epoch']}轮，验证门未过；后续用户授权test270也已完成。旧训练JSON中的无test标记仅指训练阶段。[本候选测试与访问补记]({links[k]})。"
            text = re.sub(r'<!-- ' + mark + r' -->.*?<!-- END_' + mark + r' -->', lambda _: f'<!-- {mark} -->\n{line}\n<!-- END_{mark} -->', text, flags=re.S)
        if filename.endswith('WEEKLY_REPORT.md'):
            text = paragraph(text, '更新：2026-09-21。', '更新：2026-09-21。本次组合与患者等权两组均完成200轮、108验证和用户追加授权的一次270测试；结果见下，其他授权批次单独记录。')
            top = '**新增固定评测结果：** ' + brief
            if '**新增固定评测结果：**' in text:
                text = paragraph(text, '**新增固定评测结果：**', top)
            else:
                split = text.index('\n\n', text.index('更新：'))
                text = text[:split] + '\n\n' + top + text[split:]
            text = paragraph(text, '新批次只改变各自一个主要研究因素：', '本批主要研究因素：组合在原native配方中改用公开预训练初始化；患者等权在预训练128³配方中改每例损失系数。均保持单T1ce、SegResNet和原972/108/270；训练分别约双卡3小时49分、单卡2小时39分，部署仍单次前向。')
            text = paragraph(text, '本批仅验证结论：', '**本批测试结论：** ' + brief)
            text = paragraph(text, '当前纯自动基线和证据已建立，无待自动执行训练或测试。', '本次两个固定候选的评测和证据已归档，不据本批结果自动改参数、换epoch或重复测试；其他另行授权的研究批次按独立协议执行。固定193仍为比较基线。')
        path.write_text(text)

    path = ROOT / 'AGENTS.md'
    text = path.read_text()
    heading = '## 2026-09-21组合166与患者等权172的270测试授权（当前优先）'
    note = f"\n\n- **本次终态：** {brief} 真实总状态见共同归档PIPELINE_STATUS/COMPLETION_AUDIT；两次模型访问均完成，不自动重跑。此终态仅对应166/172固定评测，不覆盖其他授权批次。"
    if '真实总状态见共同归档PIPELINE_STATUS/COMPLETION_AUDIT' not in text:
        text = text.replace(heading, heading + note, 1)
    path.write_text(text)

    path = ROOT / 'AFLOW_BRATS_METHOD_CATALOG.md'
    text = path.read_text()
    for prefix, k in [('## 公开预训练初始化＋原始分辨率微调｜', 'native'), ('## 预训练SegResNet患者等权损失｜', 'patient')]:
        a = text.index(prefix)
        b = text.find('\n## ', a + 3)
        b = len(text) if b < 0 else b
        card = text[a:b]
        cost = ('双A100微调约3小时49分、峰值35.4GiB/卡；训练/验证临时缓存约26.8GiB，测试另约7.03GiB。' if k == 'native'
                else '单A100微调约2小时39分，沿用128³缓存，不增加部署损失计算。')
        card = paragraph(card, '**代价/下一步：**', f"**代价/下一步：** {cost}部署1次前向、权重约18MiB；本次缓存就绪后270推理{d[k]['runtime']['inference_seconds']:.1f}秒。此实现已完成，不自动重测或改参数。[核对材料]({links[k]})。")
        card = card.replace('尚未测试组合，跨栅格涨幅不能相加。', '当时尚未测试组合，跨栅格涨幅不能相加。')
        card = card.replace('后者需另行固定同栅格验证诊断。', '已完成固定同栅格验证和测试参考重评分。')
        text = text[:a] + card + text[b:]
    path.write_text(text)

    report = '# 两个固定候选的270测试汇总\n\n' + brief + '\n\n'
    report += '| 候选 | 评分栅格 | Scan Dice | Subject Dice | Point |\n|---|---|---:|---:|---:|\n'
    for k, title, grid in [('native', '公开预训练＋原分辨率166', 'native'), ('patient', '预训练＋患者等权172', '128³')]:
        x = d[k]
        report += f"| {title} | {grid} | {x['scan_dice']:.9f} | {x['subject_dice']:.9f} | {x['point_hits']}/270 |\n"
    report += '\n每个候选与同格参考的全部配对区间在各自报告；预训练179的native参考只是同模型重评分，不是新训练方法或额外候选。未因测试分数调整任何模型、阈值或参数，没有新outer/confirmation。\n\n'
    report += '[组合报告](../' + R['native'].name + '/RESULT.md) · [患者等权报告](../' + R['patient'].name + '/RESULT.md) · [共同预注册](IDEA.md)。\n'
    (S / 'RESULT.md').write_text(report)
    write(S / 'RESULT.json', {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status': 'both_fixed_tests_complete',
          'candidates': {k: {q: x[q] for q in ['epoch', 'scan_dice', 'subject_dice', 'point_hits', 'comparisons', 'runtime']} for k, x in d.items()},
          'test_access': True, 'current_test_tuning': False, 'new_outer_confirmation_access': False, 'evidence': evidence})
    checks = []
    for script in ['verify_aflow_brats_leaderboard.py', 'verify_aflow_brats_method_catalog.py', 'verify_aflow_brats_doc_layout.py', 'verify_aflow_root_artifacts.py']:
        x = subprocess.run([sys.executable, 'scripts/' + script], cwd=ROOT, capture_output=True, text=True)
        checks.append({'script': script, 'returncode': x.returncode, 'stdout': x.stdout, 'stderr': x.stderr})
    write(S / 'FINAL_DOC_CHECKS.json', checks)
    assert all(x['returncode'] == 0 for x in checks), checks
    for r in R.values():
        for filename in ['RESULT.md', 'RESULT.json', 'RAW_TEST270.json', 'MODEL_PREACCESS.json', 'CACHE_RECEIPT.json', 'checkpoint_selected.pth']:
            evidence[str((r / filename).relative_to(ROOT))] = sha(r / filename)
    write(S / 'COMPLETION_AUDIT.json', {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'status': 'complete', 'candidate_counts': {'scans_each': 270, 'subjects_each': 119},
          'all_evidence_hashes_checked': True, 'raw_aggregate_checked': True, 'all_four_document_checks_passed': True,
          'frozen_sources_checked': True, 'native_cache_index_rehashed': cache['index_sha256'],
          'reference128_reproduction_error': reference['max_grid128_dice_error'],
          'no_model_inference_in_this_audit': True, 'evidence': evidence, 'script_sha256': sha(Path(__file__))})
    print(json.dumps({'status': 'complete', 'results': str(S / 'RESULT.md')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
