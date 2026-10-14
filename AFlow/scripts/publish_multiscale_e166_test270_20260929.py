"""Publish every outcome without altering old scores or the validation gate."""
import copy
import fcntl
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.multiscale_e166_test270_20260929_common import *

DOCS = ['AFLOW_BRATS_HANDOFF_CURRENT.md', 'AFLOW_BRATS_METHOD_CATALOG.md', 'AFLOW_BRATS_RESULTS.md',
        'AFLOW_BRATS_WEEKLY_REPORT.md', 'AFLOW_BRATS_LEADERBOARD.md', 'AFLOW_BRATS_LEADERBOARD.json', 'AGENTS.md']
MARK = 'MULTISCALE_E166_TEST270_20260929'
REL = str(R.relative_to(ROOT))


def block(path, mark, body):
    text = path.read_text()
    pattern = f'<!-- {mark} -->.*?<!-- END_{mark} -->'
    value = f'<!-- {mark} -->\n{body}\n<!-- END_{mark} -->'
    if len(re.findall(pattern, text, flags=re.S)) != 1:
        raise RuntimeError('missing or duplicate publication block: ' + str(path))
    path.write_text(re.sub(pattern, lambda _: value, text, flags=re.S))


def verdict(comparison):
    low, high = comparison['scan_delta_ci95']
    if low > 0:
        if comparison['scan_delta_ci97_5'][0] <= 0:
            return '复用队列的95%配对区间为正，但97.5%区间仍跨0；这是小幅收益的探索性证据，尚不能认定稳健或独立泛化提升'
        return '复用队列的配对区间为正，支持这次固定实现的Dice收益；仍不能作独立泛化结论'
    if high < 0:
        return '复用队列的配对区间为负，这次固定实现的Dice较差；不据此否定辅助监督的一般机制'
    return '配对区间跨0，没有确认稳定Dice增益；单次点估计不能证明优越或等效'


def board_update(result):
    with (R / 'board.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = ROOT / 'AFLOW_BRATS_LEADERBOARD.json'; board = read(path)
        before = copy.deepcopy(board); key = 'test270_tuned_native_rank_by_scan'
        previous = [r for r in board[key] if r['method'] != ID]
        row = {'method': ID, 'scan_dice': result['scan_dice'], 'subject_dice': result['subject_dice'],
               'point_hit': result['point_hit'], 'point_hits': result['point_hits'], 'point_denominator': 270,
               'cohort': {'scans': 270, 'subjects': 119}, 'checkpoint_epoch': 166,
               'checkpoint_path': REL + '/checkpoint_epoch166.pth', 'checkpoint_sha256': CHECKPOINT_SHA,
               'grid': 'native 1mm crop; 160x208x168 canvas; padding excluded', 'score_grid': 'native',
               'threshold': .5, 'raw_logit_threshold': 0., 'tta': False, 'postprocessing': False,
               'auxiliary_heads_in_deployment': False, 'locator_forwards': 1,
               'forward_equivalents': float(np.prod(TEST_CANVAS) / 128**3),
               'report_path': REL + '/RAW_TEST270.json', 'report_sha256': sha(R / 'RAW_TEST270.json'),
               'result_path': REL + '/RESULT.json', 'result_sha256': sha(R / 'RESULT.json'),
               'evidence_status': 'test-tuned development; validation-selected fixed one-shot exploratory test; reused internal; no current test retuning; not independent confirmation',
               'test_tuned': True, 'current_test_retuning': False, 'validation_gate_passed': False,
               'scan_ci95': result['scan_ci95'], 'subject_ci95': result['subject_ci95'],
               'role': 'candidate', 'logit_formula': 'single main-output whole-brain model',
               'training_scope': {'train_scans': 972, 'train_subjects': 444, 'validation_scans': 108,
                                  'validation_subjects': 50, 'epochs': 200, 'optimizer_updates': 24400,
                                  'public_pretraining_scans': 334, 'public_pretraining_epochs': 150},
               'paired_comparisons': result['comparisons'], 'new_outer_confirmation_access': False}
        ranked = previous + [row]
        ranked.sort(key=lambda r: (-r['scan_dice'], r['forward_equivalents'], r['method']))
        for i, item in enumerate(ranked, 1):
            item['rank'] = i
        board[key] = ranked; board['updated_utc'] = now()
        single = [r for r in ranked if r.get('role') != 'fixed_output_panel' and 'single' in r.get('logit_formula', '')]
        overall = ranked[0]; best_single = max(single, key=lambda r: r['scan_dice'])
        leaders = {'overall_native_test_tuned': {k: overall[k] for k in ['method', 'scan_dice', 'subject_dice', 'point_hits']},
                   'single_model_native_test_tuned': {k: best_single[k] for k in ['method', 'scan_dice', 'subject_dice', 'point_hits']}}
        board['latest_multiscale_e166_test270_20260929'] = {'candidate': ID, 'fixed_epoch': 166,
                   'test_tuned_development': True, 'validation_gate_passed': False, **leaders}
        for other_key, value in before.items():
            if other_key not in {key, 'updated_utc', 'latest_multiscale_e166_test270_20260929'} and board[other_key] != value:
                raise RuntimeError('historical ranking changed: ' + other_key)
        old_by_id = {r['method']: r for r in before[key] if r['method'] != ID}
        for item in ranked:
            if item['method'] in old_by_id:
                a = {k: v for k, v in item.items() if k != 'rank'}
                b = {k: v for k, v in old_by_id[item['method']].items() if k != 'rank'}
                if a != b:
                    raise RuntimeError('historical tuned row changed')
        write(path, board)
        md = ROOT / 'AFLOW_BRATS_LEADERBOARD.md'; text = md.read_text()
        table = '\n'.join(f"| {r['rank']} | {r['method']} | {r['scan_dice']:.9f} | {r['subject_dice']:.9f} | {r['point_hits']}/270 | {r['forward_equivalents']:.3f}× | [报告]({r['report_path']}) |" for r in ranked)
        replacement = ('<!-- TEST270_TUNED_NATIVE -->\n## 270例定向开发：原始栅格，test-tuned，独立分榜\n\n'
                       '本榜收录受复用270影响的开发结果及固定参考，全部非独立泛化证据。新多尺度166仅由108验证选定，本次测试未再择优或调参；历史传统榜不变。统一原始1mm裁剪栅格、排除填充；128³输出按各自协议三线性回原栅格。前向等价量按输入体素折算，并非实测耗时比；旧数值模式细微差异见各报告。\n\n'
                       '| 排名 | 方法/快照 | Scan Dice | Subject Dice | Point | 128³前向等价量 | 证据 |\n'
                       '|---:|---|---:|---:|---:|---:|---|\n' + table + '\n<!-- END_TEST270_TUNED_NATIVE -->')
        text = re.sub(r'<!-- TEST270_TUNED_NATIVE -->.*?<!-- END_TEST270_TUNED_NATIVE -->', lambda _: replacement, text, flags=re.S)
        text = re.sub(r'更新时间：[^\n]+', '更新时间：2026-09-29 UTC', text, count=1)
        md.write_text(text)
        write(R / 'LEADERBOARD_UPDATE.json', {'utc': now(), 'new_id': ID, 'new_rank': next(r['rank'] for r in ranked if r['method'] == ID),
              'historical_rows_preserved': True, 'traditional_rankings_unchanged': True, 'tuned_rows': len(ranked), **leaders})
        return leaders


def audit(result):
    verify(); raw = read(R / 'RAW_TEST270.json'); rows = raw['summary']['per_case']; validate_rows(rows)
    journal = read(R / 'INFERENCE_JOURNAL.json')
    if journal['completed_scans'] != 270 or journal['per_case'] != rows:
        raise RuntimeError('complete journal/raw mismatch')
    for name, digest in result['evidence'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('evidence hash mismatch: ' + name)
    # Recompute scalar summaries directly from raw rows without calling the summarizer.
    scan = float(np.mean([r['locator_dsc'] for r in rows]))
    patients = {r['subject_id'] for r in rows}
    subject = float(np.mean([np.mean([r['locator_dsc'] for r in rows if r['subject_id'] == s]) for s in patients]))
    if abs(scan - result['scan_dice']) > 1e-14 or abs(subject - result['subject_dice']) > 1e-14:
        raise RuntimeError('independent raw summary mismatch')
    if sum(r['automatic_point_hits_gt'] for r in rows) != result['point_hits']:
        raise RuntimeError('point count mismatch')
    combo = references()['old_combo166_native']
    by_id = {r['case_id']: r for r in combo}
    delta = float(np.mean([r['locator_dsc'] - by_id[r['case_id']]['locator_dsc'] for r in rows]))
    if abs(delta - result['comparisons']['old_combo166_native']['scan_delta']) > 1e-14:
        raise RuntimeError('comparison direction mismatch')
    board = read(ROOT / 'AFLOW_BRATS_LEADERBOARD.json'); ranked = board['test270_tuned_native_rank_by_scan']
    if [r['rank'] for r in ranked] != list(range(1, len(ranked) + 1)) or len({r['method'] for r in ranked}) != len(ranked):
        raise RuntimeError('tuned ranking identities/ranks invalid')
    if ranked != sorted(ranked, key=lambda r: (-r['scan_dice'], r['forward_equivalents'], r['method'])):
        raise RuntimeError('tuned ranking order invalid')
    markdown = (ROOT / 'AFLOW_BRATS_LEADERBOARD.md').read_text()
    for item in ranked:
        line = f"| {item['rank']} | {item['method']} | {item['scan_dice']:.9f} | {item['subject_dice']:.9f} | {item['point_hits']}/270 | {item['forward_equivalents']:.3f}× | [报告]({item['report_path']}) |"
        if markdown.count(line) != 1 or sha(ROOT / item['report_path']) != item['report_sha256']:
            raise RuntimeError('tuned Markdown/report hash mismatch: ' + item['method'])
    write(R / 'COMPLETION_AUDIT.json', {'utc': now(), 'status': 'passed', 'candidate_scans': 270,
          'candidate_subjects': 119, 'checkpoint_sha256': sha(CHECKPOINT), 'raw_sha256': sha(R / 'RAW_TEST270.json'),
          'result_sha256': sha(R / 'RESULT.json'), 'all_frozen_sources_and_evidence_exact': True,
          'raw_journal_and_independent_summaries_match': True, 'comparison_direction_candidate_minus_reference': True,
          'both_tuned_leaderboards_and_all_report_hashes_match': True,
          'one_candidate_model_access_receipt': True, 'reference_model_forwards_this_task': 0,
          'new_outer_confirmation_access': False, 'validation_gate_still_failed': True})


def main():
    verify(); result = read(R / 'RESULT.json')
    if result['status'] != 'complete_test270':
        raise RuntimeError('candidate test incomplete')
    for name in DOCS:
        destination = R / 'docs_before_publication' / name
        if not destination.exists():
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
    primary = result['comparisons']['old_combo166_native']; ci = primary['scan_delta_ci95']
    statement = verdict(primary)
    leaders = board_update(result)
    text = f'''# 多尺度辅助监督第166轮：270固定探索性评测完成

270扫描／119患者，**原始1mm裁剪栅格，排除补零**。Scan Dice **{result['scan_dice']:.9f}**（患者整簇95%CI [{result['scan_ci95'][0]:.6f}, {result['scan_ci95'][1]:.6f}]）；Subject Dice **{result['subject_dice']:.9f}**（95%CI [{result['subject_ci95'][0]:.6f}, {result['subject_ci95'][1]:.6f}]）；自动Point **{result['point_hits']}/270**。Point只评价最高logit位置是否命中肿瘤，不等同分割质量。

候选为固定108验证选出的166轮，非270选epoch；完整200轮/24400更新。验证Scan0.753092、Subject0.739669、Point94/108，对旧组合166差+0.001913、95%CI跨0，原冻结+0.01提升门未过。用户另行授权本次测试，**不事后改写验证门、不自动替换部署**。

| 同栅格既有参考 | 参考Scan | 候选−参考Scan | 119患者配对95%CI | 配对97.5%CI | Point差 |
|---|---:|---:|---|---|---:|
'''
    for key, comparison in result['comparisons'].items():
        text += f"| {key} | {result['reference_summaries'][key]['scan_dice']:.9f} | {comparison['scan_delta']:+.9f} | {comparison['scan_delta_ci95']} | {comparison['scan_delta_ci97_5']} | {comparison['point_hits_delta']:+d} |\n"
    text += f'''
主要匹配对照是同公开预训练＋原生整脑配方的旧组合166；本次研究变化仅训练时增加固定0.25/0.125半/四分之一尺度辅助监督。{statement}。对主要参考，{primary['improved_scans']}例改善、{primary['worse_scans']}例变差、{primary['tied_scans']}例相同；Subject差{primary['subject_delta']:+.6f}，95%CI{primary['subject_delta_ci95']}。这些区间是单次运行、复用内部队列的描述性不确定性，不能排除历史多次开发/选择偏倚。

绝对CI按10000次119患者整簇重采样seed20260917；配对CI候选减参考，10000次seed20260918。没有把多时间点扫描当独立患者。旧native173/193数值模式不同而本次与旧组合166同确定性bf16 batch2模式，全部复用历史逐例报告、无额外参考模型前向；数字差异不通过重新运行参考消除。不同栅格不得拿128³的0.759179直接相减。

先按原冻结预处理重建270原生缓存，图像/标签SHA与旧测试载荷完全一致，逐例canonical图像/标签/裁剪几何核对通过。图像自身框外历史标签合计4体素，未使用标签来调整框。单T1ce→N4→图像脑区框→非零z-score→原生1mm补零160×208×168→主输出整脑一次前向→raw logit>0→去补零评分。无标签定位、点击、八块滑窗、TTA、融合或后处理；辅助头不进入部署。

本候选新270图像/标签及模型访问已发生，MODEL_PREACCESS独占创建；135批、每例恰一次，RAW先于统计落盘。**test-tuned开发，复用内部探索，非独立盲测或外部泛化**；本次测试不用于改变epoch、阈值、画布或参数。无新outer/confirmation。旧训练报告无test标记只描述当时训练阶段，访问补记与本报告描述当前完整状态。

新增训练0轮；GPU0一张A10080GB、bf16 batch2，270推理{result['runtime']['inference_seconds']:.2f}秒，实测峰值{result['runtime']['peak_allocated_bytes']/2**30:.2f}GiB（含逐批小报告落盘，不含缓存构建）。缓存构建{read(R / 'CACHE_RECEIPT.json')['build_seconds']:.1f}秒、worker/tmp约7.03GiB；独立部署权重{CHECKPOINT.stat().st_size/2**20:.2f}MiB。160×208×168输入体素量是128³的2.666015625倍，不是实测耗时比；本次另一GPU不参与候选推理。

同native/test-tuned分榜总体最高Scan点估计{leaders['overall_native_test_tuned']['scan_dice']:.6f}，方法{leaders['overall_native_test_tuned']['method']}；单模型最高{leaders['single_model_native_test_tuned']['scan_dice']:.6f}，方法{leaders['single_model_native_test_tuned']['method']}。这里单模型允许多块前向，不等于1次全脑；所有旧行与原传统榜保持。新结果无自动重训/扩组/换epoch/重测/融合，当前评测结束。

[冻结七问](IDEA.md) · [原始逐例报告](RAW_TEST270.json) · [机器结果](RESULT.json) · [选优收据](SELECTION_RECEIPT.json) · [缓存一致性](CACHE_RECEIPT.json) · [访问收据](MODEL_PREACCESS.json) · [独立模型](checkpoint_epoch166.pth)。
'''
    (R / 'RESULT.md').write_text(text)
    brief = (f"**2026-09-29多尺度166固定270评测完成：** 原270扫描/119患者，native1mm裁剪栅格排除填充，Scan {result['scan_dice']:.6f}、Subject {result['subject_dice']:.6f}、Point {result['point_hits']}/270。"
             f"对同格旧组合166差{primary['scan_delta']:+.6f}、患者配对95%CI[{ci[0]:+.6f},{ci[1]:+.6f}]；{statement}。"
             f"原验证门未过，test-tuned开发/复用内部探索；此次唯一166模型已访问270，无当前test调参或新outer/confirmation。评测与归档结束，不自动重测或新训。[报告]({REL}/RESULT.md)。")
    for name in ['AFLOW_BRATS_HANDOFF_CURRENT.md', 'AFLOW_BRATS_WEEKLY_REPORT.md']:
        block(ROOT / name, MARK, brief)
    block(ROOT / 'AFLOW_BRATS_RESULTS.md', MARK, '## 2026-09-29｜多尺度166一次270评测完成\n\n' + brief +
          f" 新增训练0；主输出一次整脑前向，推理{result['runtime']['inference_seconds']:.1f}秒/峰值{result['runtime']['peak_allocated_bytes']/2**30:.2f}GiB，临时测试缓存约7.03GiB。")
    card = f'''## 原始分辨率多尺度辅助监督166固定测试｜已完成

**一句话：** 训练时粗细尺度一起练习，部署只交主输出；用270检查实际效果。
**出发点：** 200轮后验证仅较旧组合166高0.001913且配对区间跨0，用户要求固定测一次。
**具体做法：** 唯一研究改动是训练时固定0.25/0.125辅助头；单T1ce、原972/108/270、预训练起点、原生预处理、主模型与阈值固定；测试166一次整脑前向、去填充评分，没有滑窗或融合。
**结果：** native grid的270扫描/119患者Scan {result['scan_dice']:.6f}、Subject {result['subject_dice']:.6f}、Point {result['point_hits']}/270；对旧组合166差{primary['scan_delta']:+.6f}、患者配对95%CI[{ci[0]:+.6f},{ci[1]:+.6f}]。原验证门未过，test-tuned开发/复用内部探索，无新outer/confirmation。
**是否支持直觉：** {statement}；不否定一般方法，也不将测试改写成验证过门。
**代价/下一步：** 新训练0轮；单A100推理{result['runtime']['inference_seconds']:.1f}秒、峰值{result['runtime']['peak_allocated_bytes']/2**30:.2f}GiB，约7.03GiB临时缓存/18MiB模型。当前结束，不自动重测、调权或改部署。[核对材料]({REL}/RESULT.md)。'''
    block(ROOT / 'AFLOW_BRATS_METHOD_CATALOG.md', MARK, card)
    # Existing training-stage summaries remain interpretable after the new access.
    for name in ['AFLOW_BRATS_HANDOFF_CURRENT.md', 'AFLOW_BRATS_WEEKLY_REPORT.md', 'AFLOW_BRATS_RESULTS.md', 'AFLOW_BRATS_METHOD_CATALOG.md']:
        path = ROOT / name; value = path.read_text()
        pattern = r'<!-- MULTISCALE_RESUME_20260929 -->.*?<!-- END_MULTISCALE_RESUME_20260929 -->'
        match = re.search(pattern, value, flags=re.S)
        if match:
            old = match.group(0)
            old = old.replace('本组无新270/outer/confirmation访问', '训练阶段无新270/outer/confirmation访问；后续用户授权166固定270测试已完成，见最新评测条目')
            old = old.replace('无新270模型评分或外部确认', '训练阶段无新270模型评分；后续166固定270评测已完成，无新外部确认，见最新条目')
            old = old.replace('无本组270新模型访问', '训练阶段无270新模型访问；后续166固定270评测已完成，见最新条目')
            old = old.replace('不自动续训/扩组/进入270', '不自动续训/扩组/重复测试；最新授权166测试见上文')
            old = old.replace('不自动搜索权重、续训或进入270', '不自动搜索权重、续训或重复测试')
            old = old.replace('只有训练/108验证，既往270信息使本组为test-informed；无新270/outer/confirmation',
                              '该阶段只有训练/108验证，既往270信息使本组为test-informed；随后用户授权166固定270测试已完成，无新outer/confirmation')
            path.write_text(value[:match.start()] + old + value[match.end():])
    agents = ROOT / 'AGENTS.md'; value = agents.read_text()
    title = '## 2026-09-29 多尺度166一次270测试：用户最新授权'
    a = value.index(title); b = value.find('\n## ', a + len(title)); b = len(value) if b < 0 else b
    new = title + '\n\n- **本次终态：** ' + brief.replace('**2026-09-29多尺度166固定270评测完成：** ', '') + '\n'
    new += '- 唯一166模型SHA0645c5e7…；270载荷SHA与历史完全一致，135批每例恰一次/主输出全脑前向；旧参考只复用报告。本组训练FINAL_RESULT中的false是历史阶段事实，以独立测试及TEST_ACCESS_ADDENDUM为当前访问范围。不自动换epoch/阈值/融合或重测。\n'
    value = value[:a] + new + value[b:]
    old_title = '## 2026-09-29 多尺度辅助监督50轮续训到200轮：用户最新授权'
    a = value.index(old_title); b = value.find('\n## ', a + len(old_title)); b = len(value) if b < 0 else b
    old_stage = value[a:b].replace('训练/验证完成、无本组270/outer/confirmation新模型访问，不自动续训/扩组/测试',
                                '训练/验证阶段完成时无270/outer/confirmation新模型访问；后续唯一166的用户授权270测试已完成，见最新条目，不自动续训/扩组/重测')
    agents.write_text(value[:a] + old_stage + value[b:])
    addendum = {'utc': now(), 'selected_epoch': 166, 'checkpoint_sha256': CHECKPOINT_SHA,
                'new_test_model_access': True, 'test_scans': 270, 'test_subjects': 119,
                'test_tuned_development': True, 'current_test_retuning': False,
                'new_outer_confirmation_access': False, 'result_path': REL + '/RESULT.json',
                'result_sha256': sha(R / 'RESULT.json'), 'model_preaccess_sha256': sha(R / 'MODEL_PREACCESS.json'),
                'scope': 'original training-phase false flags/reports remain unchanged; fixed166 was subsequently user-authorized and tested once'}
    write(TRAIN_RECORD / 'TEST_ACCESS_ADDENDUM.json', addendum)
    audit(result)
    checks = []
    for script in ['verify_aflow_brats_method_catalog.py', 'verify_aflow_brats_leaderboard.py',
                   'verify_aflow_brats_doc_layout.py', 'verify_aflow_root_artifacts.py']:
        proc = subprocess.run([sys.executable, 'scripts/' + script], cwd=ROOT, capture_output=True, text=True)
        checks.append({'script': script, 'returncode': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr})
    write(R / 'DOC_CHECKS.json', checks)
    if any(c['returncode'] for c in checks):
        raise RuntimeError('required documentation checks failed')
    archive_bytes = sum(p.stat().st_size for p in R.rglob('*') if p.is_file())
    if archive_bytes > 8 * (1 << 30) or shutil.disk_usage(ROOT).free < 20 * (1 << 30):
        raise RuntimeError('persistent storage budget failed')
    write(R / 'PUBLICATION_FREEZE.json', {'utc': now(), 'files': {str((ROOT / n).relative_to(ROOT)): sha(ROOT / n) for n in DOCS},
          'publisher_sha256': sha(Path(__file__)), 'result_md_sha256': sha(R / 'RESULT.md'),
          'archive_bytes': archive_bytes, 'persistent_free_bytes': shutil.disk_usage(ROOT).free})
    write(R / 'DONE.json', {'utc': now(), 'status': 'complete_test270_archived',
          'result_sha256': sha(R / 'RESULT.json'), 'all_four_validators_passed': True,
          'completion_audit_sha256': sha(R / 'COMPLETION_AUDIT.json'), 'new_test_model_access': True})
    status('complete', scan_dice=result['scan_dice'], subject_dice=result['subject_dice'],
           point_hits=result['point_hits'], new_test_model_access=True, reference_model_forwards=0,
           all_four_validators_passed=True, no_automatic_further_work=True)
    print(brief)


if __name__ == '__main__':
    main()
