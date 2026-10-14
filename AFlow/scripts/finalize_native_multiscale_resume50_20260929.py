"""Finalize the 108-only native multiscale run from committed reports."""
import datetime
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.segresnet_pretrained_native_common import paired

R = ROOT / 'archive/research_records/2026-09-29_segresnet_native_multiscale_resume50'
OUT = ROOT / 'workspace/BraTS/segresnet_t1ce_wt/native_multiscale_resume50_20260929'
OLD = ROOT / 'workspace/BraTS/segresnet_t1ce_wt/pretrained_native_res_20260921/validation_e166.json'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def main():
    stop = read(R / 'TRAIN_STOP.json')
    reload = read(R / 'RELOAD_CHECK.json')
    if reload['status'] != 'passed' or stop['last_complete_epoch'] != 200:
        raise RuntimeError('training not at an approved completion boundary')
    freeze = read(R / 'EXECUTION_FREEZE.json')
    if sha(OLD) != freeze['old_validation_e166_sha256']:
        raise RuntimeError('historical comparison report changed')
    best = int(stop['best_epoch'])
    selected_path = OUT / f'validation_e{best:03d}.json'
    current = read(selected_path)
    reference = read(OLD)
    if current['cases'] != 108 or reference['cases'] != 108:
        raise RuntimeError('wrong validation case count')
    current_rows = current['per_case']; reference_rows = reference['per_case']
    if [c['case_id'] for c in current_rows] != [c['case_id'] for c in reference_rows]:
        raise RuntimeError('validation case order mismatch')
    comparison = paired(reference_rows, current_rows)
    score = current['locator_dsc_mean']
    passed = (score >= .761178640458319 and comparison['scan_delta_ci95'][0] > 0 and
              current['locator_subject_dsc_mean'] >= .7394744179646174 and
              current['automatic_point_hit_rate'] >= 89 / 108)
    output = {'utc': now(), 'status': 'completed200' if stop['last_complete_epoch'] == 200 else 'stopped50',
              'last_complete_epoch': stop['last_complete_epoch'], 'completed_updates': stop['completed_updates'],
              'best_epoch': best, 'best_validation': {k: v for k, v in current.items() if k != 'per_case'},
              'reference_old_combo166': {k: v for k, v in reference.items() if k != 'per_case'},
              'paired_old_to_candidate': comparison, 'promotion_gate_passed': passed,
              'target_scan': .761178640458319, 'point_floor': 89,
              'checkpoint_path': str((OUT / 'best.pth').relative_to(ROOT)),
              'checkpoint_sha256': sha(OUT / 'best.pth'),
              'selected_validation_report_sha256': sha(selected_path),
              'source_pretrain_sha256': freeze['pretrain_sha256'],
              'source_cache_data_audit_sha256': sha(R / 'DATA_AUDIT.json'),
              'reload_check_sha256': sha(R / 'RELOAD_CHECK.json'),
              'test_tuned_development': True, 'new_test_model_forward': False,
              'new_outer_confirmation_access': False,
              'cost_note': '2 GPUs; exact wall and peak resources from supervisor/train logs; auxiliary heads discarded for deployment'}
    output.update(resumed_after_epoch=50, original_completed_updates=6100, new_stage_updates=output['completed_updates']-6100, resume_checkpoint_sha256='93a969ad592849e2d057111c0111837726cb44b9bd126668aa6567faa6dba8a6', adaptive_budget_amendment=True, original_epoch50_gate_failed=True, current_worker=1114913)
    write(R / 'FINAL_RESULT.json', output)
    publish(output)
    return output


def replace_block(path, body):
    path = ROOT / path
    original = path.read_text()
    start = '<!-- MULTISCALE_RESUME_20260929 -->'
    end = '<!-- END_MULTISCALE_RESUME_20260929 -->'
    if original.count(start) != 1 or original.count(end) != 1:
        raise RuntimeError('publication block missing or duplicated: ' + str(path))
    first = original.index(start)
    last = original.index(end, first) + len(end)
    replacement = start + '\n' + body.strip() + '\n' + end
    path.write_text(original[:first] + replacement + original[last:])


def publish(result):
    score = result['best_validation']['locator_dsc_mean']
    subject = result['best_validation']['locator_subject_dsc_mean']
    point = round(result['best_validation']['automatic_point_hit_rate'] * 108)
    delta = result['paired_old_to_candidate']['scan_delta']
    ci = result['paired_old_to_candidate']['scan_delta_ci95']
    stop = read(R / 'TRAIN_STOP.json')
    stage = '完成200轮' if result['last_complete_epoch'] == 200 else '按50轮规则停止'
    support = ('验证数值通过冻结提升门；仍是复用验证与历史test-informed开发，不是独立泛化。'
               if result['promotion_gate_passed'] else
               '未通过冻结提升门；只评价当前权重、配方和单次运行，不证明辅助监督一般无效。')
    details = (f'108扫描/50患者原生1mm裁剪栅格：选第{result["best_epoch"]}轮，'
               f'Scan {score:.6f}、Subject {subject:.6f}、Point {point}/108；'
               f'对旧公开预训练＋原生组合166同格Scan差{delta:+.6f}，'
               f'50患者配对95%CI[{ci[0]:+.6f},{ci[1]:+.6f}]。')
    report = f'''# 原生SegResNet多尺度辅助监督结果

状态：{stage}，累计{result['completed_updates']}次更新，其中原50轮6100更新、新增150轮18300更新；从完整第50轮模型/辅助头/AdamW状态继续，原停止记录保留。用户看过50轮验证后追加预算，此次为自适应预算修订；本组只训练原972扫描/444患者，以固定108/50验证选择。**无本候选新270模型推理、outer或confirmation。** 此方向在已有270揭示后设计，标记test-informed，不能称独立盲测。

## 结果

{details}

冻结门：Scan至少0.761179、配对区间下界>0、Subject不少于0.739474、Point不少于89/108；实际{'通过' if result['promotion_gate_passed'] else '未通过'}。{support}

旧固定193仍是项目比较起点；本组直接单因素参考是相同公开预训练初始化与原生输入的旧组合166。此前最高270定向融合0.775891不改写为本组分数。

## 方法与代价

单T1ce经同N4、图像自身脑区裁剪和非零强度标准化，保持1mm原生裁剪并补零到160×208×160；每轮972各一次、双A100每卡4/global8/122更新。主SegResNet用原Dice+0.1二元交叉熵，训练时半尺度/四分之一尺度辅助损失权重0.25/0.125。部署只加载主模型，仍一次整脑前向；辅助头不进入部署权重。

旧worker1109780已不在活动清单，本阶段在worker1114913两张A10080GB上续训；每卡4/global8及原学习率不变，硬件迁移可能改变浮点轨迹，不承诺逐位连续。新worker/tmp原生训练/验证缓存与旧载荷SHA一致；原50轮主干、辅助头和AdamW步计数6100全部恢复，原最佳49轮参与完整1–200轮选择。旧50轮规则未通过的事实与原报告、快照完整保留。

所报显存峰值、墙钟/恢复次数以[训练日志](../../../workspace/BraTS/segresnet_t1ce_wt/native_multiscale_resume50_20260929/TRAIN.jsonl)、[GPU合成](GPU_SMOKE.json)、[监督收据](TRAIN_STOP.json)为准。没有为了追分调辅助头权重、训练轮数上限、阈值或模型尺寸。

## 证据和访问

[机器结果](FINAL_RESULT.json) · [逐例108原始报告](../../../workspace/BraTS/segresnet_t1ce_wt/native_multiscale_resume50_20260929/validation_e{result['best_epoch']:03d}.json) · [最佳主模型](../../../workspace/BraTS/segresnet_t1ce_wt/native_multiscale_resume50_20260929/best.pth) · [重载一致](RELOAD_CHECK.json) · [数据缓存核验](DATA_AUDIT.json) · [方案](IDEA.md) · [执行冻结](EXECUTION_FREEZE.json) · [历史对照](../../../workspace/BraTS/segresnet_t1ce_wt/pretrained_native_res_20260921/validation_e166.json)。
'''
    write_path = R / 'RESULT.md'
    write_path.write_text(report)
    replace_block('AFLOW_BRATS_HANDOFF_CURRENT.md',
        f'**2026-09-29多尺度辅助监督续训终态：** {stage}、{result["completed_updates"]}更新。{details} 冻结提升门{"通过" if result["promotion_gate_passed"] else "未通过"}；本组无新270/outer/confirmation访问、非独立泛化。旧数据处理审视和病灶均衡选块为其他已关闭批次。[完整报告](archive/research_records/2026-09-29_segresnet_native_multiscale_resume50/RESULT.md)。')
    replace_block('AFLOW_BRATS_METHOD_CATALOG.md',
        f'''## 原始分辨率SegResNet多尺度辅助监督续训｜{stage}

**一句话：** 从第50轮完整状态继续到总200轮，观察后半程是否有收益。 **出发点：** 前50最佳较旧同预算+0.002048，原0.003门未过；用户看过结果后追加预算。
**具体做法：** 相同公开预训练150轮初始化和原972单T1ce/原生1mm整脑配方，仅加训练时半/四分之一尺度软占比辅助头，固定权重0.25/0.125；验证/部署只用主输出，单模型一次前向。
**结果：** {stage}、{result['completed_updates']}更新；{details} 无本组270新模型访问。
**是否支持直觉：** {support}
**代价/下一步：** 双A100与约26.8GiB worker临时原生缓存；辅助头只增加训练开销，不增加部署前向。本组完成，不自动搜索权重、续训或进入270。[最终报告](archive/research_records/2026-09-29_segresnet_native_multiscale_resume50/RESULT.md)。''')
    replace_block('AFLOW_BRATS_RESULTS.md',
        f'2026-09-29续训终态：原生SegResNet训练时多尺度辅助监督{stage}、{result["completed_updates"]}更新。{details} 冻结门{"通过" if result["promotion_gate_passed"] else "未通过"}，{support}只有训练/108验证，既往270信息使本组为test-informed；无新270/outer/confirmation。数据缓存、模型快照和工程恢复见[归档](archive/research_records/2026-09-29_segresnet_native_multiscale_resume50/RESULT.md)。')
    replace_block('AFLOW_BRATS_WEEKLY_REPORT.md',
        f'**2026-09-29多尺度辅助监督续训终态：** {stage}；{details} 冻结提升门{"通过" if result["promotion_gate_passed"] else "未通过"}。训练/108验证归档，无新270模型评分或外部确认；最高旧test-tuned融合仍0.775891。[报告](archive/research_records/2026-09-29_segresnet_native_multiscale_resume50/RESULT.md)。')
    agents = ROOT / 'AGENTS.md'
    content = agents.read_text()
    heading = '## 2026-09-29 多尺度辅助监督50轮续训到200轮：用户最新授权\n'
    terminal = f'- **本轮终态：{stage}、{result["completed_updates"]}更新，选{result["best_epoch"]}轮108 Scan{score:.6f}。** 同格旧166差{delta:+.6f}、配对CI[{ci[0]:+.6f},{ci[1]:+.6f}]，冻结提升门{"通过" if result["promotion_gate_passed"] else "未通过"}。训练/验证完成、无本组270/outer/confirmation新模型访问，不自动续训/扩组/测试；见本轮RESULT和PIPELINE_STATUS。\n\n'
    if content.count(heading) != 1:
        raise RuntimeError('AGENTS.md target heading missing')
    if '本轮终态：'+stage not in content:
        agents.write_text(content.replace(heading, heading + '\n' + terminal, 1))
    checks = ['verify_aflow_brats_method_catalog.py', 'verify_aflow_brats_leaderboard.py',
              'verify_aflow_brats_doc_layout.py', 'verify_aflow_root_artifacts.py']
    env = os.environ.copy();env['PYTHONDONTWRITEBYTECODE'] = '1'
    for script in checks:
        command = [str(ROOT.parent / 'sammed3d-venv/bin/python'), str(ROOT / 'scripts' / script)]
        complete = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        if complete.returncode:
            raise RuntimeError(script + ': ' + complete.stdout + complete.stderr)
    write(R / 'COMPLETION_AUDIT.json', {'utc': now(), 'status': 'complete',
          'last_complete_epoch': result['last_complete_epoch'], 'completed_updates': result['completed_updates'],
          'promotion_gate_passed': result['promotion_gate_passed'], 'all_four_checks_passed': True,
          'report_sha256': sha(R / 'RESULT.md'), 'machine_result_sha256': sha(R / 'FINAL_RESULT.json'),
          'checkpoint_sha256': result['checkpoint_sha256'], 'new_test_model_forward': False})


if __name__ == '__main__':
    value = main()
    print(json.dumps({'status': value['status'], 'best_epoch': value['best_epoch'],
                      'scan': value['best_validation']['locator_dsc_mean'],
                      'promotion_gate_passed': value['promotion_gate_passed']}))
