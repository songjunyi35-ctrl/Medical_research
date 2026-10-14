"""Archive authorization and validation selection, then freeze before any new test pixels."""
import ast
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.multiscale_e166_test270_20260929_common import *


def local_dependencies(initial):
    pending = list(initial); seen = set()
    while pending:
        path = pending.pop()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        if path.suffix != '.py':
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module] + [node.module + '.' + a.name for a in node.names]
            for name in names:
                if name.startswith(('scripts.', 'sammed3d.')):
                    module = ROOT / (name.replace('.', '/') + '.py')
                    package = ROOT / name.replace('.', '/') / '__init__.py'
                    if module.exists():
                        pending.append(module)
                    elif package.exists():
                        pending.append(package)
    return seen


def insert_block(path, mark, body, top=False):
    text = path.read_text(); block = f'<!-- {mark} -->\n{body}\n<!-- END_{mark} -->\n'
    if f'<!-- {mark} -->' in text:
        raise RuntimeError('duplicate preparation block: ' + str(path))
    if top:
        first, rest = text.split('\n', 1); text = first + '\n\n' + block + rest
    else:
        text = text.rstrip() + '\n\n' + block
    path.write_text(text)


def main():
    if R.exists():
        raise RuntimeError('preparation already exists; inspect instead of overwrite')
    if sha(SOURCE) != CHECKPOINT_SHA or shutil.disk_usage(ROOT).free < 20 * (1 << 30):
        raise RuntimeError('selected model identity or20GiB reserve failed')
    train = read(TRAIN_RECORD / 'FINAL_RESULT.json')
    if train['best_epoch'] != 166 or train['checkpoint_sha256'] != CHECKPOINT_SHA or train['promotion_gate_passed']:
        raise RuntimeError('selection identity or historical gate mismatch')
    references()
    R.mkdir(parents=True)
    shutil.copyfile(SOURCE, CHECKPOINT)
    if sha(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError('checkpoint copy changed bytes')
    (R / 'IDEA.md').write_text('''# 原始分辨率多尺度辅助监督166轮：用户授权的一次270评测

状态：先于本候选新测试像素/模型访问冻结。用户最新明确“要不去测评一下吧”，授权本次固定候选的一次测试；不恢复其他关闭实验。真实阶段看PIPELINE_STATUS。

## 1. 现在的问题是什么？

新多尺度辅助监督组已完成总200轮/24400更新，由固定108扫描/50患者验证选第166轮。原生栅格验证Scan0.753092133、Subject0.739668718、Point94/108；相对旧公开预训练＋原生组合166 Scan仅+0.001913493，患者配对95%CI[-0.004999,+0.010082]，未通过原+0.01提升门。用户希望看训练完成后的270表现。

## 2. 一句话直觉

训练时让网络在粗细两种尺度都练习认肿瘤，最后只让主输出交卷；现在用固定测试病例检查这种练习有没有实际收益。

## 3. 为什么选择这个idea？

这是已完成模型的固定评估，不新增训练方法。验证点估计略高但不确定，测试应给出真实效果与患者配对区间，而不能凭验证涨幅宣称有效。多尺度监督机制和官方依据见[原训练七问](../2026-09-24_segresnet_native_multiscale_supervision/IDEA.md)；此前50轮失败与本次最终验证门失败仍保留，不据测试结果改写。训练方向和追加200轮预算受既往test诊断/验证影响，标test-tuned开发；本次固定测试不搜索epoch或参数。

## 4. 具体怎么做？

只使用原270扫描/119患者单T1ce。原始NIfTI→相同N4 shrink2→图像自身非零脑区裁剪→相同非零总体z-score→保留1mm裁剪→中心补零160×208×168→第166轮主SegResNet整脑一次前向→严格raw logit>0→去填充在原始裁剪栅格计算Dice，最高主logit产生自动点。辅助头丢弃。测试画布沿用旧测试已冻结几何，末轴168容纳最大161的裁剪，不把训练160画布截断测试。

先按旧冻结构建函数重现约7.03GiB原生测试缓存；图像与标签SHA必须分别为845ee16e29afd3c84161791ead53870aeb7f9dd06a45cb19d4d42653a6d0842a、044384d4941af5a93deac0dc252ce6138444012f2016a4f839184c983b170803。与旧128³缓存的图像、标签、裁剪几何逐例核对，仅作预处理一致性门。使用CUDA0、bf16 batch2、确定性cuDNN、关闭matmul TF32/开启cuDNN TF32，与旧组合166评测相同。135批覆盖270且每例恰一次，先落盘raw再统计。病例身份、患者隔离均核验。

## 5. 唯一变化与固定项

本次不训练，唯一评测候选是多尺度166快照SHA0645c5e781571ceff2e6d26c62ca52e12e985cf2c1b097f07d7b8a0bc7b48ce2。相对原训练对照，研究变化仅训练时两个固定权重辅助头；部署同主网络。单T1ce、972/108/270清单及患者隔离、N4/脑区框/标准化、1mm栅格、阈值0、一次整脑前向固定。没有八块滑窗、TTA、融合、后处理、GT定位或标签梯度。旧组合166（主要匹配对照）、旧native173和固定纯193同栅格映射的已有逐例报告全部复用，不额外前向参考模型，不跨128³/native栅格相减。

## 6. 怎样算成功、失败与停止？

主指标为270 Scan Dice；同时报119患者等权Subject Dice与Point/270。绝对区间10000次119患者整簇bootstrap seed20260917，候选减参考配对区间10000次seed20260918，主要解释旧组合166的95%区间；另外参考描述性报告，97.5%区间如实列出。区间跨0不称稳定提升。原验证晋级门未过，当前测试是用户授权探索，不是新自动晋级门，不自动改部署。

MODEL_PREACCESS独占创建保护候选一次模型访问；失败不自动重复患者前向。完整RAW已落盘时只可恢复统计/出版；部分journal不足270则记录失败和已访问范围。缓存不一致、CUDA/数值/存储失败即停止并归档，不调阈值/画布追分，不测试其他epoch或新增outer/confirmation。

## 7. 成本与风险是什么？

新增训练时间0。worker1114913两张A10080GB可用，本次固定batch2单卡推理，另一卡不参与候选测试。先做无患者合成形状/阈值/裁剪/严格加载和两卡CUDA检查，记录显存而不猜测实测值。测试缓存约7.03GiB仅worker/tmp；项目新增独立模型约18MiB和报告，新增上限8GiB/剩余至少20GiB。画布输入体素量为128³的2.666015625倍，这是体素量而非实测耗时比。主要风险是复用测试和test-informed开发偏倚、单seed、不确定患者区间；不称独立盲测或外部泛化。旧native173/193历史数值模式不同，配对比较保留此细微限制。

## 8. 最终结果

评测前无新精度数字。完成后见本目录RESULT.md/RESULT.json及访问补记，保留本冻结七问原文。
''')
    write_new(R / 'SELECTION_RECEIPT.json', {'utc': now(), 'epoch': 166, 'checkpoint_sha256': CHECKPOINT_SHA,
              'source_path': str(SOURCE.relative_to(ROOT)), 'source_report_sha256': sha(TRAIN_RECORD / 'FINAL_RESULT.json'),
              'validation_report_sha256': train['selected_validation_report_sha256'],
              'selected_by': 'highest Scan on fixed108 validation, before this new candidate test',
              'completed_epochs': 200, 'completed_updates': 24400, 'validation_gate_passed': False,
              'validation_scan_dice': .7530921331986233, 'validation_subject_dice': .7396687182784081,
              'validation_point_hits': 94, 'no_alternate_epoch_or_tuning': True})
    write_new(R / 'PREACCESS.json', {'utc': now(), 'user_authorization': '要不去测评一下吧',
              'scope': 'one fixed multiscale epoch166 native-grid270/119 test; reused reference reports only',
              'candidate_new_test_forward_at_preparation': False, 'test_tuned_development': True,
              'new_training': False, 'new_outer_confirmation_access': False, 'no_automatic_test_retry': True})
    starts = [Path(__file__), ROOT / 'scripts/multiscale_e166_test270_20260929_common.py',
              ROOT / 'scripts/run_multiscale_e166_test270_20260929.py']
    files = local_dependencies(starts)
    files.update([CHECKPOINT, R / 'IDEA.md', R / 'SELECTION_RECEIPT.json', R / 'PREACCESS.json',
                  TRAIN_RECORD / 'FINAL_RESULT.json', OLD_NATIVE / 'CACHE_RECEIPT.json',
                  OLD_NATIVE / 'RAW_TEST270.json', OLD_COMBO / 'RAW_TEST270.json',
                  CANONICAL_TEST / 'index.json', *MANIFEST.values()])
    write_new(R / 'FREEZE.json', {'utc': now(), 'worker': 1114913,
              'files': {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)},
              'scope': 'code, selected model, identities, canonical geometry, historical cache/reference reports; no new pixels yet',
              'canvas': list(TEST_CANVAS), 'batch': 2, 'precision': 'bf16', 'raw_logit_threshold': 0,
              'main_model_forwards_per_scan': 1, 'reference_model_forwards': 0,
              'bootstrap_samples': 10000, 'absolute_seed': 20260917, 'paired_seed': 20260918})
    status('frozen_awaiting_worker_preflight', new_test_model_access=False)
    rel = str(R.relative_to(ROOT))
    brief = f'**2026-09-29多尺度166固定测试已获授权并冻结：** 用户要求“要不去测评一下吧”，候选由108验证选定；仅一次270扫描/119患者、原生1mm去填充评分，保持阈值0和一次主输出整脑前向。原验证提升门未过，标test-tuned开发/复用内部探索。当前等待节点预检和原生缓存，不称已完成或已有新成绩，无新outer/confirmation。[七问与边界]({rel}/IDEA.md) · [真实阶段]({rel}/PIPELINE_STATUS.json)。'
    for name in ['AFLOW_BRATS_HANDOFF_CURRENT.md', 'AFLOW_BRATS_WEEKLY_REPORT.md']:
        insert_block(ROOT / name, 'MULTISCALE_E166_TEST270_20260929', brief, top=True)
    insert_block(ROOT / 'AFLOW_BRATS_RESULTS.md', 'MULTISCALE_E166_TEST270_20260929',
                 '## 2026-09-29｜用户授权多尺度166固定270测试\n\n' + brief)
    insert_block(ROOT / 'AFLOW_BRATS_METHOD_CATALOG.md', 'MULTISCALE_E166_TEST270_20260929', brief)
    agents = ROOT / 'AGENTS.md'; text = agents.read_text(); title, rest = text.split('\n', 1)
    agents.write_text(title + '\n\n## 2026-09-29 多尺度166一次270测试：用户最新授权\n\n'
                      '- 用户明确“要不去测评一下吧”，授权已完成多尺度200轮组验证选出的唯一166模型一次原270/119测试；覆盖原本组无自动test条款，不恢复其他关闭实验。\n'
                      '- 独立快照SHA0645c5e7…，native1mm裁剪/160×208×168补零、bf16 batch2、raw logit>0、一次主输出整脑前向；旧组合166/native173/纯193映射仅复用逐例报告，无额外参考模型前向。\n'
                      '- 原验证提升门未过，保持test-tuned开发/复用内部探索，无当前测试参数选择、无新outer/confirmation；失败不自动重复模型前向。新归档`' + rel + '/`，真实阶段见其PIPELINE_STATUS。\n' + rest)
    print('frozen files:', len(files), flush=True)


if __name__ == '__main__':
    main()
