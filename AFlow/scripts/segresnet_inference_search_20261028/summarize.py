"""Archive full candidate evidence, selection-aware caveats and a usable decoder."""
import csv
import hashlib
import subprocess
import numpy as np
from scipy.ndimage import gaussian_filter
from scripts.segresnet_inference_search_20261028.common import *
from scripts.segresnet_inference_search_20261028.operations import component_mask, volume_decode


def summarize(values, hits, rows):
    subjects = sorted({d['subject_id'] for d in rows})
    groups = [np.array([i for i, d in enumerate(rows) if d['subject_id'] == s]) for s in subjects]
    sums = np.array([values[g].sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    means = sums / counts
    indices = np.random.default_rng(20261028).integers(0, len(groups), size=(10000, len(groups)))
    scan = sums[indices].sum(1) / counts[indices].sum(1)
    subject = means[indices].mean(1)
    return dict(scan_dice=float(values.mean()), subject_dice=float(means.mean()), point_hits=int(hits.sum()),
                scans=108, subjects=50, scan_ci95=np.quantile(scan, [.025, .975]).tolist(),
                subject_ci95=np.quantile(subject, [.025, .975]).tolist(),
                bootstrap='10000 patient-cluster bootstrap seed20261028; selected/reused descriptive')


def replace_block(path, marker, content):
    path = ROOT / path
    text = path.read_text()
    begin, end = f'<!-- {marker} -->', f'<!-- END_{marker} -->'
    block = begin + '\n' + content.strip() + '\n' + end
    if begin in text:
        a, b = text.index(begin), text.index(end) + len(end)
        text = text[:a] + block + text[b:]
    else:
        first = text.index('\n') + 1
        text = text[:first] + '\n' + block + '\n' + text[first:]
    path.write_text(text)


def main():
    plan = verify()
    rows, geometries, _, labels = dataset()
    c1 = read(R / 'STAGE1_CANDIDATES.json')
    c2 = read(R / 'STAGE2_CANDIDATES.json')
    candidates = c1 + c2
    d1 = np.load(R / 'stage1/dice.npy')
    raw2 = np.load(R / 'stage2/counts.npy')
    dice = np.concatenate([d1, np.stack([raw2[c['local_recipe_index'], c['score_index'], :, 0] for c in c2])])
    h1 = np.load(R / 'stage1/point_hits.npy')
    h2 = np.load(R / 'stage2/point_hits.npy')
    hits = np.concatenate([np.stack([h1[c['recipe_index']] for c in c1]),
                           np.stack([h2[c['local_recipe_index'], c['point_sigma_index']] for c in c2])])
    subjects = sorted({d['subject_id'] for d in rows})
    groups = [np.array([i for i, d in enumerate(rows) if d['subject_id'] == s]) for s in subjects]
    means = dice.mean(1, dtype=np.float64)
    subjectmeans = np.stack([dice[:, g].mean(1, dtype=np.float64) for g in groups], 1).mean(1)
    for i, c in enumerate(candidates):
        c.update(scan_dice=float(means[i]), subject_dice=float(subjectmeans[i]), point_hits=int(hits[i].sum()), index=i)
    ranking = sorted(range(len(candidates)), key=lambda i: (-means[i], -subjectmeans[i],
                     candidates[i]['native_voxel_equivalent'], candidates[i]['id']))
    winner = ranking[0]
    baseline = next(i for i, c in enumerate(candidates) if c['stage'] == 1 and c['recipe_index'] == 0 and abs(c['threshold'] - .5) < 1e-8)
    best = candidates[winner]
    recipe = plan['recipes'][best['recipe_index']]
    best_metrics = summarize(dice[winner], hits[winner], rows)
    base_metrics = summarize(dice[baseline], hits[baseline], rows)
    difference = summarize(dice[winner] - dice[baseline], hits[winner], rows)
    difference['point_difference'] = int(hits[winner].sum()) - int(hits[baseline].sum())
    gate = bool(difference['scan_dice'] >= .003 and difference['scan_ci95'][0] > 0 and
                difference['subject_dice'] >= 0 and difference['point_difference'] >= -1)
    # Patient-held-out configuration selection across the observed fixed candidate space.
    # Stage2 shortlist was selected on all108, so this is explicitly not nested validation.
    subject_order = sorted(subjects, key=lambda s: hashlib.sha256(('20261028:' + s).encode()).hexdigest())
    folds = {s: i % 5 for i, s in enumerate(subject_order)}
    oof = np.zeros(108)
    oof_hits = np.zeros(108, bool)
    selections = []
    for fold in range(5):
        test = np.array([i for i, d in enumerate(rows) if folds[d['subject_id']] == fold])
        train = np.array([i for i, d in enumerate(rows) if folds[d['subject_id']] != fold])
        train_means = dice[:, train].mean(1, dtype=np.float64)
        selected = min(range(len(candidates)), key=lambda i: (-train_means[i], candidates[i]['native_voxel_equivalent'], candidates[i]['id']))
        oof[test] = dice[selected, test]
        oof_hits[test] = hits[selected, test]
        selections.append(dict(fold=fold, train_scans=len(train), heldout_scans=len(test), candidate=candidates[selected],
                               heldout_case_ids=[rows[i]['case_id'] for i in test]))
    oof_metrics = summarize(oof, oof_hits, rows)
    oof_difference = summarize(oof - dice[baseline], oof_hits, rows)
    write(R / 'PATIENT_FOLD_SELECTION.json', dict(at=now(), selections=selections, metrics=oof_metrics,
          difference=oof_difference, caveat='Stage2 shortlist and checkpoint used full validation; not nested or independent',
          per_case=[dict(case_id=d['case_id'], subject_id=d['subject_id'], fold=folds[d['subject_id']],
                         locator_dsc=float(oof[i]), baseline=float(dice[baseline, i])) for i, d in enumerate(rows)]))
    # Archive all scalar outcomes in a compact, auditable table and per-case numeric matrices.
    write(R / 'ALL_CANDIDATES.json', [candidates[i] for i in ranking])
    np.savez_compressed(R / 'ALL_PER_CASE.npz', dice=dice, point_hits=hits,
                        case_ids=np.array([d['case_id'] for d in rows]), candidate_ids=np.array([c['id'] for c in candidates]))
    with (R / 'ALL_CANDIDATES.tsv').open('w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['rank', 'id', 'recipe', 'scan_dice', 'subject_dice', 'point_hits', 'case_forwards', 'voxel_equivalent', 'sigma', 'threshold', 'rule'])
        for rank, i in enumerate(ranking, 1):
            c = candidates[i]
            writer.writerow([rank, c['id'], plan['recipes'][c['recipe_index']]['id'], c['scan_dice'], c['subject_dice'],
                c['point_hits'], c['case_forwards'], c['native_voxel_equivalent'], c['sigma'], c['threshold'], c['rule']['id']])
    # One frozen deployable choice; parameters were selected on validation only.
    write(R / 'DEPLOYMENT_RECIPE.json', dict(at=now(), checkpoint=str(CHECKPOINT.relative_to(ROOT)),
          checkpoint_sha256=CHECKPOINT_SHA, epoch=171, width=32, candidate=best, recipe=recipe,
          atoms=[a for a in plan['atoms'] if a['id'] in recipe['atoms']],
          grid='native1mm crop excludes padding', validation_canvas=CANVAS,
          training=False, test_access=False, selected_on_reused_validation=True))
    p1 = np.load(R / 'stage1/points.npy')
    p2 = np.load(R / 'stage2/points.npy')
    cases = []
    for i, row in enumerate(rows):
        cid = row['case_id']
        if best['stage'] == 2:
            probability = np.load(TMP / 'top_probabilities' / str(best['local_recipe_index']) / (cid + '.npy'))
            point = p2[best['local_recipe_index'], best['point_sigma_index'], i].tolist()
        else:
            import torch
            from scripts.segresnet_inference_search_20261028.operations import fuse
            from scripts.segresnet_inference_search_20261028.atoms import array_path
            with torch.inference_mode():
                arrays = {name: torch.from_numpy(np.load(array_path(name, cid))).cuda() for name in recipe['atoms']}
                probability = fuse(arrays, recipe).cpu().numpy()
            point = p1[best['recipe_index'], i].tolist()
        if best['sigma']:
            probability = gaussian_filter(probability, sigma=best['sigma'], mode='nearest')
        if best['rule']['kind'] == 'volume_decode':
            mask, _ = volume_decode(probability, best['rule']['temperature'])
        else:
            mask = component_mask(probability, best['threshold'], best['rule'])
        target = labels[i][slices(geometries[cid])] > 0
        tp = int((mask & target).sum())
        fp = int((mask & ~target).sum())
        fn = int((~mask & target).sum())
        den = 2 * tp + fp + fn
        direct = float(np.float32(2 * tp) / np.float32(den)) if den else 1.
        assert abs(direct - dice[winner, i]) < 1e-7
        assert bool(target[tuple(point)]) == bool(hits[winner, i])
        path = R / 'selected_predictions' / (cid + '.npz')
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, bits=np.packbits(mask.ravel()), shape=np.array(mask.shape), point=np.array(point))
        cases.append(dict(case_id=cid, subject_id=row['subject_id'], locator_dsc=direct,
                          automatic_point=point, automatic_point_hits_gt=bool(hits[winner, i]),
                          tp=tp, fp=fp, fn=fn, baseline_dice=float(dice[baseline, i]), prediction_sha256=sha(path)))
    write(R / 'RAW_SELECTED.json', dict(at=now(), metrics=best_metrics, per_case=cases,
          checkpoint_sha256=CHECKPOINT_SHA, recipe_sha256=sha(R / 'DEPLOYMENT_RECIPE.json'), test_access=False))
    pareto = []
    highest = -1.
    for i in sorted(range(len(candidates)), key=lambda i: (candidates[i]['native_voxel_equivalent'], -means[i])):
        if means[i] > highest + 1e-12:
            pareto.append(candidates[i])
            highest = means[i]
    result = dict(at=now(), state='complete', worker=1152682, checkpoint_sha256=CHECKPOINT_SHA,
        configurations=len(candidates), stage1_configurations=len(c1), stage2_configurations=len(c2),
        baseline=base_metrics, best=best_metrics, selected=best, recipe=recipe,
        difference=difference, descriptive_utility_gate=gate, patient_fold_metrics=oof_metrics,
        patient_fold_difference=oof_difference, pareto=pareto,
        caveat='Reused108/50 validation search; selected confidence intervals optimistic; no new test access',
        atoms=len(plan['atoms']), training=False, test_access=False,
        wall_seconds=time.time() - read(R / 'RUN_STARTED.json')['unix'],
        peak_reserved_gib=max(read(R / 'atoms' / (a['id'] + '.json'))['peak_reserved_gib'] for a in plan['atoms']))
    write(R / 'RESULT.json', result)
    text = f'''# 固定w32第171轮推理穷举：完成

固定同一32通道171轮权重，新增训练0。原108扫描/50患者、单T1ce、原生格排除填充；{len(plan['atoms'])}种原子推理、第一阶段{len(c1)}种配置、第二阶段{len(c2)}种配置完整评分。无新270/outer/confirmation或外部评测。

| 108验证方案 | Scan Dice | Subject Dice | Point |
|---|---:|---:|---:|
| 原始推理重算 | {base_metrics['scan_dice']:.9f} | {base_metrics['subject_dice']:.9f} | {base_metrics['point_hits']}/108 |
| 搜索最高点估计 | {best_metrics['scan_dice']:.9f} | {best_metrics['subject_dice']:.9f} | {best_metrics['point_hits']}/108 |
| 按患者分折选配置 | {oof_metrics['scan_dice']:.9f} | {oof_metrics['subject_dice']:.9f} | {oof_metrics['point_hits']}/108 |

最高方案：`{recipe['id']}`，概率平滑sigma={best['sigma']}，阈值={best['threshold']}，规则`{best['rule']['id']}`。完整可执行配方见[部署配方](DEPLOYMENT_RECIPE.json)。Scan差{difference['scan_dice']:+.9f}，患者整簇配对95%区间{difference['scan_ci95']}；描述性效用门={gate}。

这是大量搜索后的复用验证最高值，不是独立确认。按患者分折结果仍受全验证checkpoint选择及第二阶段top6筛选影响，不是完全嵌套交叉验证。验证增量不能加在历史270的0.773228上冒充新测试结果。

部署每例{best['case_forwards']}次模型前向、约{best['native_voxel_equivalent']:.3f}个native体素前向等价量；该等价量不代表FLOPs或实测延迟。GPU峰值reserved约{result['peak_reserved_gib']:.2f}GiB；本轮总墙钟{result['wall_seconds']/60:.1f}分钟。各原子时间含保存输出，详见atoms收据。输入/中间浮点输出只在worker/tmp，选定掩膜与全部指标持久保存；未增加第二模型。

[全部配置正负结果](ALL_CANDIDATES.tsv) · [原始逐例指标](RAW_SELECTED.json) · [所有配置逐例矩阵](ALL_PER_CASE.npz) · [患者分折选择](PATIENT_FOLD_SELECTION.json) · [原始推理一致性门](BASELINE_GATE.json) · [机器结果](RESULT.json)

有限空间已完成；不自动增加视图/阈值或开始新测试。是否支持提分直觉以当前配方的描述性差异和效用门为限，不宣称所有推理方法的理论上限。
'''
    (R / 'RESULT.md').write_text(text)
    relative = 'archive/research_records/2026-10-28_segresnet_inference_search'
    outcome = f"108验证原始{base_metrics['scan_dice']:.6f}→搜索最高{best_metrics['scan_dice']:.6f}（差{difference['scan_dice']:+.6f}），Subject{best_metrics['subject_dice']:.6f}，Point{best_metrics['point_hits']}/108；患者分折{oof_metrics['scan_dice']:.6f}。"
    replace_block('AFLOW_BRATS_METHOD_CATALOG.md', 'INFERENCE_SEARCH_20261028', f'''## 2026-10-28：固定w32权重推理穷举｜complete

**一句话：** 同一模型换视角重复看图，再用有限后处理规则改善输出。
**出发点：** 固定171轮108验证0.753276，检验不训练能否进一步改善。
**具体做法：** {len(plan['atoms'])}原子推理、{len(candidates)}完整配置，验证选输出与解码；固定单T1ce、权重、病例与原生评分。
**结果：** {outcome}
**是否支持直觉：** 描述性效用门={gate}，大量搜索后的最高值有选择偏差；无新测试，不称独立泛化。
**代价/下一步：** 部署{best['case_forwards']}次前向、约{best['native_voxel_equivalent']:.2f}native体素等价；训练0，有限搜索结束，不自动扩搜或test。[核对材料]({relative}/RESULT.md)。''')
    replace_block('AFLOW_BRATS_HANDOFF_CURRENT.md', 'INFERENCE_SEARCH_20261028', f'''**2026-10-28固定w32推理搜索完成。** 用户选择32通道171轮唯一权重，worker1152682单A10080GB；{len(candidates)}配置全部原108/50验证，无新训练/270/outer/confirmation。{outcome} 复用验证选择后的探索证据，非独立确认；原RACE双卡训练不受影响。搜索终止，不自动扩搜或测试。[结果]({relative}/RESULT.md)。''')
    replace_block('AFLOW_BRATS_RESULTS.md', 'INFERENCE_SEARCH_20261028', f'''## 2026-10-28：同一SegResNet权重的推理穷举

问题：不训练能否提分。做法：固定w32第171轮，worker1152682穷举{len(candidates)}种镜像/尺度/平移/旋转/强度/滑窗及解码组合，全部108验证。结果：{outcome} 描述性门={gate}，选择偏差保留，无新测试；有限搜索完成不自动扩组。[报告]({relative}/RESULT.md)。''')
    replace_block('AGENTS.md', 'INFERENCE_SEARCH_20261028', f'''## 2026-10-28固定w32第171轮推理搜索已完成

用户授权worker1152682无训练推理穷举，并明确选32通道171轮单模型。{len(candidates)}配置完成原108/50验证；{outcome} 全部复用验证选择后的描述性证据，无新270/outer/confirmation；有限搜索结束，不自动扩搜或测试。独立RACE双卡训练不受本批终态影响。[结果]({relative}/RESULT.md)。''')
    checks = []
    for script in ['verify_aflow_brats_method_catalog.py', 'verify_aflow_brats_leaderboard.py',
                   'verify_aflow_brats_doc_layout.py', 'verify_aflow_root_artifacts.py']:
        process = subprocess.run([str(PY), str(ROOT / 'scripts' / script)], cwd=ROOT,
                                 capture_output=True, text=True)
        checks.append(dict(script=script, exit_code=process.returncode, output=process.stdout, error=process.stderr))
    write(R / 'DOCUMENT_CHECKS.json', checks)
    assert all(d['exit_code'] == 0 for d in checks), 'documentation validation failed'
    files = ['PLAN.json', 'EXECUTION_FREEZE.json', 'DATA_READY.json', 'BASELINE_RAW.json',
             'DEPLOYMENT_RECIPE.json', 'RAW_SELECTED.json', 'RESULT.json', 'RESULT.md', 'ALL_PER_CASE.npz']
    write(R / 'ARCHIVE_SHA256.json', dict(at=now(), checkpoint_sha256=sha(CHECKPOINT),
          files={name: sha(R / name) for name in files}))
    status('complete', configurations=len(candidates), baseline=base_metrics,
           best=best_metrics, difference=difference, descriptive_utility_gate=gate)


if __name__ == '__main__':
    main()
