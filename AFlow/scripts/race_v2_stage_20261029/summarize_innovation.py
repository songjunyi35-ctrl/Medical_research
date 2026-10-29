"""Audit grouped108 selection and summarize both finite inference studies."""
import datetime

from .select_final_innovation import IA, IB, A, B, RECORD, read, evidence, audit_innovations
from .audit_validation import aggregate, paired
from .governance import atomic_json


def simple_key(records):
    summary = aggregate(records)
    return summary['scan_dice'], summary['subject_dice'], summary['point_hits']


def main():
    audit_innovations()
    ra = read(IA / 'RAW_VALIDATION108.json')['per_configuration']
    rb = read(IB / 'RAW_VALIDATION.json')['per_configuration']
    result_a, result_b = read(IA / 'RESULT.json'), read(IB / 'RESULT.json')
    base = ra['equal8_logit_t05']
    fold_map = read(RECORD / 'STATISTICS_FREEZE.json')['fold_assignment']
    a_order = ['equal8_logit_t05'] + [recipe['id'] for recipe in read(IA / 'PLAN.json')['recipes']]
    b_recipes = read(IB / 'PLAN.json')['configurations']
    held_a, held_b = [None] * 108, [None] * 108
    checks = []
    for fold in range(5):
        development = [i for i, row in enumerate(base) if fold_map[row['subject_id']] != fold]
        held = [i for i, row in enumerate(base) if fold_map[row['subject_id']] == fold]
        winner_a = max(a_order, key=lambda name: simple_key([ra[name][i] for i in development]))
        winner_b = min(b_recipes, key=lambda recipe: (
            -simple_key([rb[recipe['id']][i] for i in development])[0],
            -simple_key([rb[recipe['id']][i] for i in development])[1], recipe['id']))['id']
        assert result_a['patient_fivefold'][fold]['selected_id'] == winner_a
        assert result_b['patient_folds'][fold]['selected']['id'] == winner_b
        for i in held:
            held_a[i], held_b[i] = ra[winner_a][i], rb[winner_b][i]
        checks.append(dict(fold=fold, heldout_scans=len(held), a_selected=winner_a, b_selected=winner_b,
                           heldout_used_for_selection=False))
    best_a = ra[result_a['best']['id']]
    best_b = rb[result_b['best']['recipe']['id']]
    diagonal = max((item for item in result_a['ranking'] if '_diagonal_' in item['id']),
                   key=lambda item: item['metrics']['scan_dice'])
    summary = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), state='complete',
        cohort=dict(scans=108, subjects=50), source_reports=[evidence(IA / 'RESULT.json'), evidence(IB / 'RESULT.json')],
        arithmetic_audit=evidence(RECORD / 'INNOVATION108_ARITHMETIC_AUDIT.json'),
        single=dict(recipe_id=result_a['best']['id'], metrics=aggregate(best_a),
                    paired_vs_equal8=paired(best_a, base), heldout=aggregate(held_a),
                    heldout_paired_vs_equal8=paired(held_a, base),
                    full_minus_best_diagonal=paired(best_a, ra[diagonal['id']])),
        fusion=dict(recipe=result_b['best']['recipe'], metrics=aggregate(best_b),
                    paired_vs_equal8=paired(best_b, base),
                    paired_vs_old_fixed_fusion=paired(best_b, rb['old_best_fixed075']),
                    heldout=aggregate(held_b), heldout_paired_vs_equal8=paired(held_b, base),
                    heldout_paired_vs_old_fixed_fusion=paired(held_b, rb['old_best_fixed075'])),
        independent_fourfold_selection_verified=checks,
        costs=dict(new_neural_forwards=0, single_deployment_forwards=8, fusion_deployment_forwards=16,
                   a_gpu_event_seconds=result_a['gpu_algorithm_event_seconds'], a_wall_seconds=result_a['wall_seconds'],
                   a_peak_reserved_gib=result_a['peak_reserved_gib'], b_wall_seconds=result_b['seconds'],
                   b_peak_reserved_gib=result_b['peak_reserved_gib']),
        support='Both mean gains are small and paired intervals cross zero; no stable gain established',
        inherited_test_tuned_history=True, fresh_independent_confirmation=False, new270_access=False,
        training_scope_unchanged=True)
    atomic_json(RECORD / 'INFERENCE_INNOVATION108_RESULT.json', summary)
    a, b = summary['single'], summary['fusion']
    lines = ['# 两项推理创新的108验证结果', '',
        '固定108扫描／50患者、单T1ce、原生1mm完整裁剪评分。两worker分工保持：1154259原80轮训练继续；1154121 GPU0／GPU1分别完成两项事前有限配置实验，未增加网络前向。', '',
        '| 方法 | 配置数 | Scan | Subject | Point | 部署前向／例 |', '|---|---:|---:|---:|---:|---:|',
        f"| RACE局部相关分歧 |16|{a['metrics']['scan_dice']:.9f}|{a['metrics']['subject_dice']:.9f}|{a['metrics']['point_hits']}/108|8|",
        f"| 双模型配对协方差收缩 |12|{b['metrics']['scan_dice']:.9f}|{b['metrics']['subject_dice']:.9f}|{b['metrics']['point_hits']}/108|16|", '',
        f"A对最强equal8差{a['paired_vs_equal8']['scan_delta']:+.9f}，患者配对95%区间{a['paired_vs_equal8']['scan_ci95']}；改善54／变差54。full对最强diagonal仅{a['full_minus_best_diagonal']['scan_delta']:+.9f}，不能把增益归因于相关矩阵。", '',
        f"B对旧0.75固定融合差{b['paired_vs_old_fixed_fusion']['scan_delta']:+.9f}，患者配对95%区间{b['paired_vs_old_fixed_fusion']['scan_ci95']}；对equal8差{b['paired_vs_equal8']['scan_delta']:+.9f}，区间{b['paired_vs_equal8']['scan_ci95']}。未通过预设+0.001及正区间支持门。", '',
        f"总控独立复核每个四折选择与留出分配。合并留出A Scan{a['heldout']['scan_dice']:.9f}、B Scan{b['heldout']['scan_dice']:.9f}，对既定参照区间均跨0；权重与0.75锚点先前已用复用数据选择，分折是条件稳定性检查。所有区间为10000患者整簇bootstrap、seed20261029的选择后描述区间。", '',
        'Point均取实际最终分数argmax；A另列equal8点辅助，两者不能按GT择优。所有算法部署函数不接GT或病例ID；整图GT仅进入评分。相同历史参考的TP／FP／FN／Dice／Point逐例精确复现，总控验证32组报告整数与统计，无临床数组转移。', '',
        f"A算法CUDA事件{summary['costs']['a_gpu_event_seconds']:.3f}秒、108解码墙钟{summary['costs']['a_wall_seconds']:.3f}秒、峰值{summary['costs']['a_peak_reserved_gib']:.3f}GiB；B墙钟{summary['costs']['b_wall_seconds']:.3f}秒、峰值{summary['costs']['b_peak_reserved_gib']:.3f}GiB。该墙钟不包含网络缓存生成，部署成本仍8／16前向。", '',
        '决定：保留全部正负结果，各取一条固定配方迁移至原80轮训练的最终best，复用该模型108视图，不重开16／12网格。最终单模和双模各选一个，再固定270；当前新270评分为0，继承test-tuned历史，非独立盲测。', '',
        '[全部配对区间与变差病例](INFERENCE_INNOVATION108_RESULT.json) · [独立算术审计](INNOVATION108_ARITHMETIC_AUDIT.json) · [A原结果](../2026-10-29_v2_inference_innovation_a/RESULT.md) · [B原结果](../2026-10-29_race_v2_inference_b/innovation_b/RESULT.md)', '']
    (RECORD / 'INFERENCE_INNOVATION108_RESULT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
