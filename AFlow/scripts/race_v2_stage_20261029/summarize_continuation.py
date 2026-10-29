"""Independent108 arithmetic and grouped-selection audit for the next ideas."""
import datetime
from .select_final_continuation import NA, NB, RECORD, read, evidence, audit_continuation
from .audit_validation import aggregate, paired
from .governance import atomic_json


def main():
    audit_continuation()
    ra = read(NA / 'RAW_VALIDATION108.json')['per_configuration']
    rb = read(NB / 'RAW_VALIDATION.json')['per_configuration']
    a, b = read(NA / 'RESULT.json'), read(NB / 'RESULT.json')
    base = ra['equal8_logit_t05']
    foldmap = read(RECORD / 'STATISTICS_FREEZE.json')['fold_assignment']
    a_ids = ['equal8_logit_t05'] + [r['id'] for r in read(NA / 'PLAN.json')['recipes']]
    b_ids = ['old_best_fixed075', 'old_selected_covariance'] + [r['id'] for r in read(NB / 'PLAN.json')['configurations']]
    held_a, held_b, checks = [None]*108, [None]*108, []
    for fold in range(5):
        development = [i for i, row in enumerate(base) if foldmap[row['subject_id']] != fold]
        held = [i for i, row in enumerate(base) if foldmap[row['subject_id']] == fold]
        def a_key(name):
            m = aggregate([ra[name][i] for i in development])
            return m['scan_dice'], m['subject_dice'], m['point_hits']
        def b_key(name):
            m = aggregate([rb[name][i] for i in development])
            return -m['scan_dice'], -m['subject_dice'], name
        wa, wb = max(a_ids, key=a_key), min(b_ids, key=b_key)
        assert a['patient_fivefold'][fold]['selected_id'] == wa
        assert b['patient_folds'][fold]['selected'] == wb
        for i in held:
            held_a[i], held_b[i] = ra[wa][i], rb[wb][i]
        checks.append(dict(fold=fold, single=wa, fusion=wb, heldout_scans=len(held), heldout_not_used_for_choice=True))
    new_a, new_b = ra[a['best_innovation']['id']], rb[b['best_new']['recipe']['id']]
    result = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), state='complete',
        cohort=dict(scans=108, subjects=50), source_reports=[evidence(NA/'RESULT.json'), evidence(NB/'RESULT.json')],
        arithmetic_audit=evidence(RECORD/'CONTINUATION108_ARITHMETIC_AUDIT.json'),
        normalization=dict(id=a['best_innovation']['id'], metrics=aggregate(new_a),
            paired_vs_equal8=paired(new_a, base), heldout_metrics=aggregate(held_a),
            heldout_paired_vs_equal8=paired(held_a, base), all8_below_equal8=True,
            new_neural_forwards=a['new_network_forwards'], network_wall_seconds=a['forward_wall_seconds'],
            network_cuda_event_seconds=a['forward_gpu_event_seconds'], peak_reserved_gib=a['peak_reserved_gib']),
        contralateral=dict(recipe=b['best_new']['recipe'], metrics=aggregate(new_b),
            paired_vs_equal8=paired(new_b, rb['race_equal8']),
            paired_vs_fixed075=paired(new_b, rb['old_best_fixed075']),
            paired_vs_covariance=paired(new_b, rb['old_selected_covariance']),
            heldout_metrics=aggregate(held_b),
            heldout_paired_vs_covariance=paired(held_b, rb['old_selected_covariance']),
            new_neural_forwards=0, decode_wall_seconds=b['seconds'], peak_reserved_gib=b['peak_reserved_gib']),
        grouped_selection_independently_recomputed=checks,
        single_deployment_forwards=8, fusion_deployment_forwards=16,
        original_training_unchanged=True, new270_access=False, inherited_test_tuned_history=True,
        independent_confirmation=False)
    assert all(item['metrics']['scan_dice'] < aggregate(base)['scan_dice'] for item in a['ranking'] if item['id'] != 'equal8_logit_t05')
    atomic_json(RECORD/'CONTINUATION108_RESULT.json', result)
    na, nb = result['normalization'], result['contralateral']
    text = f'''# 持续推理研究：健康脑标准化与对侧图像证据

用户指出推理worker曾在上一批结束后空闲约53分钟。总控恢复两路独立idea，原1154259双卡80轮训练完全保持原执行。仅1154121 GPU0／GPU1运行本轮推理研究。

| 方法 | 新配置 | Scan | Subject | Point | 新网络前向 |
|---|---:|---:|---:|---:|---:|
| seed健康脑归一化，最佳新配方 |8|{na['metrics']['scan_dice']:.9f}|{na['metrics']['subject_dice']:.9f}|{na['metrics']['point_hits']}/108|{na['new_neural_forwards']}|
| 对侧图像证据冲突裁决，最佳新配方 |8|{nb['metrics']['scan_dice']:.9f}|{nb['metrics']['subject_dice']:.9f}|{nb['metrics']['point_hits']}/108|0|

归一化按原图seed一次→排疑健康脑矩→七镜像，部署严格8次；四个新输入组缓存一次供八解码复用。全部八项低于原equal8，最佳差{na['paired_vs_equal8']['scan_delta']:+.9f}，95%患者配对区间{na['paired_vs_equal8']['scan_ci95']}；49改善／59变差，五折全选equal8，结论阴性。真实网络墙钟{na['network_wall_seconds']:.3f}秒、CUDA事件{na['network_cuda_event_seconds']:.3f}秒，峰值{na['peak_reserved_gib']:.3f}GiB。

对侧方案不加网络、不传播邻域标签，只以图像对称区差异和健康参考可信度保守调整双模型冲突权重。最佳对旧固定融合差{nb['paired_vs_fixed075']['scan_delta']:+.9f}，区间{nb['paired_vs_fixed075']['scan_ci95']}；对既有最高cov差{nb['paired_vs_covariance']['scan_delta']:+.9f}，区间{nb['paired_vs_covariance']['scan_ci95']}，未超过cov。分折允许旧参照，合并Scan{nb['heldout_metrics']['scan_dice']:.9f}。单T1ce正向强度差不能解释为已证实的对比增强。

总控独立核对20组完整108整数报告、四个旧参照每例TP／FP／FN／Dice／Point，以及四折选择／留出分配。全crop GT保留，患者整簇bootstrap10000、seed20261029，全部区间选择后描述性，继承test-tuned历史，非独立确认。Point取最终真实score argmax。全部变差病例与成本在机器结果，没有临床数组转移。

两个新增family各取一条已固定配方等待终态best复核，原equal8／局部依赖／固定融合／cov等参照保留；最终仍最多一个单模与一个纯SegResNet双模，部署8／16前向。本轮尚无新270评分，不因阴性结果调整评分或排除病例。下一idea的诊断与构思继续，不使用无意义GPU负载。

[全部统计与变差](CONTINUATION108_RESULT.json) · [独立算术审计](CONTINUATION108_ARITHMETIC_AUDIT.json) · [A完整结果](../2026-10-29_race_v2_inference_normalization_a/RESULT.md) · [B完整结果](../2026-10-29_race_v2_inference_b/image_conflict_b/RESULT.md)
'''
    (RECORD/'CONTINUATION108_RESULT.md').write_text(text)


if __name__ == '__main__':
    main()
