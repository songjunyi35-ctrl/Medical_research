# 固定自动工作流：一次性内部test270确认

状态：六输出完成并独立核验；双模型固定精度—点—成本门通过=False。本面板关闭，不按结果重选视图或模型。

## 动机、方法与控制

三折选择不等于最终部署策略。本阶段只根据开发折winner视图集合取中位代表，并列按成本/ID；规则在开发后、测试前冻结，不使用测试分数。Seed11固定[0,5]两视图；seed12固定[0,5,6]三视图。两个native fit684/313模型权重、epoch168、128网格、bf16/float32、有序均值、阈值0、最高点不变，无新训练/教师/SAM/后处理。

同一270 scans/119 subjects，原始内部测试已反复揭示，不是新盲测、外部确认或公开SOTA。每模型identity/pruned/full8三行全部保留，不能挑测试赢家当最终模型。

| 输出 | Scan Dice | Subject Dice | Point /270 | 部署前向 |
|---|---:|---:|---:|---:|
| Fit684 native168 seed2026090711 identity | 0.739987540 | 0.728200802 | 252 | 1 |
| Fit684 native168 seed2026090711 pruned | 0.744795478 | 0.733333687 | 257 | 2 |
| Fit684 native168 seed2026090711 full8 | 0.748065184 | 0.736042143 | 262 | 8 |
| Fit684 native168 seed2026090712 identity | 0.740868264 | 0.725770935 | 252 | 1 |
| Fit684 native168 seed2026090712 pruned | 0.744617935 | 0.729756966 | 258 | 3 |
| Fit684 native168 seed2026090712 full8 | 0.746917944 | 0.732314896 | 258 | 8 |

## 固定比较及不确定性

- 2026090711 pruned_vs_full8：Scan差-0.003269706，患者簇95%CI [-0.00512344879786234, -0.0016365451884632523]；Subject差-0.002708456，CI [-0.0050087296866149185, -0.0004636254754238339]；Point差-5。
- 2026090711 pruned_vs_identity：Scan差+0.004807939，患者簇95%CI [0.0026037068142317297, 0.007228309678206384]；Subject差+0.005132885，CI [0.0024300547786244827, 0.008031943571631187]；Point差+5。
- 2026090711门：{'fixed_forwards_at_most4': True, 'point_floor': False, 'scan_ci_floor': False, 'scan_loss_at_most003': False, 'subject_ci_floor': False, 'subject_loss_at_most003': True}。
- 2026090712 pruned_vs_full8：Scan差-0.002300009，患者簇95%CI [-0.0035259011504611025, -0.001219660830166605]；Subject差-0.002557929，CI [-0.004047518584665944, -0.0011435015388002697]；Point差+0。
- 2026090712 pruned_vs_identity：Scan差+0.003749670，患者簇95%CI [0.0006229851977047922, 0.007059286407476014]；Subject差+0.003986032，CI [0.00021031920967691447, 0.00775519249167721]；Point差+6。
- 2026090712门：{'fixed_forwards_at_most4': True, 'point_floor': True, 'scan_ci_floor': True, 'scan_loss_at_most003': True, 'subject_ci_floor': True, 'subject_loss_at_most003': True}。

区间按10,000次患者簇百分位bootstrap，固定seed2026091161/62；未作多重比较校正。通过工程容差不等于正式非劣，也不能将负区间说成完全不降。历史GPU float32 Dice保留，exact_count_dice独立核验误差≤1e−7。

## 直觉是否得到支持

至少一个模型未通过固定内部确认门，不能把开发成功升级成稳定确认；具体失败门保留，不改阈值或择seed补救。工程能省时的证据仍成立，但精度/点代价未稳定满足原要求。

当前同test270总体Scan最高仍按全榜为0.757269662（full972_random_init_voxel_kd_epoch168_flip8_tta），单次前向Scan最高为0.752003176（gcsam_inner_fit864_epoch184）。本面板总体/单次是否刷新：False/False；排名点估计不等于显著性。

## 成本与复核

零训练、新权重。实际测试5670病例前向，包括额外在线pruned复核；131.4秒含结果归档，peak5.503GiB。540例在线pruned与原子融合逐体素exact。全部1620个保存输出的logit/掩膜/argmax/计数及区间独立核验。

部署每模型只需2/3次前向；既有fit阶段实测约3.96/2.65倍加速，不包含预处理或文件读取，不能称端到端加速。测试同时计算三个对照的开销不等于部署成本。新增logits12.656GiB、packed masks405MiB位于/tmp，保留可复核证据，不静默删除。

旧outer、confirmation108、fixed validation108和外部队列无新访问。共同训练源Kaggle/官方字节parity未证明限制保留。

## 证据与后续边界

[七问](IDEA.md)、[最终策略冻结](FINAL_WORKFLOWS.json)、[测试前收据](TEST_PREACCESS.json)、[原始六输出](RAW_TEST270.json)、[独立验证](VERIFICATION.json)、[机器结果](RESULT.json)、[输出掩膜索引](MASK_INDEX.json)。

[开发与在线主实验](../2026-09-11_automatic_workflow_search/RESULT.md)、[20种子敏感度](../2026-09-11_workflow_search_seed_sensitivity/RESULT.md)、[当前交接](../../../AFLOW_BRATS_HANDOFF_CURRENT.md)、[方法目录](../../../AFLOW_BRATS_METHOD_CATALOG.md)、[排行榜](../../../AFLOW_BRATS_LEADERBOARD.md)。

本固定内部确认阶段关闭，不据此继续视图/点/阈值搜索。下一步优先整理可答辩章节、因果边界和端到端工程成本；任何新外部确认另立协议，不能冒称当前全论文三项已全部完成。

- [Seed2026090711 checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/structured_background_aux_20260907/seed2026090711_control/fixed_epoch168.pth) SHA `fdad97d9abc736638bcd9911fc82ddca1a434fd0c9a46c3e9306d608276b3d6e`。

- [Seed2026090712 checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/structured_background_aux_20260907/seed2026090712_control/fixed_epoch168.pth) SHA `5b014fb256c272b828e01322b7985f10d8ac5c53f19fa475761c25537f697e43`。
