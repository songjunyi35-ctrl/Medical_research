# 导师汇报与日期流水账：原始证据索引

本页是资料层。正文不再展示完整路径、哈希和运行收据；需要核对数字时再从这里进入。原始报告、失败记录、代码、数据和权重未删除。

## 8月13日后的原流水账没有丢失

- [8月13日命名长交接的完整快照](HISTORICAL_HANDOFF_DETAIL.md)：实际追加至8月18日。
- [旧结果账本全文](RESULTS_DETAIL.md)、[旧周报全文](WEEKLY_REPORT_DETAIL.md)：标题日期较早，但正文也有8月18日记录。
- [8月19日六份续篇及顺序说明](../../handoffs_20260819/README.md)。
- 后续实验仍按日期分别保存；下面给出主要过程证据。
- [清理前方法全文](METHOD_CATALOG_DETAIL.md)、[清理前当前交接全文](HANDOFF_DETAIL.md)：所有未压入主汇报的支线、来源审计和旧阶段状态保留。

## 关键时间线证据

- [8月13日：师兄路线复现](../../handoffs_pre_20260813/AFLOW_BRATS_RESULTS.md)
- [8月14日：旧修正器搜索与剪枝](../../research_documents/2026-08-17_root_cleanup/INFERENCE_PRUNING_RESULT_20260814.md)
- [8月17日：自动定位测试](../../research_documents/2026-08-17_root_cleanup/PLAIN_LOCATOR_TEST270_RESULT_20260817.md)
- [8月18日：长训练](../../research_records/2026-08-18_plain_long_schedule/RESULT.md)
- [8月19日：模型权重平滑](../../research_records/2026-08-19_plain_ema/EMA168_TEST270_RESULT.md)
- [8月19日：输出融合](../../research_records/2026-08-19_plain_logit_ensemble/RESULT.md)
- [8月19日：蒸馏压缩](../../research_records/2026-08-19_plain_voxel_kd/RESULT.md)
- [8月24日：完整972例从头训练及后续探索测试](../../research_records/2026-08-24_voxel_kd_fulltrajectory_972/RESULT.md)
- [8月25日：多种子融合](../../research_records/2026-08-25_full972_multiseed_ensemble_distill/RESULT.md)
- [8月26日：更宽网络](../../research_records/2026-08-26_full972_width24_fulltrajectory/RESULT.md)
- [8月31日：优化器支线](../../research_records/2026-08-31_gcsam_full972_stability/RESULT.md)
- [9月3日：优化器固定轮次复制](../../research_records/2026-09-03_gcsam_fixed184_replication/RESULT.md)
- [9月4日：八镜像](../../research_records/2026-09-04_direct_locator_flip8_tta_stability/RESULT.md)
- [9月4日：镜像一致训练](../../research_records/2026-09-04_flip_equivariance_consistency_stability/RESULT.md)
- [9月5日：取消额外N4且教师隔离](../../research_records/2026-09-05_no_n4_clean_teacher_replication/RESULT.md)
- [9月5日：自动前景拟合N4](../../research_records/2026-09-05_n4_otsu_clean_replication/RESULT.md)
- [9月7日：患者均衡采样](../../research_records/2026-09-07_subject_balanced_sampling/RESULT.md)
- [9月7日：切除腔辅助监督](../../research_records/2026-09-07_structured_background_aux/RESULT.md)
- [9月8日：空间往返保真](../../research_records/2026-09-08_fit_spatial_roundtrip_v2/RESULT.md)
- [9月10日：第一轮集中数据处理](../../research_records/2026-09-10_eight_hour_data_processing/RESULT.md)
- [9月10日：第二轮集中数据处理](../../research_records/2026-09-10_resumed_eight_hour/RESULT.md)
- [9月11日：自动工作流搜索](../../research_records/2026-09-11_automatic_workflow_search/RESULT.md)
- [9月11日：固定内部确认失败](../../research_records/2026-09-11_automatic_workflow_confirmation/RESULT.md)
- [9月11日：搜索随机性](../../research_records/2026-09-11_workflow_search_seed_sensitivity/RESULT.md)
- [9月11日：单病例总耗时](../../research_records/2026-09-11_workflow_file_to_output_timing/RESULT.md)
- [9月11日：批量调度不同患者复制](../../research_records/2026-09-11_case_parallel_replication24/RESULT.md)

## 32张方法卡的对应原始材料

## M01

师兄路线基础模型复现｜已完成：[查看原始记录](../../handoffs_pre_20260813/AFLOW_BRATS_RESULTS.md)。

## M02

图像自身决定脑区裁剪｜已采用：[查看原始记录](METHOD_CATALOG_DETAIL.md)。

## M03

已知位置条件下的局部精修｜历史支线：[查看原始记录](METHOD_CATALOG_DETAIL.md)。

## M04

围绕自动点的小窗口修补｜未晋级：[查看原始记录](METHOD_CATALOG_DETAIL.md)。

## M05

直接输出的自动分割网络｜主线成立：[查看原始记录](../../research_documents/2026-08-17_root_cleanup/PLAIN_LOCATOR_TEST270_RESULT_20260817.md)。

## M06

单独预测肿瘤中心｜未晋级：[查看原始记录](METHOD_CATALOG_DETAIL.md)。

## M07

灰度强度修正｜未晋级：[查看原始记录](../../research_documents/2026-08-17_root_cleanup/INTENSITY_RESIDUAL_RESULT_20260817.md)。

## M08

更充分的低学习率训练｜有效进步：[查看原始记录](../../research_records/2026-08-18_plain_long_schedule/RESULT.md)。

## M09

全图提高到160³分辨率｜未晋级：[查看原始记录](METHOD_CATALOG_DETAIL.md)。

## M10

训练权重平滑平均｜部分有效：[查看原始记录](../../research_records/2026-08-19_plain_ema/EMA168_TEST270_RESULT.md)。

## M11

两个模型的输出融合｜有效但更贵：[查看原始记录](../../research_records/2026-08-19_plain_logit_ensemble/RESULT.md)。

## M12

直接平均不同模型参数｜明确负结果：[查看原始记录](METHOD_CATALOG_DETAIL.md)。

## M13

知识蒸馏压缩融合｜部分保留收益：[查看原始记录](../../research_records/2026-08-19_plain_voxel_kd/RESULT.md)。

## M14

肿瘤、边界、背景分开蒸馏｜未晋级：[查看原始记录](METHOD_CATALOG_DETAIL.md)。

## M15

小而安全的修正层、边界及域信息｜部分有效：[查看原始记录](METHOD_CATALOG_DETAIL.md)。

## M16

先预测哪里错，再选择修补｜未晋级：[查看原始记录](../../research_documents/2026-08-17_root_cleanup/ELPGR3D_A2_RESULT_20260814.md)。

## M17

旧修正器的工作流与推理剪枝｜受限效率证据：[查看原始记录](../../research_documents/2026-08-17_root_cleanup/INFERENCE_PRUNING_RESULT_20260814.md)。

## M18

新增训练数据从第一轮参与｜重要进步：[查看原始记录](../../research_records/2026-08-24_voxel_kd_fulltrajectory_972/RESULT.md)。

## M19

更多训练种子和更宽网络｜未稳定晋级：[查看原始记录](../../research_records/2026-08-25_full972_multiseed_ensemble_distill/RESULT.md)。

## M20

梯度中心化与锐度感知优化｜单次Scan点估计高、复制不足：[查看原始记录](../../research_records/2026-08-31_gcsam_full972_stability/RESULT.md)。

## M21

固定八镜像预测平均｜当前内部总体纪录：[查看原始记录](../../research_records/2026-09-04_direct_locator_flip8_tta_stability/RESULT.md)。

## M22

镜像一致训练、其他更新与强度预处理｜未稳定晋级：[查看原始记录](METHOD_CATALOG_DETAIL.md)。

## M23

患者均衡采样｜Dice改善、定位不稳：[查看原始记录](../../research_records/2026-09-07_subject_balanced_sampling/RESULT.md)。

## M24

切除腔辅助监督｜当前配方失败：[查看原始记录](../../research_records/2026-09-07_structured_background_aux/RESULT.md)。

## M25

空间、方向与成像增强｜固定配方未通过：[查看原始记录](../../research_records/2026-09-10_resumed_eight_hour/RESULT.md)。

## M26

软占比标签与缩放保真检查｜未获模型晋级：[查看原始记录](../../research_records/2026-09-10_fractional_occupancy/RESULT.md)。

## M27

小区域清理、异质融合与定位点规则｜未晋级：[查看原始记录](METHOD_CATALOG_DETAIL.md)。

## M28

保留原始肿瘤子区标签｜仅工程完成：[查看原始记录](../../research_records/2026-09-10_fit_subregion_labels/RESULT.md)。

## M29

自动模型的工作流搜索与剪枝｜开发通过、内部确认不稳：[查看原始记录](../../research_records/2026-09-11_automatic_workflow_confirmation/RESULT.md)。

## M30

总耗时分解｜找到真正瓶颈：[查看原始记录](../../research_records/2026-09-11_workflow_file_to_output_timing/RESULT.md)。

## M31

病例间并行预处理｜批量工程复制通过：[查看原始记录](../../research_records/2026-09-11_case_parallel_replication24/RESULT.md)。

## M32

公开外部泛化与来源核查｜已完成多个队列：[查看原始记录](../../../AFLOW_BRATS_LEADERBOARD.md)。

## 当前数字、模型与数据角色

- [当前各队列排行榜及置信区间](../../../AFLOW_BRATS_LEADERBOARD.md)。其原始报告和checkpoint链接保持原样。
- [复用开发数据授权](../../research_records/2026-09-07_reused_development_roles/AUTHORIZATION.md)；[固定患者划分](../../research_records/2026-09-07_reused_development_roles/SPLIT.json)。
- [总体领先模型的报告和权重](../../research_records/2026-09-04_direct_locator_flip8_tta_stability/RESULT.md)。
- [单次前向Scan领先检查点及区间](../../research_records/2026-08-31_gcsam_full972_stability/RESULT.md)。
- [历史教师患者接触核查](../2026-09-05_30h_chain_audit/TEACHER_INNER_EXPOSURE.json)。
- 外部比较以当前排行榜及其更正为准；旧跨队列V5“UCSD56确定为负”等措辞已经被后续数值精度审计弱化，不直接复制为最新结论。

## 保存与整理说明

[清理前压缩原件](originals/)可逐字节恢复；[整理收据](PUBLICATION_RECEIPT.json)记录前后哈希与校验。详细快照中的历史进程ID和命令只作追溯，不是恢复任务的授权。
此前生成的9月14日全量ZIP是本次清理前快照，保持原样，不冒充本次精简版。
