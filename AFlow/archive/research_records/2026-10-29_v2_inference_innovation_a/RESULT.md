# RACE八视图局部依赖融合108验证结果

已完成16个事前冻结新配置及equal8参照；最佳`race8_local_joint_k17_ridge0.2_blend1.0`，Scan **0.768314657**、Subject **0.758473906**、Point_refined **98/108**、Point_base **98/108**。对equal8 Scan差+0.000302776，患者配对描述性95%区间[-9.873267294531719e-05, 0.0008811266346394381]。

患者5折四折选择/留出汇总Scan 0.768164708。全部继承既有test-tuned历史、复用验证选择，非独立确认；全部配置和变差病例保留，未按270选择。

本轮网络前向0、训练0；复用RACE164既有8视图。GPU0算法CUDA event区间累计11.25秒，墙钟38.45秒，reserved峰值13.016GiB；event区间可含dispatch间隙，不冒充GPU忙碌率。部署仍单模型8前向，局部矩阵与上采样另增计算。

本轮新增270评分0。最终270必须等总控更新最终冻结政策；新训练best只移植已选一个配方，不重新16搜索。

[完整逐例](RAW_VALIDATION108.json) · [排名/区间/分折](RESULT.json) · [精确计划](PLAN.json) · [选择](SELECTED_RECIPE.json)

1836份保存掩膜已在同worker CPU独立重载，全部TP/FP/FN、Dice、Point_refined和Point_base逐例精确一致。最佳full与最佳diag仅差约0.000002852；配对区间跨0，不宣称相关依赖机制收益成立。固定部署与snapshot迁移取DEPLOYMENT_RECIPE.json.innovation_entry。
