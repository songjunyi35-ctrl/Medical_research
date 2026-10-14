# 八小时数据处理研究汇总

八小时工作范围完成，但超榜目标未达成。

开始2026-09-10T03:28:19+00:00，计划截止2026-09-10T11:28:19+00:00，本报告生成2026-09-10T11:28:47.364138+00:00。

## 核心结论

没有新增test270评测，也没有超过原榜单。找到可复现的开发分割增益：患者采样模型的固定八镜像平均，两seed扫描/患者Dice都提高且四个患者簇95%区间为正；但第二seed定位168→160，触发原安全门。不能用开发0.759与旧测试0.757跨cohort比大小。

| 方法 | 同dev180/78的观察 | 状态 |
|---|---|---|
| 弱形变，native层 | Scan差−.001232/−.000046；Point+2/+4 | 分割门失败 |
| 弱形变，patient层 | 对自身强成员Scan−.011245/−.004780；Point+5/−3 | 固定组合关闭 |
| 体积占比target | Scan差−.000104/−.004325；Point+5/+4 | 定位有益，分割门失败 |
| 小孤岛清理 | Scan差约−.000093/−.000121，Point不变 | 无分割收益，关闭 |
| 数据配方等权融合 | 对native正，但第二seed低于较强sampling成员 | 未稳定超过强成员，关闭 |
| 同模型固定八镜像 | Scan 0.759243559/0.758989503；Point169/160 | Dice-positive / Point门失败，test关闭 |

八镜像Scan差+0.005769/+0.003084，患者簇95%CI分别[0.003818253381104681, 0.007667861598497258]、[0.0005812212260905095, 0.005469962937227189]；Subject差+0.005255/+0.003244，对应CI [0.0026660287379831078, 0.007639378366016113]、[0.0002310871018209463, 0.006332766345659574]。同一78人、非两批独立患者，10k百分位CI未多比较校正。

## 为什么不能忽略定位门

720保存点与原GT核验exact。第二seed lost11/gained3，不只是8个点变化；6个lost病例的Dice反而提高。丢失点距GT中位1个128-grid输入体素、范围1–4.123，不能当毫米或把邻近点改算命中。它提示边界敏感性，未发现坐标评分错误。

## 控制、成本与边界

六条固定168轮训练完成，共121968 optimizer steps、12.8892 GPU小时，峰值20.664GiB。正式开发5040病例forwards，另有有限工程smoke。新占比label cache2.672GiB在/tmp；未删除材料数据。

空间增强只改训练时形变；占比target只改训练标签，原图像/抽样/原损失/硬GT评分不变；推理研究固定0.5组合或全八镜像，没有权重/视图子集/阈值网格。旧模型/单模型失败门均保留，高层新增推理研究在开发揭示后明确另注册，不称原始盲计划。

部署成本：原模型4.70M/1×，固定八镜像同模型8×，跨数据模型融合2模型/2×（若八镜像16×）。没有把多次推理叫单次模型提分。当前所有GPU计算任务均已结束。

fit684/313和dev180/78保持，未消费dev为full864训练。没有test、oldconfirmation、fixed validation108、outer、external新性能访问；训练来源镜像/官方字节parity未证明，历史test-informed背景保留。

## 同口径旧榜与下一步

旧reused test270/119整体Scan最高0.757269662、Subject0.741253585、Point265/270，Full972KD168固定8×；1次前向Scan最高0.752003176（GCSAM184）。这些数字没有被本轮开发结果刷新。

下一优先不是继续盲扫亮度、形变强度或混合比例，而是围绕已观察到的边界敏感最高点，预先设计不牺牲区域覆盖的自动定位/训练数据研究。当前不能改Point定义救门，也没有证据说数据处理或模型已到上限。

默认执行/补丁读者在收尾时出现环境级找不到路径；主机/bin/sh授权执行恢复，原始报告已存在且无重跑。汇总为机器结果驱动的报告生成和带备份的机械文档刷新，历史正文不删除。

[机器汇总](RESULT.json)、[当前交接](../../../AFLOW_BRATS_HANDOFF_CURRENT.md)、[方法百科](../../../AFLOW_BRATS_METHOD_CATALOG.md)、[旧榜](../../../AFLOW_BRATS_LEADERBOARD.md)。

- [weak_elastic结果](../2026-09-10_weak_elastic/RESULT.md)，[raw](../2026-09-10_weak_elastic/RAW_DEVELOPMENT.json)，[验证](../2026-09-10_weak_elastic/VERIFICATION_RECEIPT.json)。

- [fractional_occupancy结果](../2026-09-10_fractional_occupancy/RESULT.md)，[raw](../2026-09-10_fractional_occupancy/RAW_DEVELOPMENT.json)，[验证](../2026-09-10_fractional_occupancy/VERIFICATION_RECEIPT.json)。

- [peak_protected_cleanup结果](../2026-09-10_peak_protected_cleanup/RESULT.md)，[raw](../2026-09-10_peak_protected_cleanup/RAW_DEVELOPMENT.json)，[验证](../2026-09-10_peak_protected_cleanup/VERIFICATION_RECEIPT.json)。

- [data_recipe_ensemble结果](../2026-09-10_data_recipe_ensemble/RESULT.md)，[raw](../2026-09-10_data_recipe_ensemble/RAW_DEVELOPMENT.json)，[验证](../2026-09-10_data_recipe_ensemble/VERIFICATION_RECEIPT.json)。

- [subject_fixed_flip8结果](../2026-09-10_subject_fixed_flip8/RESULT.md)，[raw](../2026-09-10_subject_fixed_flip8/RAW_DEVELOPMENT.json)，[验证](../2026-09-10_subject_fixed_flip8/VERIFICATION_RECEIPT.json)。

- [2026090711_native_elastic checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/weak_elastic_20260910/seed2026090711_native_elastic/fixed_epoch168.pth) SHA `12bd86129ac5d387c4ffd608012f7aabdccbb30e2b5780b90c24e8f1455683d7`。

- [2026090711_subject_elastic checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/weak_elastic_20260910/seed2026090711_subject_elastic/fixed_epoch168.pth) SHA `273537ab9cc0de7e2bcd97ce1885ec9319436275f4f30ebc897c17ac41775e65`。

- [2026090712_native_elastic checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/weak_elastic_20260910/seed2026090712_native_elastic/fixed_epoch168.pth) SHA `e8e75534e582ba692624ae403e13c9d5264dd42b462d6691ca31fd03f4aad93c`。

- [2026090712_subject_elastic checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/weak_elastic_20260910/seed2026090712_subject_elastic/fixed_epoch168.pth) SHA `4fc95ae79768192c5033804b214f70f55ee36659492fe3934e62bc8141f795fe`。

- [2026090711_fractional_occupancy checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/fractional_occupancy_20260910/seed2026090711_fractional_occupancy/fixed_epoch168.pth) SHA `ea144d90429c733e998d4f38c980e8cbd542aa09e2ec1febeec8030941449d1e`。

- [2026090712_fractional_occupancy checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/fractional_occupancy_20260910/seed2026090712_fractional_occupancy/fixed_epoch168.pth) SHA `1ed5c7a850997076f4222c69e38d5c8cf676abd0d5631731dc4ea9e33ef85dae`。


[八小时后续点策略追加](CONTINUATION_ADDENDUM.md)。
