# 原始分辨率整脑纯SegResNet进度

已完成200/200轮、24400次更新，每轮完整972扫描无重复/丢尾，每卡4例。固定复用108扫描/50患者验证，在1mm原始裁剪栅格上（排除画布填充）本轮Scan Dice 0.744239、Subject 0.735281、Point hit 0.916667；Scan患者整簇95%区间[0.6944649555313738, 0.7859687053678814]。最佳验证第173轮Scan 0.746102。

同栅格对照：原N4纯193把128³预测三线性重采样回原始裁剪栅格，Scan 0.742680、Subject 0.733798、Point 0.907407。本轮配对差值（新−基线）{"scan_delta": 0.0015588211223345112, "subject_delta": 0.0014825971610844136, "scan_delta_ci95": [-0.0038692056551426963, 0.007359993829929423], "subject_delta_ci95": [-0.004632133350441498, 0.007750830686052462], "point_hits_delta": 1, "improved_scans": 62, "worse_scans": 46, "tied_scans": 0, "bootstrap": "10000 patient cluster paired percentile95 seed20260918"}。

模型是470万参数纯SegResNet，随机初始化（同基线seed），N4、图像自身脑区裁剪、非零z-score与基线一致；唯一改动是不缩放到128³，而是零填充到固定160×208×160画布，训练与部署都是1次整脑前向。Dice+0.1交叉熵；无蒸馏、灰度项、EMA、融合、镜像增强或后处理。

双A100每卡4例；前100轮lr2e-4，后100轮余弦降到1e-5。本轮训练64.5秒、验证5.2秒，每卡峰值分配35.43GiB、保留36.13GiB。

本组没有test/outer/confirmation像素评估，研究方向已受旧测试揭示影响，不称盲测。仅复用验证选优，CI是描述性区间；不根据测试调参或自动测试。

[机器指标与哈希](RESULT.json) · [固定七问](IDEA.md) · [心跳](HEARTBEAT.json) · [数据审计](DATA_AUDIT.json) · [batch决定](BATCH_DECISION.json) · [基线同栅格对照](BASELINE_E193_NATIVE_VALIDATION.json)

200轮完成，24400次更新，best/latest重载一致性通过。选第173轮，原始栅格Scan/Subject=0.746102/0.735128，Scan95%CI=[0.6987800955197883, 0.7857507645329331]，Point 0.935185；未运行test。[最终报告](FINAL_RESULT.json)。

新−基线（同108原始裁剪栅格）配对差值：{"scan_delta": 0.0034224127453786357, "subject_delta": 0.0013304398953914642, "scan_delta_ci95": [-0.0008976655242559269, 0.007705410726704575], "subject_delta_ci95": [-0.003987823780626058, 0.006618655441062791], "point_hits_delta": 3, "improved_scans": 62, "worse_scans": 46, "tied_scans": 0, "bootstrap": "10000 patient cluster paired percentile95 seed20260918"}。相同复用验证上的选优比较，非独立确认。

**后续用户授权测试已完成（2026-09-20）：** 第173轮test270/119原始栅格Scan 0.768578，同栅格对193配对差+0.007444、CI[-0.000608,+0.014305]；上文“无test”是训练完成时的历史状态，[测试结果](../2026-09-20_segresnet_native_res_e173_test270/RESULT.md)与访问补记优先。
