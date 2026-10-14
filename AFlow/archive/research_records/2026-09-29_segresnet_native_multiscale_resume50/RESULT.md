# 原生SegResNet多尺度辅助监督结果

状态：完成200轮，累计24400次更新，其中原50轮6100更新、新增150轮18300更新；从完整第50轮模型/辅助头/AdamW状态继续，原停止记录保留。用户看过50轮验证后追加预算，此次为自适应预算修订；本组只训练原972扫描/444患者，以固定108/50验证选择。**无本候选新270模型推理、outer或confirmation。** 此方向在已有270揭示后设计，标记test-informed，不能称独立盲测。

## 结果

108扫描/50患者原生1mm裁剪栅格：选第166轮，Scan 0.753092、Subject 0.739669、Point 94/108；对旧公开预训练＋原生组合166同格Scan差+0.001913，50患者配对95%CI[-0.004999,+0.010082]。

冻结门：Scan至少0.761179、配对区间下界>0、Subject不少于0.739474、Point不少于89/108；实际未通过。未通过冻结提升门；只评价当前权重、配方和单次运行，不证明辅助监督一般无效。

旧固定193仍是项目比较起点；本组直接单因素参考是相同公开预训练初始化与原生输入的旧组合166。此前最高270定向融合0.775891不改写为本组分数。

## 方法与代价

单T1ce经同N4、图像自身脑区裁剪和非零强度标准化，保持1mm原生裁剪并补零到160×208×160；每轮972各一次、双A100每卡4/global8/122更新。主SegResNet用原Dice+0.1二元交叉熵，训练时半尺度/四分之一尺度辅助损失权重0.25/0.125。部署只加载主模型，仍一次整脑前向；辅助头不进入部署权重。

旧worker1109780已不在活动清单，本阶段在worker1114913两张A10080GB上续训；每卡4/global8及原学习率不变，硬件迁移可能改变浮点轨迹，不承诺逐位连续。新worker/tmp原生训练/验证缓存与旧载荷SHA一致；原50轮主干、辅助头和AdamW步计数6100全部恢复，原最佳49轮参与完整1–200轮选择。旧50轮规则未通过的事实与原报告、快照完整保留。

所报显存峰值、墙钟/恢复次数以[训练日志](../../../workspace/BraTS/segresnet_t1ce_wt/native_multiscale_resume50_20260929/TRAIN.jsonl)、[GPU合成](GPU_SMOKE.json)、[监督收据](TRAIN_STOP.json)为准。没有为了追分调辅助头权重、训练轮数上限、阈值或模型尺寸。

## 证据和访问

[机器结果](FINAL_RESULT.json) · [逐例108原始报告](../../../workspace/BraTS/segresnet_t1ce_wt/native_multiscale_resume50_20260929/validation_e166.json) · [最佳主模型](../../../workspace/BraTS/segresnet_t1ce_wt/native_multiscale_resume50_20260929/best.pth) · [重载一致](RELOAD_CHECK.json) · [数据缓存核验](DATA_AUDIT.json) · [方案](IDEA.md) · [执行冻结](EXECUTION_FREEZE.json) · [历史对照](../../../workspace/BraTS/segresnet_t1ce_wt/pretrained_native_res_20260921/validation_e166.json)。
