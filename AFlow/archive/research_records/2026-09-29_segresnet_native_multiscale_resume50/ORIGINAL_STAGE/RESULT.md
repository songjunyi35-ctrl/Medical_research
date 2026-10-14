# 原生SegResNet多尺度辅助监督结果

状态：按50轮规则停止，共6100次更新；本组只训练原972扫描/444患者，以固定108/50验证选择。**无本候选新270模型推理、outer或confirmation。** 此方向在已有270揭示后设计，标记test-informed，不能称独立盲测。

## 结果

108扫描/50患者原生1mm裁剪栅格：选第49轮，Scan 0.718759、Subject 0.708114、Point 101/108；对旧公开预训练＋原生组合166同格Scan差-0.032420，50患者配对95%CI[-0.040790,-0.024388]。

冻结门：Scan至少0.761179、配对区间下界>0、Subject不少于0.739474、Point不少于89/108；实际未通过。未通过冻结提升门；只评价当前权重、配方和单次运行，不证明辅助监督一般无效。

旧固定193仍是项目比较起点；本组直接单因素参考是相同公开预训练初始化与原生输入的旧组合166。此前最高270定向融合0.775891不改写为本组分数。

## 方法与代价

单T1ce经同N4、图像自身脑区裁剪和非零强度标准化，保持1mm原生裁剪并补零到160×208×160；每轮972各一次、双A100每卡4/global8/122更新。主SegResNet用原Dice+0.1二元交叉熵，训练时半尺度/四分之一尺度辅助损失权重0.25/0.125。部署只加载主模型，仍一次整脑前向；辅助头不进入部署权重。

缓存在worker1109780的/tmp独立重建，图像与标签SHA严格匹配旧原生缓存；双卡全批/尾批合成通过。工程上曾在第一次纯合成检查**结果已提交后**卡在分布式退出，保留旧收据、清理该次进程并用有界退出修复，第二次检查通过，故没有患者像素或训练更新受该故障影响。第一次12进程缓存重建在部分训练影像后按资源效率中止，保留原文件及日志；24进程在独立路径按同配方重建并核验历史SHA，不是第二组训练。

所报显存峰值、墙钟/恢复次数以[训练日志](../../../workspace/BraTS/segresnet_t1ce_wt/native_multiscale_20260924/TRAIN.jsonl)、[GPU合成](GPU_SMOKE.json)、[监督收据](TRAIN_STOP.json)为准。没有为了追分调辅助头权重、训练轮数上限、阈值或模型尺寸。

## 证据和访问

[机器结果](FINAL_RESULT.json) · [逐例108原始报告](../../../workspace/BraTS/segresnet_t1ce_wt/native_multiscale_20260924/validation_e049.json) · [最佳主模型](../../../workspace/BraTS/segresnet_t1ce_wt/native_multiscale_20260924/best.pth) · [重载一致](RELOAD_CHECK.json) · [数据缓存核验](DATA_AUDIT.json) · [方案](IDEA.md) · [执行冻结](EXECUTION_FREEZE.json) · [历史对照](../../../workspace/BraTS/segresnet_t1ce_wt/pretrained_native_res_20260921/validation_e166.json)。
