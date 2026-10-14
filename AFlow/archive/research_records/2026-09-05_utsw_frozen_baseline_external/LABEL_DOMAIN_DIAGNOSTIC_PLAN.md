# UTSW362 raw人工标签域诊断计划

冻结时间：2026-09-06T00:01:58Z，首次全量unique-value扫描前。

原cache builder在BT0007因冻结`{0,1,2,4}`门停止。没有CACHE_RECEIPT、模型或指标；partial cache稀疏文件保留。

本诊断只读取已双SHA通过的362个`rtumorseg_manual_correction.nii.gz`，逐例要求finite/integer，记录unique值及每值体素数，统计值3/4的病例数与共现。禁止读取图像、运行N4、生成cache、加载checkpoint或计算分割Dice。

解释门：

- 若存在负数、>4、非整数或非finite，来源标签异常，当前baseline关闭。
- 若全量仅为0..4且所有positive仍对应论文声明的三个肿瘤区，本诊断可支持另立透明Amendment，把数值域门改为`{0,1,2,3,4}`，但WT仍严格为所有positive；不能根据模型分数决定。
- 无论结果如何，原`{0,1,2,4}`实现失败证据保留，不能覆盖。

