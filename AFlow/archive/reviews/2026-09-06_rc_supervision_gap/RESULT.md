# 切除腔不是“没见过”，而是被折叠成普通背景

状态：只读已归档标签模式与WAW-postop错误；0新数组/模型/训练/选择。

Train972有814/972 scans（83.74%）含raw label4 resection cavity，WAW early-postop31有30/31（96.77%）。因此Point被RC吸引不能简单解释为训练完全没有术腔病例。

当前模型只有一个WT输出：labels1/2/3为前景，label4和普通背景都编码0。日常类比是学生经常见到“红色危险标志”，但答案纸只告诉他“这不是目标”，从不要求他说出“它是另一类特殊结构”。大量出现不等于学到单独语义。WAW-postop中Full/GCSAM的16/17个Point miss有13/14落GT RC（81.25%/82.35%），而RC prevalence又比train高13.03个百分点；与“二值监督表示不足”相容。

这不是RC head已被证明。显式预测RC、CSF或使用手术信息会改变输出、loss和部署输入，可能改善Point，也可能损害WT；必须有新train-only development role、matched control、未见validation和七问/预注册。当前fresh容量只有2人/3 scans，不能在已见WAW-postop31上实现、调weight或用GT-RC避让。

事后day分组也只支持边界：day0–1的24例Full/Flip有12个Point落RC，day>1的7例仅1个；但后组太小且时间/患者/手术共变，不能解释为自然恢复因果或形成day路由。

机器证据：[RESULT.json](RESULT.json)。一手任务/混淆边界见[WAW-postop RC文献](../../research_records/2026-09-06_waw_postop_only31_frozen_panel/RC_LITERATURE_CONTEXT.md)。

## 2026-09-06 Fable session对照补记

新完成的UCSD-BraTS2024test56（治疗后、44/56例含RC4、近训练域patient-disjoint）中，Full972 KD168 identity/Flip8 argmax 56/56全落WT、0次落RC；GCSAM 4 miss中仅2落RC。对照WAW-post31的81–82% miss落RC，支持把本报告的"二值监督把RC折叠为背景"解释收窄为：**RC point吸引主要是域偏移放大的失败模式，不是当前监督下早期术后的必然行为**。显式RC head的优先级据此下调，但结论仍只基于两个队列的描述性对照。证据archive/research_records/2026-09-06_ucsd_brats2024test_frozen_panel/POSTHOC_POINT_CATEGORIES.json。
