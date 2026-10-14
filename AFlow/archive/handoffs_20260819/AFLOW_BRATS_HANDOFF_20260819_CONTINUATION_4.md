# AFlow BraTS continuation handoff 4 — 2026-08-19

本文件承接`AFLOW_BRATS_HANDOFF_20260819_CONTINUATION_3.md`。

## 32. 多agent一手文献检索与固定parameter soup负结果

用户判断可能出现科研瓶颈并明确授权多agent联网找idea。三个agent分别检索优化/权重空间、3D医学
分割定位/难例、ensemble蒸馏。共识是：当前为plain单模型泛化与互补知识整合的局部瓶颈，不是
分辨率或数据内无信息；成功logit ensemble已经证明互补信号真实存在。

按三方共同最高优先级，先做零训练固定parameter soup：
`theta=0.75*history168+0.25*EMA168`，不搜索alpha。validation结果scan/subject/point从历史
`0.731601/0.722121/0.925926`崩至`0.431343/0.430487/0.796296`，108/108 cases与50/50
subjects全部下降。说明两个独立seed落在不同权重盆地，output-space融合成功不代表parameter-space
可线性融合。按预注册不进test、不继续扫alpha。归档见
`archive/research_records/2026-08-19_plain_weight_soup/RESULT.md`。

下一优先级为固定ensemble knowledge distillation：teacher严格为已经validation/test验证的
`0.75 history + 0.25 EMA168` logits，student保持单个128³ 4.70M SegResNet。建议先从历史
epoch168初始化做32轮低LR pilot，与相同步数supervised-only continuation配对；teacher固定
eval/no-grad，几何增强必须对齐。先用GT DiceCE加soft logit/probability回归，再按需要加入空间
peak distribution KL；所有权重和epoch只在validation冻结，test不得调参。完整一手文献与候选排序见
`AFLOW_BRATS_RESEARCH_IDEAS_20260819.md`。
