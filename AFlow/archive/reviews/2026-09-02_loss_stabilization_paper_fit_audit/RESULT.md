# 2026术后胶质瘤loss stabilization论文适配审计

日期：2026-09-02 UTC

## 结论

不在本轮剩余时间实施或训练。它是未来有条件候选，不是已预注册的方法；因此不更新方法百科状态、不创建训练cache、不访问任何validation/test/external数据。

一手论文是Crisan与Borza的2026 arXiv预印本“Post-Operative Glioma Segmentation via Loss Stabilization, Normalization and Subspace Attention”。论文在MU-Glioma-Post上训练、UCSF-ALPTDG上外部验证，报告Generalized Dice Loss（广义Dice损失，GDL）的Whole Lesion从内部`0.88`降到外部`0.73`，提示由类别体积自动决定权重会把训练医院的组织比例写进loss，在域变化时不稳。

## 对小白最直白的解释

- **论文看到的问题：** 同一个loss在自己医院像会做题，换医院就掉了15个百分点。
- **核心直觉：** 不要让一张图里“哪类面积大/小”自动决定老师嗓门大小；先把不同扫描仪的亮度极端值截掉，再让监督更稳定。
- **论文做了什么：** 四通道MRI进入SwinUNETR；Baseline用Dice+固定类别权重交叉熵，Norm variant同时使用脑mask内1–99百分位clip后z-score和bottleneck体素对比学习，SACA则把768个bottleneck通道拆成4个类别子空间做attention。

## 为什么不能直接搬进当前系统

1. 当前系统只有T1ce单通道和二分类WT，而SACA的`S=K=4`明确对应ET/NETC/SNFH/RC四类。把它硬改成一个WT query会变成另一个方法，论文8% ET sensitivity机制不再成立。
2. 论文Norm variant一次改变两项：percentile normalization和voxel contrastive loss，不能从其结果推断哪个因素有效。当前规范要求单因素，不能整包缝合。
3. 当前loss是DiceCE+voxel teacher KL，并不是论文判负的standard GDL；所以论文不能证明改loss会提升本项目。
4. 百分位clip与Gamma增强不是同一件事：Gamma是随机非线性亮度扰动且fresh final失败；固定1–99 clip是确定性去极值。Gamma负结果不否定clip，但也不提供放行证据。
5. UCSF-ALPTDG是BraTS 2024明确来源cohort，不能作为本项目新外部验证；论文的external结果只支持一般域偏移风险，不能当我们的独立test。

## 唯一可干净检验的未来分支

若以后磁盘充足，可只改变一项：在N4之后、非零总体z-score之前，将每例脑内强度固定clip到该例1st/99th percentile；Control保持现有不clip。其余Full972、fresh subject-disjoint inner、模型、teacher、loss、sampling、epoch、阈值0、无TTA/SAM/postprocess全部相同。

这仍不能现在开跑：true128 train cache约7.4GB，当前约12GB可用；新cache后只余约4–5GB，不满足安全余量。另需约两卡各3.4小时paired inner、实现/合成测试/N4 cache/GPU smoke和完整归档，剩余窗口不足。未经用户授权也不能删除材料腾盘。

## 未来门槛

任何真正实施都必须另写IDEA/PREREG并在标签/validation前冻结：至少两个fresh train-only inner seeds均要求Candidate相对matched Control的Scan/Subject点估计非负、Point少不超过1/108、两项paired CI lower≥`-0.003`；只有全部通过才允许fresh Full972 fixed validation。UCSD75已经访问并关闭，绝不用于clip percentile、N4顺序或epoch选择；最终确认应等待未见RHUH或另一人工标签cohort。

## 成本与访问

- 本审计：0训练、0GPU、0cache、0模型forward。
- 仅阅读论文和既有项目结果；没有新增train/inner/fixed validation/outer/confirmation/test270/UCSD external访问。
- 未来预计新增约7.4GB cache，paired inner约6.8 GPU-hours、约3.4小时双卡墙钟，部署推理仍1×。

## 一手来源

- [arXiv 2607.22749论文](https://arxiv.org/abs/2607.22749)
- [BraTS 2024 post-treatment challenge论文](https://arxiv.org/abs/2405.18368)

