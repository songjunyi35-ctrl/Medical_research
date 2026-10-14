# Experiment idea: 每位训练患者得到相等的抽样机会

Status: proposed prospectively before current development results; queued after the existing auxiliary-supervision study finishes. No new training started.

## 1. 现在的问题是什么？

用户授权开发池固定fit684 scans/313人、dev180 scans/78人。现有native-volume采样按扫描分配概率：患者间总概率相差约22倍，最高32人占27.0%训练机会，每轮968次draw预计只见271名患者。主指标Subject Dice却按患者等权。当前辅助监督实验尚未访问开发集；本提案依据训练元数据而不是其性能。

## 2. 一句话直觉

像班里有的人交10页作业、有的人交1页：按页抽查会反复点到少数人。先公平地点到每个人，再挑他的一页，可以让训练覆盖更多患者。

## 3. 为什么选择这个idea？

拟合单位与评价单位不一致是可独立检验的因素。统计学对簇大小影响和簇内重采样有一手讨论；这为患者级抽样提供动机，并非保证CNN精度改善。[Williamson等，2003](https://onlinelibrary.wiley.com/doi/10.1111/1541-0420.00005)。

之前native-volume、固定病例覆盖/小病灶回放主要改变扫描或病灶体积侧的机会，未直接把同患者多时点合为等概率患者单位。新方案仍可能丢失全局小病灶优先效果，因此必须同时检查Point和Scan。

## 4. 具体怎么做？

输入、网络、监督均与当前无教师control相同。先计算每例原有体积权重w_i，再除以该患者所有扫描权重之和；全体归一化后每个患者总概率相等，患者内部扫描相对权重仍按旧w_i。沿用相同有放回draw数、batch、增强、168轮和两seed，从相同随机初始化训练。

对照复用本轮已经预注册、正在产出的两个native-volume control，不重新挑选；候选只改采样概率。对照和候选使用相同双通道网络但辅助监督系数都为0，最终均导出单通道WT（whole tumor，全肿瘤）、阈值0、1次前向。主损失相同，不加新教师、损失、TTA（Test-Time Augmentation，测试时增强）或后处理。

## 5. 唯一变化与固定项

- 唯一变化：从全局按扫描体积加权，改为患者等概率、患者内保持原体积相对权重。
- 固定：同fit684/dev180患者、cache、双通道结构/初值、无教师与辅助系数0、主Dice+BCE（二元交叉熵）、972 draws/968有效draw/121步、batch8、168轮、两seed2026090711/12、AdamW、学习率和增强分布、阈值0、导出1×。
- 对照是预先指定的旧control重用，明确reused control/reused development；不称同时间训练或新患者复制。采样索引和batch图像应不同，这是实验因素；随机增强RNG序列仍需一致。

## 6. 成功、失败和停止规则

两候选全部训练完成并冻结后才统一评dev；对照使用前一实验已归档的相同180例输出，不重复推理。每seed Subject差≥.003、Scan差>0、Point最多损失1；Subject与患者聚类Scan配对95%CI下界均≥−.003；两seed平均Scan差≥.003。全部工程和性能门通过才称reused-development支持。

源control的168轮、初值、主监督、采样定义和工程门须审计合格；候选初值匹配同seed control、每患者概率总和exact至数值容差、患者内比值保持；时间/显存门同每臂6h/24GiB。失败不搜索患者温度/混合系数、epoch或种子；不自动访问test/confirmation/外部。

## 7. 成本与风险

新增两条168轮候选，双A100并行预计约2小时；复用两条预先固定的对照节约重复训练，不节约证据核验。复用已有fit RC缓存（aux系数0以匹配control计算），不建新图像cache。新checkpoint/恢复文件约150MB量级，部署4.70M/1×不变。

最大风险是降低小病灶患者的全局优先度；增加患者覆盖只是概率性质，不保证泛化收益。开发集已是复用角色，本方法即使支持也不是独立外部确认。

## 8. 最终结果（完成后填写）

尚未实施/训练。须等当前四模型实验完成后执行，不能根据其dev结果改变本提案；后续正式参数和hash在PREREG及实现冻结收据中绑定。
