# Experiment idea: 固定旧对照的原始类别错误拆解

Status: planned descriptive audit, not a new predictor or promotion experiment.

## 1. 现在的问题是什么？

native dev Scan0.746762/0.744177、Point170/163；患者均衡Scan0.753475/0.755906、Point167/168。平均分无法说明漏画的主要组织类别；旧切除腔辅助0.1配方失败，也没有证明切除腔一定不是主要误差来源。

## 2. 一句话直觉

把错题按题型分堆：先知道错在哪里，再判断未来的数据处理是否对症。

## 3. 为什么选择这个idea？

BraTS2024的原始标注区分非增强核心、周围异常区域、增强组织和切除腔；项目只合并1/2/3作为WT，4排除。官方挑战原文 https://arxiv.org/abs/2405.18368 与作者指标实现 https://github.com/rachitsaluja/BraTS-2024-Metrics/blob/main/metrics_GLI.py 提供背景。本报告保留label1/2/3/4数字为权威，不能仅凭名称把所有label2称肿瘤或水肿。此诊断不训练新的辅助监督、不重新搜索旧配方。

## 4. 具体怎么做？

固定已有native与患者均衡两seed共四checkpoint，重新跑reused dev180原图，仅1次前向。原生整数标签按旧image-only裁剪及nearest-exact映射到原128；必须合并1/2/3后与旧hard WT cache每例byte exact。每模型先核逐例Dice计数、点位置/命中与已存raw相同，再将false negatives分为1/2/3、false positives分为0/4，并记录Point所在原始类别。
无训练、loss或融合变化，输出仅诊断报告和二值mask供限定核验；不修改部署、阈值或定位器。

## 5. 唯一变化与固定项

没有预测因素变化，仅增加错误的分组统计。固定fit684训练过的四模型、原128图像/标签、batch8/BF16、logit>0和argmax、dev180/78、源报告/模型/cache/清单SHA；不加载旋转新checkpoint。标签类别cache在/tmp，180×128³uint8=360MiB；原始WT不改。四模型mask另180MiB，优先/tmp；不建立影像cache。

## 6. 成功、失败和停止规则

先合成测试类别计数恒等式，fit前8例图像模型smoke可验证有限性；随后固定一次四模型开发描述读出。旧source raw SHA/逐例预测计数/点parity失败立即停、保存已访问范围，不挑例或用新输出替换旧分数。类别计数须严格还原TP/FN/FP、Point总数；保存mask用于零前向独立核对，注意没有保存logits不能独立重算argmax。无性能promotion、无bootstrap、无因果/上限推断。该诊断不改变已经固定的旋转/条件模糊参数与是否执行规则，不触发test访问。

## 7. 成本与风险

CPU建标签cache预计1–5分钟，模型720前向约1分钟但与训练共享一张GPU；诊断allocated<8GiB，总显存安全才跑，不停止训练。临时标签360MiB+四mask180MiB，报告<3MiB。标签各类体积不同，大类错误多不意味着单位体素更难，所以同时报告各类真实体素量与漏检比例，不只报错误占比；患者等权和总体体素汇总分开。

## 8. 最终结果（完成后填写）

未执行；0新候选开发前向/0test。仅即将重读已揭示旧模型开发表现。
