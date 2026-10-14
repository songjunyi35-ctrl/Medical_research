# 患者采样Point变化：净值不等于全部变动病例

这是正式结果后的描述性核查，不是预注册性能检验。只读保存的开发结果，0新forward、0影像/标签/checkpoint数组访问，原联合门不变。

| control→candidate | seed11病例数 | seed12病例数 |
|---|---:|---:|
| 命中→命中 | 164 | 158 |
| 命中→未命中（lost） | 6 | 5 |
| 未命中→命中（gained） | 3 | 10 |
| 未命中→未命中 | 7 | 7 |
| 净命中变化 | −3 | +5 |

因此seed11不是“只有3例发生变化”，而是9例发生Point类别变化；seed12有15例变化。之前体积分组报告的“最小组净损失3”仍正确，但其他体积组也可能同时有lost和gained并相互抵消，不能误写成所有失去命中的病例都在最小组。

两seed共同lost的病例仅BraTS-GLI-02239-100和BraTS-GLI-02881-100；共同gained仅BraTS-GLI-02139-103。不按这些ID筛病例、路由模型或调阈值。

seed11的6个lost中，1例Dice提高、5例下降；seed12的5个lost中，3例Dice提高、2例下降。例如02239-100在seed11的Dice由0.609098升到0.628678，但Point从命中变未命中；02881-100在seed12的Dice由0.507287升到0.578466，同时Point失去命中。这是实际记录中的指标分离，不仅是理论可能。

不能据此说所有Point失败都是无关紧要的“排序小问题”：不少lost病例Dice也下降。原报告未保存完整logit，无法判断最大值量化并列、局部置信度排序或具体空间病因，本次不重跑模型解释。它只证明整体mask重叠改善不能担保单个最高分点稳定。

所有lost/gained病例的原/新Dice、真值/预测大小和点坐标完整保存在[机器结果](SUBJECT_SAMPLING_POINT_TRANSITIONS.json)，没有只挑上述例子保留数据。参考[审计计划](POINT_TRANSITIONS_PLAN.md)与[正式主结果](../../research_records/2026-09-07_subject_balanced_sampling/RESULT.md)。
