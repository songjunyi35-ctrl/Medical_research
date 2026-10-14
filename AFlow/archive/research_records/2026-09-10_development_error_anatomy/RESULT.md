## 旧模型错误按原始类别拆解 ✅ 完成，描述性不晋级

**一句话：** 已把错题按原始标签类别拆开，模型与评分未改变。

**出发点：** 原native与患者模型开发Dice约0.744—0.756，但平均分不足以定位数据处理的具体问题。

**具体做法：** 四旧fit684模型重读同dev180/78；720例原计数和Point位置逐例exact，再从原label0—4分类。无新训练、不加载旋转模型，不改其冻结参数。

**结果：**

| 旧模型 | Point /180 | FN中label2占比 | label2体素召回 | FP中label4占比 |
|---|---:|---:|---:|---:|
| 2026090711_native | 170 | 92.40% | 74.86% | 12.45% |
| 2026090711_subject | 167 | 91.99% | 75.47% | 11.63% |
| 2026090712_native | 163 | 91.23% | 74.73% | 12.05% |
| 2026090712_subject | 168 | 91.87% | 75.57% | 10.81% |

**是否支持直觉：** 说明错误在原始类别中的分布，不说明某类别是因果瓶颈或其大小就是困难度；完整各类分母、患者等权召回及Point类别见机器结果。无CI，不把四模型当四份独立队列。

**代价/下一步：** 720旧模型前向/40.5秒，诊断峰值4.815GiB；/tmp类标签360MiB、mask180MiB。独立掩膜核对720/720，未保存logits所以不声称重算argmax。只用于解释，不触发test、不修改本轮旋转/模糊参数。

[七问](../../../archive/research_records/2026-09-10_development_error_anatomy/IDEA.md)、[结果](../../../archive/research_records/2026-09-10_development_error_anatomy/RESULT.json)、[原始](../../../archive/research_records/2026-09-10_development_error_anatomy/RAW_DEVELOPMENT.json)、[核验](../../../archive/research_records/2026-09-10_development_error_anatomy/VERIFICATION.json)。
