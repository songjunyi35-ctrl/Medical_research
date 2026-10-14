# T1ce-only目标可观测性：train972与WAW152标签组成审计

状态：train-role 972 labels与已保存WAW metadata只读完成；0 validation/outer/test新访问，0图像/模型/训练。

## 结论

“没有enhancing label3”本身不是WAW独有：Train972有269/972 scans（27.67%，145/444 subjects至少一例），WAW有29/152（19.08%）。因此不能说模型从未见过无强化病例。

真正明显偏移的是“WT只有FLAIR abnormality label2，且没有label1/3/4提供其他结构线索”：

| cohort | only `{0,2}` scans | scan比例 | 涉及subjects |
|---|---:|---:|---:|
| train972 post-treatment | 30/972 | `3.09%` | 18/444 (`4.05%`) |
| WAW152 preoperative | 26/152 | `17.11%` | 26/152 (`17.11%`) |

WAW scan比例高`14.02`个百分点、约为train的`5.54×`。Train中许多label3缺失扫描仍有label4切除腔（虽不属于WT）或label1核心，T1增强图可能仍有定位线索；WAW是术前、没有label4。WAW三输出共同5例零Dice中4例正是only-label2。

这支持“单T1增强输入对纯FLAIR定义区域有可观测性限制，并且训练中该纯模式较少”的解释，但不证明因果：术前/术后、发布者标注协议、scanner、强度和体积同时不同。不能按GT label组成/grade排例或路由，也不能用已见WAW启动重采样；当前immutable-role只剩2人/3 scans，缺新泛化证据。

## 证据与访问

- [机器结果](RESULT.json)，SHA-256 `139765822000b869f58c2dc1f44b06a70a98a9fa91c789087757ca5d2e1d4148`
- Train manifest SHA `6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65`
- WAW cohort SHA `25de15904d8db5e6d6e304bc73058dba4f43e7534ae0a533f4f06636613a263a`
- WAW failure-tail SHA `cdfed37eb2e7586c65525cc25d7b1bf420e172b63a76798050d4a887675ae714`

访问仅train-role原始标签972和已归档WAW JSON；validation/outer/confirmation/internal test图像或指标、外部新arrays、model forward均为0。

一手定义与消融支持“信息受限、并非绝对不可能”的精确表述：BraTS2024的SNFH由T2/FLAIR异常定义；独立模态消融把T1CE定位为ET/TC强模态、FLAIR或T1CE+FLAIR定位为WT最小输入；另一研究中T1CE-only仍可取得较高WT Dice但加入FLAIR继续提高。详见[文献边界](LITERATURE_CONTEXT.md)。这些不同模型/队列绝对数值不能搬入本项目。
