# Experiment idea: Full972 Lookahead-AdamW慢权重稳定优化

Status: planned and frozen before split generation, implementation, GPU model construction, training, or new evaluation access.

## 链路审视与启发

权威链路仍是：`raw NIfTI → subject级manifest/角色冻结 → image-only brain crop + true-128³ cache → random-init Full972 voxel-KD → train-only inner → fresh-seed Full972 final → fixed validation gate → 仅全过门才作reused test270固定报告`。

1. Full972从随机初始化开始使用972例，相对matched Full756 validation为`+0.010943/+0.007982`，说明早期表示学习比后期补课重要。
2. LAWA6（最近六轮权重平均）在三个fresh seed中相对各自raw都提高Scan/Subject：seed61`+0.002319/+0.001955`、seed62`+0.001268/+0.002772`、seed63`+0.000464/+0.000576`，Point分别`+3/+1/0`。这支持“轨迹平滑”机制，不是单seed偶然正号。
3. 但LAWA6 final仍比archived强seed低`0.007093/0.004897`且少1 Point，说明只修最后六轮太晚，seed差异更早在表示形成期累积。
4. GCSAM（梯度中心化锐度感知最小化）在fresh inner只有`+0.000807/+0.002558/0`且CI门失败；它用每步两次前反传改变局部锐度，没有建立贯穿全程的慢权重中心。
5. 容量、160³、旧loss/feature-KD、困难病例聚合、固定/自适应多seed融合和后期LAWA窗口均已关闭。新方法只改变优化器时间尺度，不增加模型、监督目标或teacher。
6. Fixed validation与test均已多次揭示，当前仍只有同一BraTS数据源。本实验属于development/post-test research；方法选择、`k/alpha`、seed、epoch和门槛不使用test逐病例结果。

采用的启发是第2–3条：既然后期平均有一致但太小的收益，就把“稳健中心”从训练第一步开始维护，而不是继续搜索已揭示LAWA窗口。

## 1. 现在的问题是什么？

当前reused test最高Scan点估计仍为GCSAM`0.752003`，但Point只有252/270；Full972 voxel-KD为`0.751760/0.737327/262/270`。新LAWA6证明同轨迹末期平滑有用，却在fresh final fixed validation只有`0.732919/0.726370/102/108`，低于archived Full972 seed32的`0.740013/0.731266/103/108`。需要从表示学习早期减少AdamW随机轨迹方差，同时保持1×部署。

## 2. 一句话直觉

像登山时先连续走5小步探路，再让“稳健位置”只向探路终点走一半并从那里重新出发；每5步纠偏一次，比到终点才平均最后六个位置更早、更持续。

## 3. 为什么选择这个idea？

- NeurIPS 2019 Lookahead原论文维护fast/slow两套权重：内部优化器走固定`k`步后，slow向fast插值，再把fast同步回slow；论文报告它能降低内部优化器方差并改善Adam/SGD稳定性。
- 作者官方代码提供PyTorch实现并建议评测slow weights。
- 使用论文默认`k=5, alpha=0.5`，不从项目inner/validation搜索。
- GCSAM负结果只否定当前`rho=0.05`两次梯度扰动；Lookahead不计算锐度、不加第二次forward/backward。
- LAWA6 final负结果只否定最后六个epoch的事后uniform平均；Lookahead从第一轮起每5 optimizer steps形成slow center，时间尺度和反馈机制不同。
- Cross-seed parameter soup失败不相关：本方法只在同一个run内把slow向自己的fast插值。

Primary sources:

- Paper: https://arxiv.org/abs/1907.08610
- NeurIPS proceedings: https://papers.neurips.cc/paper/2019/hash/90fd4f88f588ae64038134f1eeaa023f-Abstract.html
- Authors' official code: https://github.com/michaelrzhang/lookahead

来源边界：论文实验是ImageNet/CIFAR/语言任务，并非3D医学分割；本项目把Lookahead包在AdamW外用于T1ce/WT，不能称论文复现。

## 4. 具体怎么做？

### Stage A：fresh train-only inner配对

从Full972 development-train按subject新建`864 fit / 108 inner`。Control与Candidate都用seed20261071、exact相同初始化、sampler、增强、4.70M SegResNet、history/EMA teacher、DiceCE+voxel-KL、AdamW、late-cosine与固定epoch168。

- Control：每步使用原AdamW参数。
- Candidate：AdamW先更新fast weights；每5步执行`slow ← slow + 0.5×(fast−slow)`并把fast复制为slow，从这个更稳位置继续。完成全部20328步后按论文/官方代码评测slow weights；最后3个未同步fast steps不用于候选。

两臂训练脚本不构造inner loader。训练与pairing收据全部通过后，才一次性在fresh inner评估Control raw168与Candidate slow168。

### Stage B：fresh Full972 final

只有Stage A全过门才用新seed20261072在full972上做完全配对的Control/Candidate到epoch168，冻结checkpoint后一次性访问fixed validation108。

输入→teacher→loss→输出：同一增强T1ce进入固定history/EMA teacher形成`0.75/0.25`raw-logit目标；student用真实WT DiceCE加`0.3×T²`逐体素KL学习；Candidate只在optimizer更新后维护slow权重。部署时删除optimizer与fast权重，只运行slow student一次，raw-logit阈值仍为0，最高logit为自动point。

## 5. 唯一变化与固定项

- 唯一变化：原AdamW，或外包`Lookahead(k=5, alpha=0.5)`并部署slow weights。
- 固定：数据源/角色、image-only brain crop、true-128³ cache、4,700,897参数width16 SegResNet、dropout0.1、每阶段两臂相同seed/初始state、batch8、native-volume weighted replacement、972 requested/968 consumed/121 steps、DiceCE、voxel-KD weight0.3、temperature2、AdamW内部超参、late-cosine 200轮分母、固定停168、shift8、clip5、阈值0、无测试时增强、无Segment Anything Model、无后处理、1×推理。
- Lookahead不会消费随机数、不增加forward/backward；slow初始化为step0 student，插值只作用于trainable floating parameters。

## 6. 成功、失败和停止规则

### Stage A

- Candidate slow168必须严格超过matched AdamW Control168的Scan与Subject，Point不得下降。
- Scan/Subject 10,000次paired bootstrap 95% CI下界均`>=-0.003`，bootstrap seed20261073。
- 初始state、全部168轮sampler index SHA、首个增强batch tensor SHA必须两臂exact；Candidate必须完成20328 fast steps、4065 slow updates，并在保存前加载slow weights。
- 任一门失败则不运行Stage B。

### Stage B

- Candidate必须严格超过fresh matched Control与archived Full972 seed32`0.740012535/0.731266217`的Scan/Subject，Point至少103/108且不低于Control。
- 相对两个comparator的四个paired CI下界全部`>=-0.003`，bootstrap seed20261074。
- 全过门才允许一次冻结的reused test270固定报告；否则不访问test。

### 工程停止

- 文档/manifest/cache/native artifact/teacher/decision SHA不符；fit/inner case或subject泄漏；初始state/sampler/首batch pairing不exact；Lookahead step/update/final-slow合同不符；teacher有梯度；loss/gradient非有限；reload/deploy不exact；peak>24GiB或单run>6h立即停止。
- 失败后不调`k`、`alpha`、slow/fast评测选择、epoch、LR、seed、loss、threshold、TTA、SAM或postprocess。

## 7. 成本与风险

- Stage A paired fit864双卡并行预计3.5–4h；Stage B过门后paired Full972双卡并行再3.5–4h；两阶段墙钟约7–8h，formal总约14–16 GPU-hours。
- Candidate额外保存约18MB slow parameters；预计peak仍约20–21GiB，硬门24GiB。
- 不建新cache；预计workspace新增`<300MB`。
- 推理仍4.70M参数、1×forward；训练forward/backward与Control相同，仅每5步多一次参数插值。
- 最大风险：周期性slow更新可能降低方差但也阻碍当前AdamW的快速表示学习；或只提高自身raw而仍不能超过偶然强seed。

## 8. 最终结果（完成后填写）

- Stage A fresh inner：待运行。
- Stage B Full972 final/validation：仅Stage A全过门后运行。
- Test：仅Stage B全过门后执行一次固定报告。
- 置信区间/病例分布：待运行。
- 是否支持原直觉：待运行。
- 学到了什么：待运行。
- 报告/归档/哈希：待运行。
