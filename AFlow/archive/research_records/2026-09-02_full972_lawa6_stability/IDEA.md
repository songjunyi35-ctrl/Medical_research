# Experiment idea: Full972 同轨迹最近六轮权重平均

Status: planned and frozen before split generation, implementation, GPU model construction, training, or new evaluation access.

## 链路审视与启发

当前权威链路是：`raw NIfTI → subject级manifest/角色冻结 → N4 + image-only brain crop + true-128³ cache → random-init Full972 voxel-KD → train-only inner → fresh-seed Full972 final → fixed validation gate → 仅全过门后作reused test270固定报告`。

1. 数据表示和早期参与产生了最大可靠收益：image-only brain crop旧匹配协议约`+0.054` Scan，epoch99→168约`+0.0154`，Full972从第一轮参与相对matched Full756 validation为`+0.010943/+0.007982` Scan/Subject。限制是这些结果仍缺独立外部cohort。
2. 加体素或参数没有解决泛化：true-160³约`1.95×`体素成本却降到test Scan`0.737060`；width24由4.70M增到10.57M参数，fresh final反而`-0.002872/-0.003525`。因此不再改容量或全图分辨率。
3. 训练轨迹波动是当前未解决瓶颈：fresh seed31/32/33 inner Scan为`0.746412/0.736109/0.732792`；等权输出共识能救弱seed约`+0.0108–0.0162` Dice，却只胜强seed`+0.000530/+0.000063`且CI失败。
4. 单点Scan提高可能损伤Point：GCSAM相对Full972 KD reused-test为`+0.000244/-0.000061/-10 Point`。因此本实验同时冻结Scan、Subject和Point，不用单指标放行。
5. 固定/GT路由多老师已经关闭：GT hard routing相对control为`-0.003104/-0.002975/-3 Point`。本实验不组合不同seed或teacher，只平均同一训练轨迹的相邻权重。
6. 证据边界：本地只有同一BraTS 2024 post-treatment GLI数据源；fixed validation108与test270都已多次揭示。本实验先用新train-only inner和两个fresh seed，属于development/post-test research，不以test逐病例输出选任何设置。

采用的启发是第3条：如果同一seed在晚余弦阶段围绕一个好盆地小幅抖动，把最近几轮参数的中心作为单模型，可能比只取epoch168边缘点更稳定。

## 1. 现在的问题是什么？

Reused internal test270最高Scan点估计是GCSAM`0.752003`，但Point只有`252/270`；Full972 voxel-logit knowledge distillation（逐体素logit知识蒸馏，简称voxel-KD）Scan只低`0.000244`，却有`262/270` Point。Fresh训练又显示不同seed的inner Scan相差最高`0.013620`。当前需要的是不增加推理forward、同时减少末期轨迹偶然性的方案。

## 2. 一句话直觉

像把同一个学生最后六次小测的答题习惯取平均，而不是只拿最后一次：目标是抹平偶然抖动，不是让六个不同学生硬拼答案。

## 3. 为什么选择这个idea？

- LAWA（Latest Weight Averaging，最近权重平均）原论文提出等权平均最近`k`个逐epoch checkpoint，默认`k=6`在其ImageNet/BERT实验中表现稳定，并观察到等权优于指数衰减。
- SWA（Stochastic Weight Averaging，随机权重平均）原论文解释同一权重盆地内的平均点可能更靠近宽平区域中心，最终仍是一个模型。
- 项目跨seed parameter soup失败，是因为不同独立轨迹不在同一参数盆地；本实验只平均同一seed、连续六轮，不受该负结果直接否定。
- 旧EMA（Exponential Moving Average，指数移动平均）使用step级`0.999`长记忆、旧Full756轨迹，并会给更早参数非零权重；本实验是Full972随机初始化完整轨迹、只等权平均epoch163–168，所以旧EMA负结果没有检验当前因素。
- GCSAM改变每步优化并增加约`1.554×`训练时间；它没有测试相同AdamW轨迹的后期权重中心，因此其CI失败也不否定本实验。

Primary sources:

- LAWA paper: https://arxiv.org/abs/2209.14981
- LAWA official code: https://github.com/JeanKaddour/LAWA
- SWA paper: https://arxiv.org/abs/1803.05407
- PyTorch averaged-model documentation: https://docs.pytorch.org/docs/stable/optim.html#stochastic-weight-averaging

来源边界：论文主要是ImageNet/BERT/CIFAR且SWA原始实验多用SGD与固定/循环学习率；本项目保持AdamW晚余弦schedule，只做LAWA式最近六个epoch checkpoint等权平均，是3D T1ce/WT适配，不冒充论文复现。

## 4. 具体怎么做？

### 阶段A：两个fresh seed的train-only inner稳定性门

从Full972 development-train按subject新建`864 fit / 108 inner`。在fit864上并行训练seed`20261061/20261062`到固定epoch168。每个run的输入T1ce、4.70M模型、history/EMA teacher、真实标签Dice加0.1交叉熵、`0.3×T²`逐体素KL散度、`T=2`、native-volume采样、AdamW和late-cosine全部不变。

每个epoch训练完成后，只在epoch163–168捕获六份模型参数。Candidate对每个floating parameter/buffer做精确等权平均；整数buffer要求六份完全相同并原样保留。训练脚本不构造inner loader。两个run和两个平均checkpoint全部冻结后，才一次性在fresh inner比较每个seed的raw epoch168与其LAWA6。

### 阶段B：fresh-seed Full972 final

只有两个Stage A seed都全过门才运行seed`20261063`的Full972训练到epoch168，同样同时生成raw168与LAWA6。冻结checkpoint和pre-access receipt后一次性访问fixed validation108。

输入→teacher→loss→输出：同一增强T1ce先进入两个冻结history/EMA teacher形成`0.75/0.25`raw-logit教学目标；student用真实WT标签DiceCE加逐体素KL学习；训练结束只把student自身epoch163–168参数等权平均；部署输入T1ce后只跑这个平均后的student一次，阈值仍为raw logit`>0`，最高logit为自动point。

## 5. 唯一变化与固定项

- 唯一变化：最终使用raw epoch168参数，或同一run epoch163–168六份参数的等权平均。
- 固定：数据源/角色、image-only brain crop、true-128³ cache、4,700,897参数width16 SegResNet、dropout0.1、每个run的seed/initial state、batch8、native-volume weighted replacement、972 requested/968 consumed/121 steps、DiceCE、voxel-KD weight0.3、temperature2、AdamW、late-cosine 200轮分母、固定停168、shift8、clip5、threshold0、无测试时增强（TTA）、无Segment Anything Model（SAM）、无后处理、1×推理。
- 冻结LAWA公式：`k=6`，逐epoch采样，epochs`163,164,165,166,167,168`，uniform coefficients`1/6`；不使用inner/validation选择窗口、权重或epoch。

## 6. 成功、失败和停止规则

### Stage A

- 在两个fresh seed中，LAWA6都必须严格超过同run raw168的Scan与Subject，Point不得下降。
- 四个paired Scan/Subject 95% CI下界全部`>=-0.003`；10,000次subject-aware/既有配对bootstrap合同，seed`20261064`。
- 两个seed任一门失败，Stage B不运行。

### Stage B

- LAWA6必须严格超过同run raw168和archived Full972 seed32 validation`0.740012535/0.731266217`的Scan与Subject，Point至少为matched raw的Point且`>=103/108`。
- 相对两个comparator的四个paired Scan/Subject 95% CI下界全部`>=-0.003`；10,000次bootstrap seed`20261065`。
- 全过门后，用户本次“主要为了超越leaderboard”的明确目标只授权一次冻结的reused test270固定报告；测试不得触发窗口、epoch、seed、阈值或后处理调整。

### 工程停止

- 文档/manifest/cache/native artifact/teacher SHA不符，fit/inner case或subject泄漏，参数捕获epoch不等于`163..168`，整数buffer不一致，平均公式/reload/deploy logit不exact，teacher有梯度，loss/gradient非有限，peak allocated`>24GiB`或单run`>6h`立即停止。
- 失败后不改`k=6`、epoch窗口、系数、seed、loss、LR、阈值、TTA、SAM或postprocess。

## 7. 成本与风险

- Stage A两个fit864 seed双卡并行，预计墙钟3.5–4h；Stage B过门后单个Full972 run再3.5–4h。
- 峰值显存预计20–21GiB/卡，硬门24GiB。
- 在线CPU FP64累加或等价受控平均最多保留一份累计状态；raw/LAWA checkpoint和报告预计每run约110–130MB，总新增`<500MB`，不建cache。
- 推理仍4.70M参数、1×locator forward；训练forward数量与raw control完全相同。
- 最大风险：晚余弦epoch163–168可能太接近，平均几乎无变化；或平均提高Dice却让最高logit Point下降。两种情况都按门停止。

## 8. 最终结果（完成后填写）

- Stage A two-seed fresh inner：待运行。
- Stage B Full972 final/fixed validation：仅Stage A全过门后运行。
- Test：仅Stage B全过门后执行一次冻结的reused内部报告。
- 置信区间/病例分布：待运行。
- 是否支持原直觉：待运行。
- 学到了什么：待运行。
- 报告/归档/哈希：待运行。
