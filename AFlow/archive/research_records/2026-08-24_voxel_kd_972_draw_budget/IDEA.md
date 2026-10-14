# Experiment idea: 972病例配额一致的voxel-KD

Status: planned and frozen before implementation, GPU smoke, formal training, and validation access.

## 1. 现在的问题是什么？

历史最佳单模型KD200只用756 scans / 346 subjects，在fixed validation108达到Scan/Subject/Point `0.734384/0.724434/101/108`，reused internal test270达到`0.742976/0.725179/0.955556`。

后续训练池扩到972 scans / 444 subjects，但所有972实验每轮仍只请求756次、drop-last后消费752次。Native972达到`0.734060/0.724703/101/108`；最新coverage/replay达到`0.734158/0.724950/98/108`并相对same-compute replacement control提高Scan/Subject `+0.001481/+0.001890`。这说明额外216例有多样性价值，但972病例共享原756预算后，每例平均更新次数下降，可能抵消收益。

## 2. 一句话直觉

练习册从756页增加到972页，却仍只做752页；这次让做题量随练习册同比增加，看看更多病例能否真正转化为更好的单模型。

## 3. 为什么选择这个idea？

- 三次972相关实验都给出相对matched control的Dice正信号，数据多样性机制不是空想。
- 最新coverage实验说明减少重复能改善Dice，但Point退化；它没有增加总训练次数。
- 当前唯一尚未直接测试的主变量是972病例是否需要同比计算预算，而不是新的loss、teacher或架构。
- LoMix、DCD、KAS、校准、困难体素、病例尾部、adapter和coverage/replay负结果都固定了约752 effective draws/epoch，没有否定预算扩容。
- 这不是把validation用于选择972这个数字；972是冻结训练池的自然病例数，candidate只把请求draw从756改为972。

## 4. 具体怎么做？

两臂输入均为同一972病例的T1ce `128^3` cache，并使用原生1mm WT体积权重`sqrt(median/volume)` clip`[0.5,4]`带放回抽样。Student从historical168初始化；Teacher仍为`0.75 history168 + 0.25 EMA168` raw-logit融合；loss仍为Dice+`0.1`交叉熵+`0.3*T^2`逐体素KL，`T=2`。

- `budget752_control`：请求756 draws，batch8/drop-last消费752，94 optimizer steps/epoch。
- `budget968_candidate`：请求972 draws，batch8/drop-last消费968，121 optimizer steps/epoch。

两臂都跑global169–200共32轮，保存176/184/192/200。输出仍是raw logit阈值0的4.70M参数plain locator。

## 5. 唯一变化与固定项

- 唯一变化：每轮请求draw `756→972`，对应有效draw `752→968`、optimizer steps `94→121`，训练计算增加28.72%。
- 固定：972 scans / 444 subjects、同一cache、原生体积权重、history168 student、history/EMA teacher、KD公式、seed20260924、batch8、workers4、shift8、AdamW、LR按global epoch、weight decay、clip5、32轮、checkpoint grid、threshold0、无TTA/SAM/postprocess、1×推理。

## 6. 成功、失败和停止规则

- Candidate只按fixed validation108 Scan最大选择176/184/192/200；Subject exact tie，再以更早epochtie。
- Selected candidate必须同时严格超过same-epoch budget752 control和archived KD200的Scan/Subject；Point至少101/108。
- 相对两个comparator的Scan/Subject paired bootstrap 95% CI下界均不低于`-0.003`，10,000 draws、冻结seed。
- Cache/native weight/manifest/checkpoint/document hash不符、draw/step不精确、loss/gradient非有限、teacher有梯度、peak allocated>24GiB、control formal>45min、candidate formal>60min或reload非exact立即停止。
- 只有全部门通过才允许一次reused internal test270；失败后不改draw、epoch、LR、threshold或postprocess。

## 7. 成本与风险

- 两张A100并行：control预计约32min，candidate约41–43min；墙钟约43min。
- 显存：两臂预计约20.3GiB，硬门24GiB。
- 磁盘：复用train/validation cache；8 checkpoints和报告预计小于260MB。
- 推理：仍4,700,897参数、1×forward，无部署增量。
- 最大风险：更多step可能在相同global-epoch LR日程下过拟合，或者改善Dice但仍不能恢复Point。

## 8. 最终结果（完成后填写）

- Validation：待运行。
- Test：仅全部门通过后允许。
- 是否支持原直觉：待运行。
- 下一步：待运行。
