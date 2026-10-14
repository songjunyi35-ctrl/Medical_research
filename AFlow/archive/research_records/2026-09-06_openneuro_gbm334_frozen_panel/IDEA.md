# Experiment idea: OpenNeuro-GBM334冻结checkpoint流式面板

Status: completed negative at input gate; closed before checkpoint-with-external-data, model inference, or performance access.

## 1. 现在的问题是什么？

现有冻结checkpoint在外部队列表现随医院明显变化：GCSAM184相对Full identity的Scan Dice差在Africa95为`+0.068617`且配对95%区间全正，在CPTAC33、WAW术前152和WAW早期术后31只有`+0.00195/+0.01022/+0.00166`且区间跨0。固定Flip8在五个报告cohort的Scan点估计都正，但只有internal270、CPTAC33和WAW术前152区间全正，而且要8倍推理。项目需要更大、全新医院、专家复核术前标签的外部队列，而不能继续从已揭示病例挑模型。

OpenNeuro ds007045 v1.0.1公布337名八医院术前原发GBM。按版本化的未N4 MNI T1-CE与最终标签严格交集，在任何像素前冻结334名；3例因缺一侧文件排除。

## 2. 一句话直觉

像把两把已经封存刻度的尺子带到八家新医院，一口气量334个新病例：只换被测病人，不看结果后磨尺子。

## 3. 为什么选择这个idea？

- 334例比现有任何单个外部队列大，可缩窄“小幅差异到底稳定不稳定”的不确定区间。
- 八家医院均不属于BraTS2024的Duke/UCSF/Missouri/UCSD/Heidelberg/Michigan/Indiana七站；无患者桥，所以只称institution-disjoint/no-known-overlap。
- 标签遵循BraTS 0/1/2/3约定，由SegResNet初始化后全部经神经放射科医师检查，87例小修、11例大幅人工重画；比只有自动mask的LUMIERE和2026 Duke DIR数据更适合当外部真值。
- 现有N4家族失败只是否定重新训练的no-N4/Otsu配方；本方案保留已冻结canonical N4，不被它否定。Africa/CPTAC/WAW差异也没有评过这八家医院。
- 一手依据：[Scientific Data数据描述](https://www.nature.com/articles/s41597-026-07953-2)与[OpenNeuro v1.0.1](https://openneuro.org/datasets/ds007045/versions/1.0.1)。

## 4. 具体怎么做？

每次只下载一个冻结病例的MNI152配准、skull-stripped、未发布者N4/z-score的T1-CE和同空间最终mask，先核git-annex SHA-256。转到canonical RAS后严格检查shape/affine，WT（whole tumor，全肿瘤）定义为标签`{1,2,3}`的并集；运行项目固定N4偏置场校正、只按图像非零区域裁剪、非零z-score，图像三线性/标签最近邻缩到128³。

同一Full972 KD168 checkpoint运行固定八个三轴镜像；identity既是第一个视图，也单独计Full identity，八个raw logits翻回原方向后float32等权平均为Flip8。GCSAM184只跑identity。阈值`logit>0`生成mask，raw-logit最大点用于Point。每例完成后立即释放临时NIfTI，不常驻多GiB cache；仅保存源SHA、几何/标签门、Dice、Point、intersection/predicted/target计数。

334例全部发布绝对Scan=Subject Dice/Point和10,000次病例bootstrap；配对报告Flip8−Full identity与GCSAM−Full identity。不同历史cohort只并列解释，不pool。

## 5. 唯一变化与固定项

- 唯一变化：新增OpenNeuro v1.0.1冻结334名术前原发GBM外部cohort。
- 固定不变：版本和病例交集；T1-CE来源variant；WT=`{1,2,3}`；canonical RAS、单线程N4 shrink2、image-only crop、非零z-score、128³；Full/GCSAM checkpoint及SHA；Full八视图顺序、raw-logit float32等权融合；阈值0；无Segment Anything Model（SAM）、连通域、形态学或模型ensemble；指标、empty policy、bootstrap与解释门。
- 禁止：用MGMT/年龄/性别/医院选模型，换另一预处理variant，按结果排病例，调阈值/视图/权重，或重测已关闭cohort。

## 6. 成功、失败和停止规则

- Primary指标：Full identity、Full Flip8、GCSAM identity的Scan=Subject Dice和Point；两个相对Full identity的配对Dice差、95%区间、改善/变差/tie及Point差。
- 解释门：delta>0且配对CI下界>0写“该队列支持”；delta>0但CI跨0写“正向但不确定”；delta≤0写“不支持/反向”。不是模型promotion，不触发重训。
- 工程门：334对下载SHA exact；每例图像/标签有限、geometry≤`1e-5`、label只允许0/1/2/3、WT非空；image-only crop不能丢任何WT；首例预处理跨独立调用byte exact；checkpoint/evaluator/预注册/模型前收据哈希一致。
- Failure/stop：任一例任一门失败立即停在当时边界，保留已访问源/性能范围；绝不丢例、扩框、换variant或放宽容差。网络瞬时失败可在同一versionId/SHA对象上重试，不得换对象。
- 是否允许进入test：它本身是一次性public external-to-project评测、not hidden blind；不访问internal/fixed validation/outer/confirmation或其他旧cohort。

## 7. 成本与风险

- 训练时间：0；无teacher、loss、optimizer。
- 峰值显存：预计5–6 GiB，单卡。
- 磁盘：源总量约2.03 GiB但不常驻；单例临时文件通常<10 MiB，raw/收据为MiB级。当前只余约1.9 GiB，禁止建334例float32 cache。
- 推理forwards/延迟：Full八视图+GCSAM一次，共334×9=`3006` forwards；Flip8部署延迟约identity的8倍。
- 最大技术风险：v1.0.1有3例缺配对，README中心计数有一例误差；发布者先做MNI注册/skull stripping，项目再N4；逐例流式网络可能中断；标签是expert-reviewed/refined而非全部从空白手画；无跨库patient-ID桥；继承checkpoint训练源官方Synapse字节parity未证明。

## 8. 最终结果

- Validation：不适用；这是外部来源输入门。
- Test：未运行。全334输入预检中332例通过，BO118有4/348742、KA86有6/97700个WT体素落在图像正值crop外，违反零丢失门。
- 置信区间/病例分布：无性能、无Dice/Point/CI；不得报告332例子集。
- 是否支持原直觉：来源规模与标签价值成立，但“现成canonical crop可无损直接接入”不成立；不能检验模型跨院泛化直觉。
- 学到了什么：即使shape/affine/SHA/label字典全对，skull-stripped图像支持仍可与专家mask差几个体素；输入门必须在模型前覆盖全体病例。
- 下一步：关闭当前面板，不丢例、GT扩框、换variant或放宽门。等待发布者修正支持，或未来独立训练并先过门的全FOV模型。
- 报告/归档/哈希：[结果](RESULT.md)、[机器结果](RESULT.json)、[输入收据](INPUT_PREFLIGHT.json)。
