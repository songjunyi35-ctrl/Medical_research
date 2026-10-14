# Experiment idea: 缩小标签时保留每格肿瘤体积占比

Status: prospective_cpu_preparation_before_any_new_candidate_metric

## 1. 现在的问题是什么？

fit684/313标签crop→128³→native诊断平均Dice0.965432（不是模型分数），184个原生微小组件未在硬缓存中取到自身ID，边缘离散误差存在；当前native dev Scan0.746762/0.744177，Subject0.733213/0.729318。提高整幅分辨率、亮度处理及小孤岛清理未提供稳定收益，弱形变主路线仍在完整训练、没有新dev成绩。

## 2. 一句话直觉

一小格若只有30%体积属于肿瘤，训练时告诉网络0.3，不急着把它硬判为全黑或全白；边界内部仍可保留真实占比。

## 3. 为什么选择这个idea？

[SoftSeg原文](https://arxiv.org/abs/2011.09041)讨论保留部分体积信息，但同时使用归一化线性输出与回归损失，不能把其整套收益归给只换标签。全文BraTS实验为20名高级别胶质瘤、2D切片、tumor core，且开发选择二值阈值；与本单T1ce全肿瘤/3D/固定阈值不同。原五臂消融没有直接给出本项目“原sigmoid+原DiceCE、只换标签”的保证。

因此本实验依据是项目真实的缩小离散误差加独立、无强度参数的体积守恒构造，属于待检验推论，不是SoftSeg完整复现。先前160³负结果不是同一栅格软target对照；小孤岛清理发生在输出端、会删真阳性，也没有测试训练标签占比。仅软化可能被原Dice目标重新推向二值，是明确风险。

## 4. 具体怎么做？

仅fit原生标签→沿旧image-only crop→把原生WT（标签1/2/3）视为单位格内恒定的0/1占据→计算各128³输出格与原生格重叠体积加权平均→保存0到1软target。按三个轴依次做分段常数函数的积分差，float64计算、float16存储；不是Gaussian模糊、阈值化或PyTorch自适应平均池化近似。

旧128³图像完全不变→原翻转/平移/强度/噪声增强，软target仅同步翻转/平移→同4.70M网络、原Dice+0.1交叉熵→原单通道输出。两头匹配初始化、RC辅助系数仍0，RC缓存原样。无教师、无新增elastic。开发仍旧硬标签/阈值logit>0/最高logit Point，不按软target重定义评分。

## 5. 唯一变化与固定项

唯一因素=训练target从nearest-exact硬0/1变为精确格内占比。固定fit684/313/dev180/78、cache图像/裁剪/网格、native体积采样、seed2026090711/12、原初始化、batch8/972draw/968有效/121步/168轮、AdamW/学习率/主损失/原增强RNG、部署1×/阈值0/无后处理。旧native两seed168轮checkpoint与raw作为预先指定control，不重选seed。

训练日志自带Dice会把soft target以>0.5硬化，仅作该训练内部日志，不能与旧硬标签fit Dice当完全相同目标比较。软标签不意味着网络概率已校准，也不修改临床标注原件。

## 6. 成功、失败和停止规则

CPU工程先冻结source label SHA（继承旧AUX_CACHE_RECEIPT的684标签SHA），原hard nearest-exact重建必须684/684与缓存byte exact；soft有限且[0,1]、sum>0；float64质量守恒相对误差≤1e−10，float16存储后相对误差≤0.001；shape/坐标/crop与旧一致，不丢病例。记录fractional边界数、hardened>0.5后是否为空，不据其数据选平滑强度。失败保存partial，不换核/精度/阈值救门。

正式训练仅在弱形变主路线已关闭或无合法晋级资格，且两张GPU空闲、余时能容纳完整168轮双seed+评测/归档时准入；至少按旧完整运行时间×1.1+20min准入。本CPU准备不是训练授权收据，不能提前读其新dev。原弱形变若通过则优先其full864，本备选不混入、不启动。

每seed相对同seed nativecontrol：Subject差≥0.003，Scan>0，Point≥−1，两Dice10,000次患者簇95%CI下界≥−0.003；两seed平均Scan差≥0.003。bootstrap seeds2026091081/82，百分位CI，未多比较校正。必须两seed完整、无选epoch/强度/seed；失败关闭当前target表示与原loss组合，不否定所有软标签方法。

如本备选两seed过门，本8小时内不再承诺额外完整864重训，而允许事前固定seed2026090711（不挑较好seed）的candidate/control各identity和固定八镜像一次旧test270对照，需另行test前checkpoint/代码/网格/SHA冻结；seed12仅开发复现。此模型只fit684、391人授权池中实际训练313人，训练范围必须标注，不能称full864/972。旧test270同输入/硬标签/阈值可比但非盲非外部；只有超过完整精度旧榜值才称该cohort点估计超榜，配对CI另报。整个8小时只允许一个最终test panel；弱形变若已进入test，本分支不再测试。所有4条test输出必须更新两榜。旧confirmation/fixed validation/outer/external不访问。

## 7. 成本与风险

只新建fit软标签：684×128³×2bytes=2,868,903,936 bytes（2.672GiB），放固定/tmp/aflow-fractional-occupancy-20260910，不挤workspace，不删资料。CPU积分/源重hash预计数分钟，源标签需要读取；不读取新影像像素/开发标签来构造。两候选双卡约2–2.3h/4–4.6GPUh，峰值仍≤24GiB先smoke，部署4.70M/1×或事先固定8×。

风险：原损失未必利用软信息，固定阈值可能不匹配；更平滑训练目标可能在硬栅格评分受罚；临时缓存不永久保存，需留代码/index/raw源SHA以便重建。不能拿质量守恒当Dice提升，不能把20人2D肿瘤核心论文效果搬到本队列。

## 8. 最终结果（完成后填写）

尚未实施/训练/读新开发。CPU准备不改变正在运行的弱形变四臂。
