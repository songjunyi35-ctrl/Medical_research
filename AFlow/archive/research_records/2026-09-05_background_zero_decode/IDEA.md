# Experiment idea: 线性标准化MRI的背景零点解码

Status: preregistered engineering-only; RHUH image-format QC already accessed, no RHUH labels or model outputs accessed.

## 链路审视与启发

见[30小时链路审视](../../reviews/2026-09-05_30h_chain_audit/SESSION.md)。Brain crop旧协议+0.054250、完整972轨迹对control验证Scan+0.010943支持输入/数据链路价值；160³与width24失败限制了盲目堆计算。镜像一致loss下降35–36%却无Dice提升，强调工程机制与准确率必须分开。此处先解决文件输入契约，不提出分割性能主张。

## 1. 现在的问题是什么？

RHUH80图像已在身份、官方MD5和输入格式门冻结后下载；80/80含负值，未变换的正值N4路线按预注册停止。四份只读抽查中，背景为同一负常数，占全图约81.9–82.4%，且该常数是全图最小值。把所有负数清零会误删一部分组织；负值体素不能全部解释成脑组织。RHUH标签从未下载，模型从未运行。

## 2. 一句话直觉

尺子的零刻度被整体挪走了；从确定的黑背景找回零点，而不改变组织之间的亮度顺序。

## 3. 为什么选择这个idea？

官方RHUH论文说明脑提取后做z-score；官方final_pipeline.sh调用CaPTk Preprocessing -zn，Python备用入口明确(data-mean)/std。原背景0经这类线性变换后变成同一常数。旧1–99 percentile clipping删除强度尾部，而本方法仅减常数，不截断非背景值。此前Gamma与灰度残差的负结果也没有检验文件编码可逆性。

一手来源：[RHUH论文](https://arxiv.org/html/2305.00005v2)、[官方脚本](https://github.com/smcch/RHUH-GBM-dataset-MRI-preprocessing/blob/main/final_pipeline.sh)、[官方Python线性标准化](https://github.com/smcch/RHUH-GBM-dataset-MRI-preprocessing/blob/main/preprocess_mri.py)。这些支持线性变换形式，不能证明拥有原始扫描完整强度参数。

## 4. 具体怎么做？

输入MRI→合并六个外表面的体素，找出现次数最多的精确浮点值b→验证至少95%表面样本等于b且b为全图精确最小值、全图有非背景组织→float64减b后转float32→得到非负、背景精确0的图像→沿用历史N4 shrink2、正值脑包围盒、源非零总体标准化、128³三线性缩放。外表面的边角重复计数规则固定，不做mask膨胀、百分位或阈值搜索。

文字之后的代数：若发布值z=(x−μ)/σ，原背景x=0，则b=−μ/σ，z−b=x/σ。只能恢复到全局比例；N4偏置场校正和后续标准化对该比例的实际数值影响必须实测验证，不能只用公式声称字节相同。

## 5. 唯一变化与固定项

- 唯一变化：输入文件的背景零点解码。
- 固定：实际MRI信号相对顺序、空间几何、N4设置、brain crop、标准化和128³；本工程审计不读取任何分割标签、不运行任何模型，也不选择checkpoint或阈值。
- 三个真实控制图：Full972 train manifest中按case_id排序的首3个病例，只读取T1ce图像；不根据图像外观、标签或结果挑病例。
- 对控制图只模拟两个预先固定编码：以正值体素均值/总体标准差做全图z-score；以及固定z=x/7−3。全程含零背景，恢复后的支持、包围盒必须与原图完全一致。

## 6. 成功、失败和停止规则

- Primary：文件解码与原信号成比例的数值等价，不是Dice。
- 合成门：对既定正比例/偏移编码，恢复结果相对理论x/σ的max absolute error≤1e-5；支持精确一致；finite/shape/dtype契约正确；非主导表面、非全图最小背景、恒定图像、NaN必须拒绝。
- 真实门：两种编码×3 train图的支持和脑包围盒精确一致；恢复信号的源正值z-score与原信号max absolute difference≤1e-4；包括N4和128³全部输入链路后的max absolute difference≤1e-3、mean absolute difference≤1e-5。容差不是字节等价声明。任一失败关闭当前实现，不放宽容差。
- 工程门全部过后才允许RHUH80逐图解码格式QC，仍不访问标签/模型。80/80须满足上述背景契约，不删病例；失败即阻止外部评测。
- 进入任何外部评测仍须单独完整预注册，明确image-QC-informed输入解码、源机构身份证据的局限，并冻结模型/公式/阈值/统计/全部病例；不自动授权旧test270或UCSD75重测。

## 7. 成本与风险

- 训练0，模型推理0，GPU0；CPU预计数分钟，最多20分钟真实N4检查。
- 流式按例读取，不创建多GiB cache；每例临时内存约数百MB，报告<1MB。
- 风险：并非所有MRI都有常数背景或线性变换来源；95%表面与最小值契约失败必须拒绝，不能凭猜测补齐。RHUH原始标准化还可能在计算均值/方差时筛体素，这不改变输出的线性形式，但源数据如有截断/非线性则不能恢复丢失信息。

## 8. 最终结果（完成后填写）

待工程审计。没有验证/外部Dice；不声明分割直觉成功。
