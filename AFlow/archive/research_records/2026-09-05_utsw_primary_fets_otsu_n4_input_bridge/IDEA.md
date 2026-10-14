# Experiment idea: UTSW官方主FeTS图的Otsu-N4输入桥接

Status: completed，固定3例input-only工程门通过；0模型。

## 1. 现在的问题是什么？

Otsu候选正在未揭示E42/F42上训练；若最后只有它通过，RHUH Otsu发布态桥接已经因最终float image跨进程不exact而关闭，UTSW目前也只验证了no-N4和canonical正值mask N4。这样Otsu会没有合法的新外部输入路线。

## 2. 一句话直觉

在知道考试成绩前，先检查第三种“补光时只看可靠脑组织”的药方能否在同三份样卷稳定复现。

## 3. 为什么选择这个idea？

- Otsu候选在train三例的编码敏感机制门与972 cache重建门均已通过，当前正式训练也未开卷。
- RHUH失败输入经过负背景解码和发布态z-score；UTSW主FeTS图是直接官方非负float、背景0。RHUH当前实现失败不证明UTSW Otsu也不稳定。
- UTSW的主图、全网格`r`人工标签、双传输SHA和canonical/no-N4配对geometry已固定，允许只改变N4拟合mask。
- SimpleITK官方N4示例使用Otsu前景mask：[官方实现](https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html)。

## 4. 具体怎么做？

输入仍是BT0001/2/3两个官方副本的`brain_t1ce.nii.gz`与`rtumorseg_manual_correction.nii.gz`。copy A/B在独立Python进程、SimpleITK线程1运行`OtsuThreshold(full image,0,1,200)`拟合mask、shrink2和四级50次N4，把偏置场应用到完整正支持；随后同正支持bbox、population z-score、图像三线性/标签nearest-exact到128³。UTSW所有positive标签先合为WT。父进程比较跨进程输出及已归档canonical/no-N4桥接；无模型/teacher/loss/预测。

## 5. 唯一变化与固定项

- 唯一变化：相对canonical N4只把拟合mask从全部正值换为Otsu200自动前景。
- 固定不变：BT0001/2/3、双源SHA、主FeTS图、全网格`r`标签、all-positive WT、N4 shrink/iterations、RAS、bbox/z-score/128³、线程1、0模型/旧cohort。

## 6. 成功、失败和停止规则

- Primary工程指标：copy A/B输出image/label/geometry exact数量。
- Promotion gate：跨进程三项均3/3 exact；对canonical/no-N4的label与geometry3/3 exact；Otsu image相对两者均3/3 different；fit mask非空且只含源positive。
- Failure/stop rule：任一失败关闭当前UTSW Otsu桥接；不重跑、不改200 bins/线程/N4/病例/容差。
- 是否允许进入test：否。通过只为未来另立Otsu-vs-canonical UTSW362条件协议提供依据。

## 7. 成本与风险

- 训练时间：0。
- 峰值显存：0GPU。
- 磁盘：复用约56MB，只增JSON。
- 推理forwards/延迟：0。
- 最大风险：RHUH曾跨进程不exact；UTSW仍可能失败。即使通过，也不说明Otsu分割更准。

## 8. 最终结果（完成后填写）

- Validation/Test：不适用，0模型。
- 置信区间/病例分布：固定3例工程探针。
- 是否支持原直觉：支持input-only直觉。跨进程image/label/geometry/mask3/3 exact，对canonical/no-N4标签/几何同且图像不同。
- 学到了什么：RHUH发布态的Otsu exact失败没有迁移到UTSW直接非负主FeTS图；来源/输入表示确实影响数值确定性。
- 下一步：不访问全362/模型；仅Otsu Stage A通过后另立Otsu-vs-canonical外部协议。
- 报告/归档/哈希：[RESULT.md](RESULT.md)、[RESULT.json](RESULT.json)、INPUT_QC SHA=`55b7331695fcdb80723d68c4ed0377d275cab547d2b8f2f52b2a8448fd58a82c`。
