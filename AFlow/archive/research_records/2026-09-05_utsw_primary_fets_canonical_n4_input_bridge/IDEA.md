# Experiment idea: UTSW官方主FeTS图的canonical N4输入桥接

Status: completed，固定3例input-only工程门通过；0模型。

## 1. 现在的问题是什么？

UTSW官方625例中362例人工细化，主FeTS图+全网格`r`标签的no-N4输入已在BT0001/2/3双副本中3/3可复现。但未来若只给no-N4模型一个绝对外部分数，没有same-seed canonical端到端对照，无法区分方法差异与术前UTSW本身的难度；canonical外部输入尚未建立。

## 2. 一句话直觉

比较两种药方时，两边都按各自原配方煎药：no-N4不补光，canonical则按训练时同样做N4补光。

## 3. 为什么选择这个idea？

- UTSW固定3例主图均finite、非负、黑背景0、正脑非空，可以直接进入historical canonical正值mask N4，不需要RHUH的负背景零点解码。
- 旧0.057679数值失败来自float32负背景编码往返的近零信号误差被N4放大；UTSW这里直接读取同一官方非负float图的双SHA副本，故旧负结果没有否定它。
- RHUH Otsu-N4最终image重建失败也不等同于canonical单线程N4；本桥接独立使用两个进程检查，不能假设确定性。
- no-N4 Stage A仍未开卷，现在冻结canonical外部输入，不受候选性能影响。

一手实现依据：项目冻结的canonical N4 cache builder；[SimpleITK N4 Bias Field Correction](https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html)。

## 4. 具体怎么做？

输入仍是BT0001/2/3的`brain_t1ce.nii.gz`与`rtumorseg_manual_correction.nii.gz`两份官方副本。每个copy在独立Python进程中把SimpleITK全局线程固定为1，转RAS、把所有positive UTSW标签合为WT，调用canonical `n4_correct(source, shrink=2)`，再走相同正支持bbox、脑内population z-score、图像三线性/标签最近邻缩到128³。父进程只比较两份输出哈希和geometry，并与已通过no-N4 QC对照。没有teacher、loss、融合、checkpoint或预测。

## 5. 唯一变化与固定项

- 唯一变化：相对UTSW主FeTS no-N4桥接加入原canonical正值mask N4。
- 固定不变：BT0001/2/3、双源SHA、主`brain_t1ce`、全网格`r`标签、all-positive→WT、RAS、shrink2、原N4迭代配置、bbox、z-score、128³、单线程、旧cohort与模型均不访问。

## 6. 成功、失败和停止规则

- Primary工程指标：两个独立进程的3例输出image/label/geometry exact数量。
- Promotion gate：跨进程image/label/geometry各3/3 exact；输出finite、WT非空；相对no-N4，label与geometry3/3 exact、image3/3不同。
- Failure/stop rule：任一门失败即关闭当前canonical UTSW桥接；不重跑、不改线程/N4 mask/迭代/容差/病例。
- 是否允许进入test：否。通过只允许未来另立成对的全362 cache/外部评测协议。

## 7. 成本与风险

- 训练时间：0。
- 峰值显存：0GPU。
- 磁盘：复用已有56MB，只新增小JSON。
- 推理forwards/延迟：0。
- 最大技术风险：N4即使单线程也可能跨进程出现float差异；如果不exact，不能用它构建严格配对外部cache。

## 8. 最终结果（完成后填写）

- Validation/Test：不适用，0模型。
- 置信区间/病例分布：固定3例工程探针。
- 是否支持原直觉：支持。跨进程canonical-N4 image/label/geometry3/3 exact；对no-N4 label/geometry不变、image3/3改变。
- 学到了什么：UTSW官方非负主FeTS图可稳定进入historical canonical N4，且单线程N4没有在固定3例引入跨进程漂移。
- 下一步：不访问全362/模型；只有no-N4资格、Full972和RHUH优先链满足后，才另立配对外部协议。
- 报告/归档/哈希：[RESULT.md](RESULT.md)、[RESULT.json](RESULT.json)、INPUT_QC SHA=`1cec4ba98fae05e54caca2a1eb0a14681758e79ed6ac35ec47b0a28a58353be0`。
