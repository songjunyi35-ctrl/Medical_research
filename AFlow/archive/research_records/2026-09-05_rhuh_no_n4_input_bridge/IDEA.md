# Experiment idea: RHUH发布态影像的no-N4输入桥接

Status: preregistered input-only feasibility; no labels or model inference

## 1. 现在的问题是什么？

RHUH-GBM官方包已经固定后两个时点80 scans/40 subjects，80份T1ce共450,981,845 bytes且官方MD5全部匹配；但80/80图像都有负值，无法直接进入训练时假设“黑背景=0、脑组织>0”的输入管线。此前背景零点解码在支持和原始z-score层面成立，却在继续执行历史正值mask N4偏置场校正后仅4/6通过，最坏完整输入差0.057679/0.002245，因而在模型前停止。RHUH标签尚未下载，模型forward仍为0。

## 2. 一句话直觉

RHUH发布者已经把亮度统一过了：像一份已经校过曝光的照片，只需把被平移的黑纸恢复为零，不再自动“补光”第二次。

## 3. 为什么选择这个idea？

- RHUH发布者官方预处理仓库固定commit `46cf3825278243036c08131b62dcfcfb1d3bba39`明确列出：DICOM转NIfTI、T1ce配准SRI24、其他模态配准、SynthStrip去颅骨、CaPTK z-score、DeepMedic初分割；其中没有N4步骤。
- 旧工程失败明确发生在解码后继续经过N4，而不是背景支持、裁剪框或源z-score恢复失败。当前no-N4候选恰好去掉这个数值敏感阶段，所以旧负结果没有否定本桥接。
- 这不是用RHUH选择训练方法：现在只做无标签输入可行性。只有no-N4在预注册E/F全门通过并完成Full972训练后，才可能另行冻结checkpoint并一次性外部评测。
- 一手来源：发布者[官方预处理仓库](https://github.com/smcch/RHUH-GBM-dataset-MRI-preprocessing/tree/46cf3825278243036c08131b62dcfcfb1d3bba39)与[TCIA RHUH-GBM](https://www.cancerimagingarchive.net/collection/rhuh-gbm/)。

## 4. 具体怎么做？

输入是80份已下载并逐文件通过官方MD5的T1ce，不加载分割。每份图先转canonical RAS；从六个边界面的众数找发布态黑背景常数，要求它覆盖边界至少95%且等于全图最小值。减去该常数，把黑背景恢复成0；以正值支持得到image-only包围盒，只裁脑区；在裁剪前的正值脑组织上重新做总体z-score，再把裁剪区域三线性缩到128³，并用最近邻脑mask把外部清零。输出只计算哈希与数值QC，不保存cache、不进模型。

若以后获准外部评测，输入仍按同一桥接处理，交给已经固定的no-N4 Full972自动定位器；输出raw logit，固定Flip8三轴八镜像逆翻平均、阈值0，Point为raw logit全局argmax。RHUH所有正标签均视为Whole Tumor，避免发布README的增强区标签4与既有审计所见标签3编号差异改变语义；该规则必须在标签下载前冻结。

## 5. 唯一变化与固定项

- 唯一变化：对RHUH发布态z-score影像做严格背景零点解码，并与no-N4模型一致地不再运行N4。
- 固定不变：80 scans/40 subjects、时点1/2全纳入、官方文件身份、canonical orientation、image-only brain crop、source非零总体z-score、true-128³、4.70M模型、固定checkpoint、raw-logit阈值0、Flip8公式、无Segment Anything Model/无后处理。任何标签内容不得用于裁剪、病例排除或预处理选择。

## 6. 成功、失败和停止规则

- 当前Primary工程指标：80/80官方图像哈希重验；六面背景众数占比≥0.95且等于全局最小值；解码后background exact 0、支持非空；canonical spacing正且finite；image-only bbox非空；源正值z-score mean绝对值≤1e-6、population std误差≤1e-6；128³输出finite且脑mask非空；固定首3例独立重建byte exact。
- 当前Promotion gate：所有病例与所有门必须全过，才把“no-N4输入桥接可行”保留为候选外部路径。
- Failure/stop rule：任一失败即模型前关闭；不调0.95门、背景算法、病例、方向、裁剪、标准化或插值。
- 是否允许进入test：本阶段不允许标签下载、模型forward或任何Dice/Point。未来必须同时满足no-N4 E/F全部性能门、条件Full972完成、80标签官方MD5/geometry/标签域门以及独立preaccess收据；否则不评。fixed validation、outer、confirmation、test270和UCSD75永不由本桥接授权。

## 7. 成本与风险

- 训练时间：0。
- 峰值显存：0 GPU。
- 磁盘：不建cache，只新增KB级报告；已存在450.98MB官方T1ce不复制。
- 推理forwards/延迟：当前0；若未来获准，Flip8为80×8=640 forwards。
- 最大技术风险：发布态CaPTK z-score可能与训练原始强度分布不同；通过格式门只证明管线可定义和可重复，不证明分割准确。40个患者也不足以稳定检出约0.01量级差异。

## 8. 最终结果（完成后填写）

- Validation：等待input-only QC。
- Test（如授权）：未授权。
- 置信区间/病例分布：无模型指标。
- 是否支持原直觉：待QC。
- 学到了什么：待填写。
- 下一步：全门通过也只等待no-N4 Stage A/Full972资格，不提前下载标签。
- 报告/归档/哈希：待填写。
