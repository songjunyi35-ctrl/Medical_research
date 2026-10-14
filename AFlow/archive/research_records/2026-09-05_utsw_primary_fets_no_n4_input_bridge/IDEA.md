# Experiment idea: UTSW官方主FeTS图的no-N4输入桥接

Status: completed，固定3例input-only工程门通过；0模型。

## 1. 现在的问题是什么？

UTSW官方来源有625病例/362人工细化，完整清单6349文件。固定BT0001/2/3桥接已证明`rtumorseg_manual_correction`是三例一致的240×240×155全网格标签，但`brain_t1ce`与`brain_t1ce_ants`都和它几何exact，因此预注册唯一pair门以2个胜者失败。尚无一个不依赖模型分数的输入版本选择。

## 2. 一句话直觉

出版社同时给标准版和附加排版版时，先按出版说明用标准版，不通过考试得分倒选版本。

## 3. 为什么选择这个idea？

- UTSW数据论文把Federated Tumor Segmentation（FeTS，联邦肿瘤分割工具）的配准、去颅骨和初始分割列为主数据流程；Advanced Normalization Tools（ANTs，高级归一化工具）skull-strip版本是“additional set”。目录命名中无`_ants`的`brain_t1ce`对应主版本。
- 当前项目no-N4输入也只依赖非零脑支持、标准化和128³缩放；它会先转RAS并按脑支持裁剪，不要求UTSW的SRI24原网格等于BraTS2024 post-treatment的MNI网格。
- 上一负结果只否定“靠几何得到唯一image-label pair”；它同时给出全网格`r`标签证据，没有否定按发布谱系预先选择主FeTS图。
- RHUH no-N4桥接成功不等于UTSW也可复现；两套发布预处理不同，必须独立检查。

一手来源：[UTSW数据论文](https://doi.org/10.1038/s41597-026-07274-4)、[TCIA UTSW-Glioma](https://www.cancerimagingarchive.net/collection/utsw-glioma/)。

## 4. 具体怎么做？

输入是已双传输exact的BT0001/2/3 `brain_t1ce.nii.gz`和`rtumorseg_manual_correction.nii.gz`。每份先转canonical RAS；标签把所有positive类别合为whole tumor，因为UTSW的4是enhancing tumor，不是BraTS2024 post-treatment的resection cavity。随后完全调用当前no-N4训练输入：非正值清零、按正值脑支持取bbox、脑内population z-score、图像三线性/标签最近邻缩到128³。对copy A/B独立重建并比较image、label、geometry哈希。没有模型、teacher、loss、融合或预测。

## 5. 唯一变化与固定项

- 唯一变化：在两个几何可行图中按官方主流程固定无`_ants`的FeTS `brain_t1ce`。
- 固定不变：BT0001/2/3；全网格`r`人工标签；双传输SHA；canonical RAS；所有positive→WT；no-N4；正支持bbox；source z-score；128³；不访问其余622例、checkpoint或旧cohort。

## 6. 成功、失败和停止规则

- Primary工程指标：copy A/B的3例输出image/label/geometry exact数量。
- Promotion gate：三例源图finite、无负值、同时有0背景和positive脑；image-label canonical shape/affine≤`1e-6`；标签finite整数且positive；输出3/3 image、label、geometry字节/结构exact，输出finite且WT非空。
- Failure/stop rule：任一失败即关闭当前主FeTS no-N4桥接；不改用ANTs、不放宽门、不换病例、不加N4或后处理。
- 是否允许进入test：否。成功只允许未来另立362例全量输入cache协议，不授权模型评测。

## 7. 成本与风险

- 训练时间：0。
- 峰值显存：0GPU。
- 磁盘：复用已有约56MB双副本，只新增小型JSON。
- 推理forwards/延迟：0。
- 最大技术风险：UTSW主FeTS图可能含负值或两次CPU重建不exact；术前SRI24与治疗后MNI的域差仍然存在，即使工程门通过也不代表Dice会好。

## 8. 最终结果（完成后填写）

- Validation/Test：不适用，0模型。
- 置信区间/病例分布：固定3例输入工程探针。
- 是否支持原直觉：支持input-only工程直觉。官方主FeTS图的no-N4输入在双副本中3/3 image/label/geometry exact。
- 学到了什么：三例主图均非负且有黑背景0；UTSW所有positive标签必须合入WT。共享工具的BraTS-only ID正则曾造成实现失败，Amendment仅放行固定3个BT ID后通过。
- 下一步：不下载全362、不跑模型；只有另行预注册且上游方法资格允许时才扩展。
- 报告/归档/哈希：[RESULT.md](RESULT.md)、[RESULT.json](RESULT.json)、INPUT_QC SHA=`7573bed6d03f194a429d7eacb4ffb3b711feeda15e91a47539aaf025020deb59`。
