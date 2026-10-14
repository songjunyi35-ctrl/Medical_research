# Experiment idea: ReMIND T1ce/手工肿瘤首例来源探针

Status: preregistered source-format probe; no model inference

## 1. 现在的问题是什么？

现有mirror-uint8 UCSD75已揭示且来源有量化，RHUH80只能在no-N4候选过门后运行。TCIA ReMIND提供CC BY 4.0的114人手术数据库、113例手工preoperative whole tumor；官方series、完整`.sums`和公开临床表在任何像素访问前交集出43名弥漫性胶质瘤患者：19 Glioblastoma、15 Astrocytoma、9 Oligodendroglioma。他们各有且仅有一份preop T1 postcontrast series与一份直接参考T1 postcontrast的manual tumor NRRD。但ReMIND只deface、未去颅骨，且DICOM/NRRD物理几何尚未实测，不能直接运行模型。

## 2. 一句话直觉

先拿排序第一份病例检查“插头、插座和答案纸是否真的对齐”，再决定是否值得搬整箱数据；不能看到模型分数后再挑能接上的病例。

## 3. 为什么选择这个idea？

- ReMIND来自Brigham and Women’s Hospital单一机构，不在BraTS2024的Duke/UCSF/Missouri/UCSD/Heidelberg/Michigan/Indiana七个来源站；官方ID也无桥接。机构差异降低重叠风险，但仍不能仅凭namespace宣称患者级绝对零重叠。
- 一手论文说明肿瘤由neurosurgical fellow手工画、attending neurosurgeon复核/必要时修改；增强T1双人标注的Dice约92.2%±2.8%，是高质量人工标签。
- 旧RHUH负背景/N4问题是发布态z-score造成；ReMIND是原始DICOM、不同输入链，旧负结果没有否定它。GLIS-RT因受控访问且GTV/CTV语义不符而关闭，也不否定ReMIND。
- 一手来源：[TCIA ReMIND](https://www.cancerimagingarchive.net/collection/remind/)、[Scientific Data论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11093985/)。

## 4. 具体怎么做？

当前不训练/推理。先用官方NBIA series metadata、package755 `.sums`和临床表生成固定43例候选清单。探针固定取按PatientID排序首例`ReMIND-002`：只下载它的preop T1 postcontrast DICOM series、直接参考该series的manual tumor NRRD，以及同一T1ce参考的自动cerebrum NRRD。逐文件/序列核对官方UID/MD5/计数，分别读取DICOM和NRRD header/array，比较shape、spacing、origin、direction和物理角点；检查影像finite/背景/强度，检查manual tumor为非空二值mask且完全位于同一物理图像域。

自动cerebrum只用于判断是否存在一个提供式非肿瘤brain mask路径，不作为肿瘤真值。若未来依赖它，方法必须标`provided-mask assisted`，不能混入全自动direct-locator主榜。

## 5. 唯一变化与固定项

- 唯一变化：新增一个固定外部病例的来源/格式探针。
- 固定不变：排序首例ReMIND-002；official preop T1 postcontrast UID；manual `preop-SEG-tumor`、automatic `preop-SEG-cerebrum`路径；不做N4/裁剪/标准化选择，不加载checkpoint，不计算Dice/Point，不查看其余42例像素。

## 6. 成功、失败和停止规则

- Primary工程门：DICOM下载完整且Patient/Study/Series UID与官方metadata exact；两个NRRD MD5命中官方`.sums`；三者finite 3D；manual tumor与T1ce物理网格exact或在冻结1e-5几何容差内；tumor二值、非空；cerebrum二值、非空并覆盖tumor；影像背景/非零支持可描述。
- Promotion gate：所有身份、完整性、几何和标签域门通过，才允许设计独立输入桥接；本探针不直接授权43例下载或模型。
- Failure/stop rule：任何门失败即记录并停止，不换病例、不用GT裁剪修复、不重采样后冒充原生exact。
- 是否允许进入test：不允许模型/指标。未来需新的43例或子队列七问/PREREG、全量input/overlap gate与固定checkpoint后才可能一次性外部确认。

## 7. 成本与风险

- 训练：0；GPU：0；模型forward：0。
- 下载：约21.66MB DICOM metadata所示原始总量，加manual tumor约KB级、cerebrum约1.95MB；实际压缩包大小另报。
- 持久磁盘：探针数据放`/tmp`，archive只留KB级报告/官方清单。
- 最大风险：raw DICOM含完整头颅，而当前模型训练输入已skull-stripped；提供的cerebrum mask只覆盖43候选中的34例，且使用它会降低自动化证据等级。另有患者重叠无法用ID桥完全证明的问题。

## 8. 最终结果（完成后填写）

- Validation/Test：无模型访问。
- 置信区间：不适用。
- 是否支持直觉：待首例probe。
- 下一步：仅根据预设工程门决定是否设计桥接，不用任何模型分数。
- 报告/哈希：待填写。
