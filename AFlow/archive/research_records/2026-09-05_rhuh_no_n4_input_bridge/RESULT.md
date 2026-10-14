# RHUH发布态影像no-N4输入桥接：80/80工程门通过

状态：input-only feasibility完成并通过；等待no-N4 Stage A与条件Full972资格。没有下载/读取RHUH标签，没有加载checkpoint、模型forward或Dice/Point，也没有访问fixed validation、outer、confirmation、test270或UCSD75模型结果。

## 结果

固定的TCIA package684后两个时点80 scans/40 subjects全部通过。每份官方T1ce先重算下载收据SHA，再canonical RAS；六面背景众数占比最小`0.983648`（门`≥0.95`），背景值范围`[-13.165734,-2.727962]`且均为全局最小值。减去背景后，80/80背景exact为0、正支持非空，正体素范围`1,156,631–1,708,278`。

不运行N4，以decoded正支持取image-only bbox，在源正支持上重新做population z-score，再三线性缩到128³并用最近邻脑mask清零。80例标准化后最大绝对均值`6.11e-11`、population std最大误差`1.79e-10`，均远低于冻结`1e-6`门；所有输出finite，128³脑mask体素范围`849,331–1,057,382`。80例canonical shape均为`240×240×155`、spacing均`1mm³`、方向均RAS。固定首3例第二次独立构建的image/mask SHA全部byte exact。

## 解释

直觉在工程层面得到支持：RHUH发布者的官方流程已执行SynthStrip与CaPTK z-score、未列N4，因此把负常数背景恢复成0后直接进入no-N4裁剪/标准化，可以得到确定、有限、可重复的模型输入。此前背景解码失败只发生在继续运行数值敏感的正值mask N4完整链；它不否定这条no-N4桥接。

这不是准确率证据，也不授权“外部最佳”主张。RHUH发布态强度和BraTS原始强度仍可能存在域差异；40 subjects的统计区间也会较宽。只有no-N4在E/F两个fold、identity/Flip8全部性能门通过并完成新Full972固定训练后，才能按已冻结规则另写preaccess、下载80份官方标签并一次性跑固定Flip8。若no-N4失败，本桥接作为来源可行性证据保留，但不运行RHUH模型。

后续标签仍只审计了inventory metadata：80条官方路径/MD5全在`.sums`中exact、40人各恰有时点1/2、总大小2,355,783 bytes，并冻结`RHUH-0035_2`唯一文件名特例`segmentation.nii.gz`；本地`raw_labels`目录不存在。机器证据见[`RHUH_LABEL_INVENTORY_READINESS.json`](../../reviews/2026-09-05_30h_chain_audit/RHUH_LABEL_INVENTORY_READINESS.json)，这不等于标签已下载或体素已访问。

## 成本与访问边界

- CPU墙钟`27.315s`；3项合成合同；0 GPU、0模型forward。
- 新增仅KB级代码/报告，无持久图像cache；复用已存在450.98MB官方T1ce。
- 标签下载/label voxel/model/新指标均为false；旧评测cohort无新增访问。

证据：[`INPUT_QC.json`](INPUT_QC.json)、[`RESULT.json`](RESULT.json)、[`PREREG.md`](PREREG.md)、[`IMPLEMENTATION.json`](IMPLEMENTATION.json)。
