# RHUH-GBM 外部确认可行性审计

日期：2026-09-02 UTC

## 结论

RHUH-GBM是当前第二优先级的高质量外部真值候选，但本轮在官方文件inventory之前停止：没有下载NIfTI、没有读取分割体素、没有运行模型，也没有建立Leaderboard行。

证据能力需要预先降级预期：用已封存UCSD subject配对差做粗略设计估算，40人的95% CI半宽约`0.02364`，大于当前模型差`0.01610`；检测同等效应约需177名独立患者（α=0.05、80% power）。因此RHUH即使解锁也主要是方向/稳健性复现，不能预期单队列解决显著性。

它的科学价值很高：40位西班牙患者在术前、早期术后和复发各有一次多模态MRI，共120 scans；TCIA官方说明分割由DeepMedic初始生成，再由两位神经外科影像专家逐例人工复核修正。对于当前术后T1ce/Whole Tumor任务，可预先限定早期术后+复发两个时间点，即80 scans/40 subjects，标签`{1,2,3}`全部属于WT且没有切除腔标签4。

然而，TCIA官方只提供2.9GB IBM Aspera package684。本机没有`ascp`。虽然成功按IBM公开流程拿到临时public-link JWT，TCIA Faspex 5.0.15的scope兼容问题阻止了package详情和文件浏览；没有官方文件名、大小或hash时，不能用第三方镜像替代正式来源。

## 小白七问

1. **现在的问题：** UCSD75上GCSAM与Full972 KD分别为Scan `0.755647/0.745554`，差值CI跨0；还缺另一所医院的人工标签确认。
2. **一句话直觉：** 再找一所完全不同医院、由专家改过答案的考卷，比继续看同一份卷更能判断模型是否真会做题。
3. **为什么看RHUH：** 它来自独立西班牙医院、包含真实早期术后和复发时点、专家人工修正；UCSD的结果没有否定跨医院复现。LUMIERE只有RANO临床评价而非同口径四区体素真值，MU又是自动标签，因此RHUH更合适。
4. **具体怎么做：** 当前只审计来源。未来若获得官方inventory，先冻结80个post-op/recurrence T1ce+seg配对，做patient/header/content hash查重，再把同一两份冻结模型以阈值0、1×forward原样评一次。
5. **只改变什么：** 若未来执行，唯一变化是cohort；模型、128³ image-only brain crop、WT标签、threshold、无TTA/SAM/postprocess都不变。本轮没有模型动作。
6. **成功/失败：** 官方inventory、80/40计数、零重叠、文件hash和标签域全部冻结后才允许预注册；任何一门缺失就停止。当前因无官方inventory停止。
7. **代价：** 当前0 GPU、0 NIfTI、0 inference、0大文件。未来raw约2.9GB，80例cache约0.78GB、两个模型160 forwards；当前约12GB可用空间理论足够，但仍应先检查磁盘。

## 官方证据

- TCIA Version 1：40 subjects、120 studies、720个NIfTI image/seg files、2.9GB、CC BY 4.0。
- 预处理：T1ce配准SRI24，其他模态配准到T1ce，SynthStrip脑提取，CaPTk强度归一化。
- 标签：1 necrosis，2 peritumoral signal alteration，3 enhancing tumor；由DeepMedic生成后，两位expert neurosurgeons人工复核和修正全部分割。
- 三时点：preoperative、early postoperative、recurrence。正式T1ce/WT外部确认应在标签前固定只取后两者，不能看分数后选择。
- 一手来源：[TCIA RHUH-GBM collection](https://www.cancerimagingarchive.net/collection/rhuh-gbm/)、[Data in Brief数据论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10551826/)、[官方预处理代码](https://github.com/smcch/RHUH-GBM-dataset-MRI-preprocessing)。

## 官方访问诊断

- TCIA页面公开链接指向Faspex package684；本机`ascp`/Aspera Connect均不可用。
- Faspex公开配置：API prefix`/aspera/faspex/api/v5`，server response `X-IBM-Aspera: F5.0.15`。
- IBM 5.0.12 release note说明新版UI改用`GET /authorize_public_link`再交换bearer token。
- 新端点成功302并签发一次性authorization code；`POST /auth/token`成功，JWT有效期约12小时。
- JWT claim却为scope=`external_download_package_id`。`GET /api/v5/account`、`configuration`及`packages/684`均返回HTTP400/code1150：服务器要求scope必须是`external_download_package`等允许值。token交换额外请求正确scope也被忽略。
- 服务器公开OpenAPI仍列PasscodeAuth；直接`Passcode` header访问package详情返回HTTP403/code2009，说明5.0.15已禁用该端点的旧passcode访问。文档旧路径`/auth/authorize/public_link`返回404。
- 因此没有调用transfer spec、没有遍历文件、没有下载包。

## 患者级官方CSV

- `clinical_data_TCIA_RHUH-GBM.csv`：40 rows / 40 unique IDs，恰为`RHUH-0001…RHUH-0040`，SHA-256 `32d638906d34aaf8f66f5ec41c53c044216aed73bac22c776fb399bf2f741728`。
- `GC_manifest_RHUH-GBM_20260326.csv`：600 rows / 40 participants，每人精确15行，全部file type为DICOM，SHA-256 `5a70dc03e33ac651d1afc020e505351142b15228ad0583f9f12b72a19e4acd49`。这与3 timepoints×5 modalities一致。
- 两表40人ID集合exact；没有BraTS/UCSF ID字段。它们能冻结患者层清单，但GC manifest是受控DICOM series清单，不包含公开2.9GB NIfTI/seg的文件名或hash，所以不能替代Aspera inventory。
- 两个CSV合计223,696 bytes已归档在`source_metadata/`；没有NIfTI或segmentation voxel访问。
- 可复现入口`scripts/audit_rhuh_gbm_metadata.py`两项测试通过；机器审计`METADATA_AUDIT.json` SHA-256 `9ed80a1d5d454c5286678915f8e70e4695c27f61444b149891a1ad923ebbae7b`。

## 第三方镜像边界

Hugging Face的`MedOtter/RHUH-GBM`公开页声称提供未修改TCIA NIfTI、120例manifest和约3.1GB文件，但它不是TCIA官方分发端点，当前又没有官方逐文件hash可交叉验证。本轮仅把它作为“数据可能可达”的线索，不下载、不引用为正式证据。

固定commit `0104d76bfbbe089e8a0dd4e444feddfe97524c6b`的第三方API metadata列出724项，其中720个NIfTI路径；轻量`subjects_manifest.json`核对为40 patients×3 timepoints、每时点5模态+1 segmentation。预设后两时点恰为80 scans/40 subjects、160个T1ce/seg路径，均存在于tree；119例shape为240×240×155，`RHUH-0028/0`为230×230×138。后两时点的第三方label-presence metadata显示27例`{0,2}`、18例`{0,2,3}`、35例`{0,1,2,3}`，即80例均有WT，但这些信息绝不用于删病例。另有一个命名特例`RHUH-0035/2/segmentation.nii.gz`。

只归档了114,333 bytes第三方metadata：API JSON SHA-256 `a25a7ddbfb7b3c1c34ef179d650cef5f70be69355c3a2810d1c5b70e919db6f0`，subjects manifest SHA-256 `cf2e380d197ec28142d8ce8914bec49b661382e564c8791bc875b7484cffa107`。没有下载任何第三方NIfTI/seg。未来只有在TCIA官方inventory/hash可得并逐文件匹配后，这160个路径才可升级为正式下载计划。

## 下一决定

保持planned/blocked-on-source-inventory，不建立方法卡或榜单。可接受的解锁方式只有：

1. 用户提供已安装的IBM Aspera Connect/`ascp`环境并从package684拉取官方包；或
2. TCIA修复public JWT scope/API；或
3. 数据发布者提供官方逐文件manifest/hash，可据此验证第三方镜像。

解锁后仍必须重新写完整IDEA/PREREG和preaccess receipt，先证明80 scans/40 subjects及与现有1,350+UCSD75零重叠，才能访问标签或模型。
