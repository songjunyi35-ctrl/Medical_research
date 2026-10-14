# Fable 8.5小时接力session记录

- start_utc: 2026-09-06T13:32:17Z
- deadline_utc: 2026-09-06T22:02:17Z（8小时30分钟 = 30,600秒，硬截止）
- 收口阶段起点: 2026-09-06T21:32Z（最后30分钟不再启动预计无法完成的长任务）

## 接手确认（13:34–13:40 UTC）

- 已完整阅读AGENTS.md、AFLOW_BRATS_HANDOFF_CURRENT.md、AFLOW_BRATS_LEADERBOARD.md、AFLOW_BRATS_METHOD_CATALOG.md、archive/reviews/2026-09-05_30h_chain_audit/SESSION.md与EXTERNAL_BLOCKER_REGISTER.md。
- 三个canonical验证器（doc layout / method catalog / leaderboard）在接手时全部passed。
- 全仓pytest已后台启动（结果待记录）。
- 两张A100空闲（0MiB/0%），无后台训练/评测/下载进程。
- 磁盘：根分区125G总量、可用仅1.4G（99%）。消耗主因：WAW官方preop/postop两个大包共6.1G（保留的原始来源）、MOTUM快照975M、RHUH来源431M、既有cache 46G。已知预授权可回收项：`workspace/BraTS/cache/t1ce_wt_n4_braincrop_resolution_20260818/formal_128`（7,927,508,320 bytes，2026-09-02审计证明为972 cache的756例byte-exact子集）；本session暂不删除，只有实际需要空间时按精确路径回收并记录。

## 本轮工作计划

1. 基线验证（pytest + validators）。
2. 外部阻塞登记表逐项复查：发布者侧可解除项（Africa2024新版、UTSW 3/5数据字典、UCSF-PDGM患者桥、UPENN官方manifest）当前状态。
3. 新公开数据源全面检索：寻找合法、公开、患者/机构隔离、含T1对比增强与专家人工修订WT标签的新外部队列。
4. 若发现合格候选：来源审计→七问→预注册→实现→synthetic→smoke→formal→归档→榜单全流程。
5. 无可运行实验时：证据一致性/哈希/文档审计。
6. 21:32 UTC起收口：更新handoff/catalog/（如有test）榜单，跑四条验证命令，终态中文报告。

## 执行日志

- 13:34 验证器3/3 passed；pytest后台启动。
- 13:40 磁盘审计完成（见上）。开始外部来源监测。

## 外部阻塞监测结果（13:42–13:55 UTC）

- pytest基线：`754 passed, 24 warnings`（36.82秒），与交接一致。
- Africa2024（阻塞#4）：TCIA BraTS-Africa仍只有Version 1（2024-09-04，146例）；未解除。
- UTSW数据字典（阻塞#3）：TCIA UTSW-Glioma仍只有Version 1（2026-03-18），无label 3/5定义；未解除。
- UCSF-PDGM桥（阻塞#5）：仍是Version 5（2025-05-30），只有BraTS21映射；未解除。
- UPENN manifest（阻塞#10）：TCIA UPENN-GBM仍是Version 2（2022-10-24），无逐文件NIfTI manifest/checksum；未解除。
- BRISC（新候选筛查）：6,000张对比增强T1为2D JPEG 512×512切片（axial/coronal/sagittal），非3D体数据，不能计算3D WT Dice；关闭，不下载。
- TCIA新闻页2026年6–9月无新成人胶质瘤集合；2026-02有BraTS-PEDs儿科分割资源（域/任务不匹配，低优先级，未进一步访问）。

## 新线索：UCSD-PTGBM BraTS-GLI 2024 Test Data已公开（13:55 UTC）

- TCIA UCSD-PTGBM版本历史：V1 2025-03-25（test data "coming soon"）；**V2 2026-01-22 "Released BraTS 'Test' data"**；V3 2026-03-13修正临床表patient ID错误（"Some patient ID were badly assigned in previous versions"）。
- 新公开包："BraTS-GLI 2024 Test Data"，11.88GB，MR+segmentation NIfTI，约42 subjects/59 studies/1,322 series，CC BY 4.0。
- 项目2026-09-02 UCSD75审计明确该包"未进入inventory、下载或评估"——因此是从未访问的新患者证据。
- 价值判断：治疗后GBM（与BraTS2024训练阶段语义一致）、UCSD-PTGBM标签为neuroradiologist-approved、患者按设计不在BraTS2024 training（挑战test集），且UCSD有官方BraTS ID映射可逐患者排重（不同于UCSF-PDGM的缺桥）。身份定位：public post-challenge、非hidden blind、同机构此前已有UCSD75评测但患者不同（需在像素前证明与1,350例union和已评UCSD75/136 main release患者零重叠）。
- 风险/核查点：V3患者ID修正对映射的影响；标签值域（BraTS2024 post-treatment {0,1,2,3,4}?）；每study是否有T1CE+seg；官方checksum；磁盘（需先按预授权回收7.93GB冗余cache）。

## 来源审计与磁盘回收（14:00–14:25 UTC）

- package 1221只读审计完成：60目录+1,322 symbolic_link文件、59 studies/42 subjects、每study有T1post+BraTS_tumor_seg、无.sums。官方v3临床表桥审计：仅UCSD-PTGBM-0038（BraTS-GLI-02289∈union）重叠，且全表唯一跨ID歧义也在0038；排除后冻结41 subjects/56 scans/112文件/312,277,333 bytes。证据archive/reviews/2026-09-06_ucsd_brats2024test_source_audit/（RESULT.md、SOURCE_AUDIT_V2.json SHA=9fbc2467…、PACKAGE1221_INVENTORY.json SHA=ec78b96c…、FROZEN_COHORT.json SHA=aed4cfcb…）。V1 SOURCE_AUDIT.json因只计type=="file"报0文件，保留并由V2更正（RHUH式symbolic_link交付）。
- 七问IDEA.md（SHA=73709613…）与PREREG.md（SHA=acefe705…）已在任何像素访问前冻结于archive/research_records/2026-09-06_ucsd_brats2024test_frozen_panel/。
- 磁盘回收：按用户既有授权（handoff"磁盘不足时允许自主删除已审计、可重建、已哈希归档的派生缓存"），在复核2026-09-02审计的两个index SHA（superset 7373cb30…、subset d26b823a…均exact）与972例superset完好后，删除精确目录workspace/BraTS/cache/t1ce_wt_n4_braincrop_resolution_20260818/formal_128/（7,927,512,416 bytes）。可用空间1.4G→8.8G。该cache可由raw NIfTI与归档builder重建；原始数据、checkpoint、代码、报告均未动。

## UCSD-BraTS2024test56冻结面板完成（14:00–14:15 UTC）

- 双传输59.6秒112文件全exact→标签域扫描passed（无空WT）→首例probe→cache 325.7秒全门过（零crop丢失56/56）→smoke→preaccess→唯一504 forwards（10.15秒/5.07GiB）。
- 结果：Full Flip8/identity/GCSAM Scan=`0.732973/0.730002/0.717689`，Subject=`0.733625/0.730740/0.715545`，Point=`56/56、56/56、52/56`。Flip8−identity=`+0.002971`跨0；GCSAM−Full=`−0.012312`，scan/subject配对CI首次整体为负、Point−4。
- 终态验证passed（112源+2 checkpoint+3 cache文件重hash exact；三绝对+两配对全统计独立重算exact）。
- 归档/榜单/百科/handoff全部更新；leaderboard validator新增`ucsd_brats2024test_external_rows:3`通过；doc layout 495链接通过；catalog通过。
- 全仓pytest：既有native-sampling测试假定已删cache存在，按上轮WAW删除先例改为"cache在则全比较、被授权删除则验证冗余审计+972 superset"，冻结权重SHA断言保留；最终`760 passed`（新增6项本面板合同测试）。

## 事后审计与跨cohort综合刷新（14:12–14:22 UTC）

- POSTHOC_ERROR_AUDIT：GCSAM−Full precision=`−0.02263`（事后CI全负）、recall近零——近域队列中GCSAM扩张倾向纯损失precision；Flip8 precision=`+0.01111` CI全正、预测体素显著减少（保守去假阳性机制5/5外部复现）。本队列WT中位仅居train972分位48.35%（Africa 95.47%），支持"GCSAM优势只在远超训练支持的大目标域"的coverage解释。
- POSTHOC_POINT_CATEGORIES：44/56例含RC4，Full identity/Flip8 argmax 56/56全落WT、0落RC（对照WAW-post31 81%落RC）；RC吸引是域偏移放大的失败模式而非普遍规律；GCSAM 4 miss全为小WT病例。
- 跨cohort综合刷新：GCSAM V5（七cohort，唯一CI全负新队列加入，JSON/PNG SHA=19d842c4…/4776ac1b…）；Flip8 V4（六cohort点估计全正、三CI全正，SHA=55aa9784…/b1871cf8…）；evidence matrix V3（七cohort/712 scans，SHA=092324aa…前16位）；claim registry CLAIMS_V2.json（C1/C3更新、新增C13）。
- 检索：其他BraTS2024来源站未发现同类test份额公开；MU-Glioma-Post仍自动标签。

## 后续审计（14:18–14:26 UTC）

- 样本量敏感性V3：新增近域subject级SD场景（Flip8 SD=0.01915→检出+0.005/+0.01约需116/29人；GCSAM SD=0.05231→+0.01/+0.02约需215/54人）；单cohort不可移植。RESULT_V3.json（sha前16=6f54be5c…）。
- GCSAM coverage综合V3：六外部cohort recall/预测体积点估计6/6正、precision仅2/6正、Point从未增加；近域队列precision差事后CI全负。RESULT_V3.json（sha前16=2ed708af…）。
- 标签谱系一手论文补记：全部UCSD-PTGBM分割为自动→放射科医生修正→神经放射科医师批准；论文"40人/60时点/2025 challenge"与包实际"42 subjects/59 studies/BraTS-2024-test-set"的计数差异已显式保留、不猜测调和。
- RC supervision gap对照补记：Full argmax在新队列0/56落RC（对照WAW-post31约81% miss落RC）→RC吸引主要是域偏移放大的失败模式；显式RC head优先级下调。
- 榜单MD↔JSON新分节全数值交叉核对ok；doc layout/催化validator全过。
- 来源监测追加：RHUH仍V1；OpenNeuro近两月无胶质瘤新集；其他BraTS2024站点无test份额公开。
- 阻塞登记表加2026-09-06补记（package1221已消费关闭；四发布者侧阻塞均未解除）。

## BraTS-PEDs来源审计（14:26–14:33 UTC）

- Package 1217完整分页inventory：348例（Training257有seg/Validation91无）、1,649文件/35.17GB、无.sums、348全部唯一subject；候选T1c+seg约6.8GB。
- 标签合格（专家迭代修订、字典ET1/NET2/CC3/ED4、treatment-naive、CC BY 4.0）；但官方明确not skull-stripped（仅defaced）、无模板/1mm标准化——canonical brain-crop输入链前提失败（训练与既往七cohort全为skull-stripped）。按RHUH/MOTUM先例在模型前关闭，0像素/0模型；登记为阻塞#14。儿科官方WT不含ED的语义分歧也已记录（未来须冻结all-positive目标）。

## 继续候选筛查与收尾质量（14:33–14:35 UTC）

- DFCI-BCH-BWH-PEDs-HGG：61例儿科HGG，但官方页明确"expert segmentations will be released as part of a future publication"（当前无标签）且同样未去颅骨；metadata即关闭，不建审计目录。
- RSNA ryai.250902论文核实为BraTS-PEDs同一数据集论文，无新来源。
- 已把/tmp cache的index.json归档为record内CACHE_INDEX_FINAL.json（SHA与receipt一致）；/tmp cache与copy B丢失时可由workspace copy A+归档builder重建。
- 新增0-forward时点组成描述POSTHOC_TIMEPOINT_DESCRIPTION.json（后缀100 vs 101/102 Dice约0.70 vs 0.78，仅组成描述）。

## 文献边界与候选清扫（14:35–14:55 UTC）

- GBM-Reservoir（Data in Brief 2025）：BraTS2022+TCGA+UPenn重打包/配准增广，非新患者来源，与训练血缘重叠；关闭，不下载。
- LITERATURE_CONTEXT.md：为Flip8"保守去假阳性"（细胞TTA一手先例）与GCSAM近/远域方向可逆（SAM-OOD文献本就混合/条件性）提供一手上下文；不改变数字、不授权调参。

## 定时安排（14:41 UTC设定）

- 新归档目录已写FINAL_SHA256SUMS.txt（panel 17项 / 来源审计6项 / PEDs审计3项）。
- persistent monitor已设UTC检查点16:30/18:30/20:15（周期复核：validators、GPU/进程、发布者页抽查）与21:25（收口触发）；deadline 22:02:17后不启动新命令。
- 当前无后台科研进程；两张A100空闲；磁盘8.5G可用。

## 16:30 UTC周期复核

- 系统：GPU双卡0MiB/0%、无后台科研进程、磁盘8.5G可用。
- Validators：leaderboard/doc layout passed。
- 发布者抽查：BraTS-Africa仍仅Version 1（2024-09-04），阻塞#4未解除。
- 无新增可运行实验；继续等待下一检查点（18:30 UTC）。

## 18:30 UTC周期复核

- 系统：GPU双卡空闲、无后台科研进程、磁盘8.5G。
- 全仓pytest：`760 passed`。
- 发布者抽查：UTSW-Glioma仍仅Version 1、无3/5数据字典（阻塞#3未解除）。
- 无新增可运行实验；等待20:15检查点。

## 20:15 UTC周期复核

- 系统：GPU双卡空闲、无后台科研进程、磁盘8.5G；catalog validator passed。
- UCSD-PTGBM确认仍为V3（2026-03-13），本轮评测所依据的版本状态无变化。
- 无新增可运行实验；21:25进入收口。

## 收口（21:25–22:02 UTC）

- Monitor定时器自然结束；无其他后台任务。
- handoff终态段已写（起止13:32:17–22:02:17 UTC、连续8.5小时、无后台进程、/tmp派生物可重建）。
- 最终验证命令输出见下方记录；deadline后不启动新命令。

本session完成清单：UCSD-BraTS2024test56全流程外部评测（完成并关闭）＋7项事后/综合刷新（error/point/timepoint posthoc、GCSAM V5、Flip8 V4、matrix V3、coverage V3、claims V2、样本量V3）＋3项新来源审计（BraTS-PEDs关闭#14、DFCI无标签、GBM-Reservoir重打包）＋4项发布者阻塞复查（均未解除）＋磁盘回收7.9GB＋文献边界备注＋榜单/百科/handoff/登记表全套更新。

最终验证（21:2x UTC）：pytest `760 passed, 24 warnings`；doc layout passed（9 root MD/497链接）；catalog passed（17卡）；leaderboard passed（含新分节ucsd_brats2024test_external_rows=3）。GPU双卡0MiB/0%；无后台科研/下载进程。session按期结束。
