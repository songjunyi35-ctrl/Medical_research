# UCSD-PTGBM「BraTS-GLI 2024 Test Data」（package 1221）来源可行性审计

日期：2026-09-06（Fable 8.5h session）。只读metadata审计，未下载任何影像/标签像素，未运行模型。

## 发现

- TCIA UCSD-PTGBM集合版本历史：V1 2025-03-25（test data标注"coming soon"）；**V2 2026-01-22发布"BraTS 'Test' data"**；V3 2026-03-13修正临床表患者ID（官方说明"Some patient ID were badly assigned in previous versions"）。集合页SHA-256=`c4437703250871e018af4da6c547ba9f60932a738fd028e90637508d45a3e4d6`。
- 项目2026-09-02 UCSD75外部确认明确排除了该包（当时ID已知为1221，"未进入inventory、下载或评估"）。本审计是首次访问其文件清单。
- 包标题`UCSD-PTGBM-BraTS-2024-test-set`；完整分页遍历得60目录+1,322个`symbolic_link`数据文件（与RHUH package684一样按符号链接交付，`total_bytes=117669`不可用）。首版`SOURCE_AUDIT.json`只统计`type=="file"`而报0文件，已由`SOURCE_AUDIT_V2.json`更正并保留原件。
- 59个study目录/42 subjects（`UCSD-PTGBM-XXXX_YY`命名），**每个study均有`T1post.nii.gz`与`BraTS_tumor_seg.nii.gz`**，另有约21种其他序列/参数图（本项目不使用）。
- 包内**没有`.sums`官方逐文件校验清单**（不同于main release package1019）。传输完整性门只能用双独立传输逐文件SHA-256一致（Africa95/MOTUM先例）。

## 患者桥审计（官方v3临床表，243行）

- 59个study全部在v3表中有非空BraTS case ID；timepoint后缀分布100/101/102=42/12/5。
- 逐患者any-timepoint规则（与2026-09-02相同）：42人中**仅`UCSD-PTGBM-0038`**的BraTS时间点（`BraTS-GLI-02289`）出现在项目1,350例union；该患者同时是先前main release 136人之一。
- 全表唯一跨ID异常也恰好是0038：`BraTS-GLI-02289`同时映射`UCSD-PTGBM-0036/0038`，0038又映射`BraTS-GLI-02289/02291`（v3所修正的错误分配类型）。排除0038即消除唯一歧义。
- 其余41人与union（613 subjects）、先前已评UCSD75（53 subjects）及main release（136 subjects+映射保守并集137）零重叠。union manifest SHA与2026-09-02归档值逐一核对一致。

## 冻结队列（像素前）

- **41 subjects / 56 scans**：package1221除0038三个study外的全部study；每study取`T1post`+`BraTS_tumor_seg`共112文件、312,277,333 bytes。
- 身份定位：public post-challenge（BraTS-GLI 2024隐藏test的UCSD部分公开发布）、**非hidden blind**；与先前UCSD75同机构、不同患者（patient-disjoint by official mapping）；UCSD是BraTS2024七个来源站之一，故**不是institution-disjoint**，必须独立分榜，不与UCSD75/internal混排。
- 阶段语义：治疗后GBM，标签为BraTS 2024 post-treatment约定，与训练WT=`{1,2,3}`排除RC=4完全同口径——这是当前所有外部队列中阶段/标签语义与训练最接近的一个，直接补WAW-postop31之后的治疗后证据缺口。

## 下一步（须另行七问+预注册后才可访问像素）

1. 双独立传输112文件、逐文件SHA一致门。
2. 逐例raw label域门（⊆{0,1,2,3,4}）、geometry、WT非空、零WT crop丢失门。
3. 冻结面板：Full972 KD168 identity+固定Flip8、GCSAM184 identity，阈值0，10k患者簇bootstrap，一次性评测、无promotion门。

## 证据

- [SOURCE_AUDIT.json](SOURCE_AUDIT.json)（V1，0文件计数错误，保留）
- [SOURCE_AUDIT_V2.json](SOURCE_AUDIT_V2.json)（SHA-256=`9fbc2467e84bbf536676f056ffd581c4807b259453a47cb9e49692c9b06fbbf4`）
- [PACKAGE1221_INVENTORY.json](PACKAGE1221_INVENTORY.json)（SHA-256=`ec78b96cd7b5f0eee5a45441ee9ae8b38177f9cc42b040e0dee7109ff272d4a8`）
- [PATIENT_BRIDGE_LOCAL.json](PATIENT_BRIDGE_LOCAL.json)
- 官方v3临床表：[../2026-09-05_30h_chain_audit/source_metadata/UCSD_PTGBM-clinical-information_v3_2026-12-Mar.xlsx](../2026-09-05_30h_chain_audit/source_metadata/UCSD_PTGBM-clinical-information_v3_2026-12-Mar.xlsx)（SHA-256=`bbf40bbdd289266924869052806de4d7e8fc9ee73286cdc0cc066672ef5bb51d`）

## 标签谱系一手论文补记（2026-09-06晚，评测后只读）

Scientific Data论文（s41597-025-06499-z）描述全部UCSD-PTGBM分割流程为：OnQ Neuro自动分割→受训放射科医生逐例人工修正→board-certified neuroradiologist "approver"复核批准（迭代直到达标），未对保留test份额另设流程——支持本队列标签与main release同为专家批准级。**计数/措辞差异需保留：** 论文写"40 patients (60 timepoints) held out for the 2025 BraTS challenge, released early 2026"，而TCIA包1221实际为42 subjects/59 studies且命名"BraTS-2024-test-set"、v3映射后缀为BraTS-GLI 2024格式。以包inventory与官方版本说明为准（论文成文早于发布），不猜测调和；该差异不影响患者排重（基于官方v3逐study映射）。
