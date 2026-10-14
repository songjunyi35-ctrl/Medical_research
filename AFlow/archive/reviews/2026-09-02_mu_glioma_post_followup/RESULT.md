# MU-Glioma-Post 外部确认可行性复审

日期：2026-09-02 UTC

## 结论

不下载11GB影像/自动标签包，不运行模型，不把MU-Glioma-Post加入正式External Leaderboard。

这不是因为队列没有价值，而是它当前不能提高本项目最缺的证据等级：官方论文与TCIA页面明确说明标签由Federated Tumor Segmentation工具预处理，并由基于nnU-Net的多个自动模型输出经STAPLE融合；论文还专门展示了FLAIR过分割、漏分割和把脑室误标成切除腔等错误。它适合做部署压力测试或弱标签研究，但不如已完成的UCSD人工修正/神经放射科批准标签适合作为真值确认。

## 小白解释

- **现在的问题：** UCSD75已给出一次患者隔离外部结果，但GCSAM相对Full972 KD的Dice置信区间仍跨0。直觉上想再找一所医院确认。
- **一句话直觉：** 第二张考卷若答案本身也是另一台机器猜的，就不能像专家批改卷那样强力证明谁真正答对。
- **为什么审计它：** MU有203名术后患者、约596个时间点、四模态与相同1/2/3/4标签，域和任务匹配度高；现有UCSD/内部结果并没有否定跨医院确认。
- **怎么审计：** 只读取TCIA页面、Scientific Data论文和两个约409KB官方元数据表；检查访问方式、ID、病例/时间点口径和标签来源，不接触NIfTI或模型。
- **改变与固定：** 没有模型实验；只增加来源审计。UCSD结果、模型、阈值和榜单均不变。
- **成功/失败：** 只有找到可逐文件下载、能在标签前冻结唯一病例清单、能证明与BraTS/UCSD患者无重叠、且标签证据强度足够，才允许另开预注册。任一项不满足就停止。
- **代价：** 0 GPU、0 inference；下载两个官方表409,321 bytes。11GB整包未下载，当前约12GB余量未被压满。

## 官方来源与访问

- TCIA Version 1页面列出203 subjects、2,978 files、11GB NIfTI images+segmentations，下载依赖IBM Aspera package1030。
- 官方`data-nih/tcia` GitHub releases索引没有MU-Glioma-Post条目，常见tag猜测均404；不像UCSD，当前没有已验证逐文件asset API。
- 官方直接表：
  - `MU-Glioma-Post_ClinicalData-July2025.xlsx`，101,568 bytes，SHA-256 `4341badf954511d67fc52b82b3b488ed7323f7e46d249a354ff9b3a62269c79b`。
  - `MU-Glioma-Post_Segmentation_Volumes.xlsx`，307,753 bytes，SHA-256 `4ba60e04279e91d920974f5641b313fd04fc2d0319f782f37567b48bc7795fa6`。
- `openpyxl`不在既有环境中；第一次Pandas读取在解析workbook前因缺依赖停止。没有安装包或改环境，随后用Python标准库`zipfile+xml.etree.ElementTree`只读解析成功。
- 可复现入口`scripts/audit_mu_glioma_post_metadata.py`不依赖`openpyxl`；3项测试通过。机器审计`METADATA_AUDIT.json` SHA-256 `eef56d84ec970cc13b374f460332d24cafc9b8eb3121c39ddeee63f9ec8f2e9a`。

## 元数据审计结果

- Clinical表工作表“MU Glioma Post”有335 data rows、203 unique Patient IDs。重复行对应同一患者信息重复/时间点表达。
- 6个MRI timepoint列共有597个非空记录，而TCIA/论文摘要为596 timepoints；在没有影像manifest时不能擅自决定哪一条应删除。
- Segmentation Volumes四张表只记录对应标签非空的行，不能直接充当完整扫描inventory：
  - Label1：334 rows，146 normalized patient bases；272 bare IDs、62带`Post-treatment`后缀。
  - Label2：614 rows，202 normalized patient bases；419 bare IDs、195带后缀；行数甚至高于官方596时间点。
  - Label3：569 rows，195 normalized patient bases；408 bare、161带后缀。
  - Label4：542 rows，189 normalized patient bases；368 bare、174带后缀。
- 表中ID是`PatientID_XXXX`，没有BraTS ID、UCSD ID或跨库患者bridge。不同机构让真实重叠概率低，但“看起来不同”不能替代可证明的patient/hash audit。
- 文件命名据论文为每患者/每时间点的`t1n/t1c/t2w/t2f`与seg；在获得Aspera manifest前，不能把597临床时间点或某一标签表行数当成冻结596例清单。

## 决策边界

当前同时失败三道开跑门：没有选择性asset manifest；597/596口径未解析；没有跨库患者bridge。即使以后解决这些门，标签仍是自动模型融合，只能标为`automated-label external stress test`，不得与UCSD人工批准External榜混排。

如果未来有足够磁盘和可审计Aspera manifest，可先做**不运行模型**的三步：冻结596个timepoint文件清单；用临床表解析597/596差异；用NIfTI/header/content fingerprint与现有1,350例及UCSD75查重。完成前不得下载整个11GB包或创建新Leaderboard行。

## 一手来源

- [TCIA MU-Glioma-Post collection](https://www.cancerimagingarchive.net/collection/mu-glioma-post/)
- [Scientific Data论文](https://www.nature.com/articles/s41597-025-06011-7)
