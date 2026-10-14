# BraTS-Africa官方来源审计结果

状态：✅ TCIA package948目录/队列身份完成；后续冻结baseline评测已另行预注册并完成。

## 官方数据到底是什么

TCIA Version 1 package948为1,727,632,847 bytes，共146例，每例恰有T1、T1CE、T2、T2-FLAIR和seg五个正大小NIfTI，合计730文件。全部路径无官方checksum，因此后续仅下载冻结T1CE+seg，并用两份独立官方传输逐文件SHA一致替代发布者checksum。

发布者目录已经把任务分开：

- `95_Glioma`：95例×5文件；唯一T1CE+seg共190文件、277,630,584 bytes/副本；
- `51_OtherNeoplasms`：51例×5文件，含脑膜瘤、转移瘤等，像素前永久排除，不混入成人弥漫性胶质瘤榜。

这95例对应2023 BraTS-Africa的60 train + 15 validation + 20 test；TCIA当前包已经公开95份seg。本项目不在其中训练/选参，合并为`public post-challenge external95`，不是服务器隐藏blind test。

## 标签与机构证据

一手论文和TCIA页面说明：影像来自尼日利亚1.5T常规临床MRI，按BraTS流程做N4、SRI配准、1mm重采样和脑提取；nnU-Net先预标，最终由有经验的board-certified neuroradiologist人工修订。95例seg全量值域最终为`{0,1,2,3}`，每例WT非空。

来源中心数存在层级差异：challenge论文描述更大收集网络为七中心，TCIA页面写六中心，package datainfo实际映射当前146例仅五中心，实际`95_Glioma`只含四中心：NKDC48、CRV28、MEDHUB14、LASUTH5。最终报告以95例逐ID datainfo为准；原raw里“six centers”保留并通过显式更正文件纠正。

这四个尼日利亚中心不在BraTS2024 post-treatment的七个欧美来源站内，故称institution-disjoint/no-known overlap；没有跨库patient ID bridge，不能声称ID级绝对证明patient-disjoint。

## 审计实现失败

三次失败均保留且均在payload前：首次根目录后TLS EOF；第二次8并发列146病例时TLS错误；限制每页3次短重试并降4并发后完整网络列表成功，但报告端字典缺sort key再次失败。只修连接并发/重试和按path排序，最终列出150目录/879 entries/730 files。另一次cohort freezer把Python `False`写成JSON `false`，也在输出/payload前停止后修复。

这些是实现/传输失败，不是数据失败；没有用失败过程筛病例。

## 后续访问与结果

冻结95例后，双源190/190 exact；首例与全量cache工程门完成，两个冻结模型按预注册一次评测。来源审计本身不使用分数选择队列；完整指标见[方法结果](../../research_records/2026-09-06_brats_africa95_frozen_baselines/RESULT.md)。

## 证据

- [冻结计划](PLAN.md)
- [完整官方inventory](OFFICIAL_INVENTORY.json)，SHA-256 `16f6eee4fe258e9e75a69a2bda75c7f3f5a00365ab833308bf1e1943fbcee45b`
- [目录机器报告](SOURCE_AUDIT.json)，SHA-256 `ae835e9304e75906ecbf55bc6ede6fc7c56831b50dae51530b386163b9805c8f`
- [冻结Glioma95](FROZEN_GLIOMA95_COHORT.json)，SHA-256 `a2ce66d546b43e3bc03242a51827a155c7601956b1af52eefddb4fe99a72c449`
- [中心来源更正](../../research_records/2026-09-06_brats_africa95_frozen_baselines/CENTER_PROVENANCE_CORRECTION.json)
- [TCIA collection](https://www.cancerimagingarchive.net/collection/brats-africa/)、[challenge paper](https://doi.org/10.1093/noajnl/vdag082)、[dataset paper](https://doi.org/10.1148/ryai.240528)
