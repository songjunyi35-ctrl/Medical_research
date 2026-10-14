# UPENN-GBM官方来源审计

状态：source-only metadata audit，未下载影像/标签、未运行模型，尚未分配train/inner/confirmation/test角色。

动机：当前严格排除已揭示内部评测患者后只剩38人87次扫描，其中E/F84次已冻结用于no-N4。需要新的患者证据来源，不能持续给旧划分改名。RHUH来源已恢复但原N4输入格式兼容门失败；UCSD75已经关闭且只代表8位镜像输入。

本次只核验TCIA官方UPENN-GBM患者/时间点/文件清单、标签审核来源与身份重叠风险。论文记录611次基线扫描及60次再次手术前随访，60例中19例仅有随访；新患者身份与最终文件配对必须实证冻结。不能仅因病例名称不同就称独立，也不预先把第三方整理镜像视为官方原始文件。

允许当前动作：下载官方小CSV元数据、只读Faspex公开package604目录及校验和。package938用途需先核对页面，不盲目下载。任何影像/标签下载、缓存或模型评测须另行明确数据角色和预访问协议。

来源：[TCIA collection](https://www.cancerimagingarchive.net/collection/upenn-gbm/)、[Scientific Data原始论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9338035/)。
