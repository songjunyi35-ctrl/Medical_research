# 两项复用开发实验的收据链复核

状态：通过。辅助监督4条训练、患者采样2条新训练，全部固定168轮/121步、无教师、无开发指标读取，初值与各自STARTED一致、导出误差0。两份ARCHIVE_SEAL共44条文件条目重新SHA匹配；其中复用control文件被两项归档引用，不代表44个唯一文件。

训练前收据绑定实现冻结、合成检查和缓存收据；每份STARTED绑定该训练前收据且记录时间在其后。开发前收据绑定最终checkpoint（辅助还绑定四TRAIN_DONE SHA，采样绑定源control收据），raw绑定开发前收据，RESULT/VERIFICATION/CHAIN_DONE互相核对相应哈希。两项结果均完成，但联合效果门均false；“链路通过”不等于“方法晋级”。

两个采样control逐例行与源实验原记录完全相同，未重复控制推理；患者采样IDEA/PREREG仍与源dev揭示前DESIGN_FREEZE的哈希一致，记录中没有按前一效果决定是否修改配方。两项新实验的保护队列访问字段均false。

完整168轮抽样核查及两份点标签核查状态均通过，其文件SHA也纳入机器见证。没有新增forward或标签数组读取；本次只读收据/报告及已归档checkpoint字节作SHA。

范围限制：本地时间戳和记录不是第三方公证时间；SHA保证当前内容与所存摘要一致，不能证明原始医学标注正确；没有独立重放模型前向；本报告不证明30小时已经用满。Goal时长和最终文档还需另行核验。

证据：[机器见证](RECEIPT_LINKS.json)、[辅助归档](../../research_records/2026-09-07_structured_background_aux/ARCHIVE_SEAL.json)、[采样归档](../../research_records/2026-09-07_subject_balanced_sampling/ARCHIVE_SEAL.json)。
