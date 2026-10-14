# 当前训练池额外MRI序列：本机未保留

状态：只读文件名与历史manifest审计完成，0影像/标签数组、0模型、0新开发指标；不改变当前辅助监督实验。

在已冻结的684次fit扫描、313名患者目录中，逐例检查实际文件名：T1对比增强（t1c）684份，T1平扫（t1n）、T2（t2w）、FLAIR液体衰减反转恢复（t2f）均0份；每个目录只含当前T1c与seg两文件，其他文件0。

本机`/mnt/hdfs/sjy/datasets`当前只列出`brats2024_gli_t1ce_seg_v1`。历史train manifest确实记录了四模态，原路径为`/mnt/bn/liveaim/songjunyi/brats2024/...`；但本轮检查发现该路径及songjunyi子目录在本机不存在。不能据此说远端原始四模态已被删除，只能说当前挂载位置不可用。

历史来源状态记录1350 scans/613 subjects/6750文件，指当时Kaggle v2三包下载状态；它既不能证明本机现在有这些文件，也不能证明与官方Synapse corrected-v2逐字节一致。

结论：若未来研究“训练期教师利用更多MRI序列、部署学生仍只看T1c”，必须先恢复合法可用的同患者额外序列，并在fit范围证明T1c/seg来源parity、几何对齐和教师隔离。当前没有这些条件，不启动该新方法、不用合成FLAIR充当真实数据，也不把目录里不存在的文件写入训练配置。

该审计只排除了“直接利用本机已有四模态即可开跑”的假设，不否定更多模态的价值。当前两组切除腔辅助监督只需现有T1c和seg，仍正常运行。

证据：[机器清单](RESULT.json)、历史`data/datasets/brats2024_random_subject_80_20_seed20260722_v1/train_cases.jsonl`、`workspace/BraTS/training_protocols/brats2024_source_provenance_status_v1.json`。只检查fit目录；未枚举dev目录或读任何像素。
