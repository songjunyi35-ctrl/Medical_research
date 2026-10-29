# 推理B交接

108有限实验已完整完成：12统一融合配置、两模型各8视图，实测最高0.769554565（RACE0.75/w32 0.25、logit、原始0.5），Subject0.759239295、Point98/108。最强单模0.768011881；五折合并0.768177869，稳定优势未确认。全部历史test-tuned与复用验证属性保留，无本阶段新270评分或270病例前向。

独立进程统计恢复未重做864 w32前向；源码run_v2隔离阶段，旧失败run和所有冻结保留。当前数组仅1154121/tmp只读，没有跨目的地备份。恢复/丢失边界和额外成本见REBUILD_REGISTER。

已冻结final-only伙伴/12配方入口：每折仅四折重新选1伙伴和配方，最多24缓存比较，最终部署2模型16前向。root终态注册后由snapshot watcher PID2529接A新864 atoms，不新增网络；best0不重复RACE164。

final270入口全尺寸168纯合成batch8/尾6通过，峰值29.6875GiB。final watcher PID2442仅等真正root FINAL_TEST_SELECTION；必要时w32一次2160，RACE只读A，固定配方不test调参。独立保存掩膜审计后发布final270/RAW_FUSION.json。最终选择尚不存在，两个watcher均安静依赖等待。

模型、输入/原子SHA及实验结果分别见ATOMS_READY、EXECUTION_FREEZE_V3、RESULT/RAW_VALIDATION；快照规则见SNAPSHOT_PLAN/ENTRY_FREEZE；final接口见FINAL_ENTRY.md、FINAL270_ENTRY_FREEZE。
