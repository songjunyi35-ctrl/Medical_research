# 正式训练首完整轮核验

北京时间2026-10-29T15:23:22.622763+08:00，RACE-v2已完成新增第1轮、122次真实更新，原972例各一次、108完整验证、full latest CPU严格重载及AdamW计数、双rank参数SHA一致通过。冻结新增80轮／9760更新，继续原配方；有效batch8／每轮122更新。单T1ce、原生1mm、原972/108/270不变。

首轮Scan0.756934833、Subject0.745170641、Point98/108；目前低于e164，只是早期结果，不构成最终收益结论，也不据此调参或扩组。源e164 best没有历史optimizer，本次新AdamW不是无缝续训。

主节点/tmp已实际收到full latest与best并逐SHA核验；latest optimizer步数122，best若为e0只保留本次新optimizer的空初始状态，不冒充历史动量。每个复制先固定worker原子checkpoint inode，避免读取覆盖中的latest。推理仍只接完整预算终态后的固定CPU严格重载快照。

[启动机器证据](../2026-10-29_race_v2_training/STARTUP_VERIFIED.json) · [实际备份](../2026-10-29_race_v2_training/RECOVERY_BACKUP.json) · [预算](../2026-10-29_race_v2_training/BUDGET_DECISION.json) · [执行链](EXECUTION_CHAIN.md) · [首阶段108结果](INITIAL_STAGE_RESULT.md)
