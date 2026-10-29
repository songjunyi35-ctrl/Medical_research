# RACE-v2训练队列交接

本阶段是新授权：只使用worker1154259的GPU0、GPU1作本机双卡DDP，禁止使用推理worker的GPU。总控新期限见`../2026-10-29_race_v2_stage/AUTHORIZATION.json`；保守初始24h，不预先支用条件96h。当前状态看`PIPELINE_STATUS.json`，不要按本文件的启动时刻推测后续状态。

## 数据与数值

worker/tmp独立；本节点初始没有训练/验证载荷。`/tmp/aflow_race_v2_20261029/native/{train,validation}`由原数值builder重新生成，validation优先。每split的SHA、实际bytes、case顺序、crop/offset/reflection/reference_enabled必须与历史一致；`PAYLOAD_AUDIT_*.json`为第二遍实际读取审计。972另核历史classes.u8 SHA；不把index存在当payload完备。全部数组由只读Dataset打开。

`CPU_CHECKS.json`和最终`SMOKE_DDP.json`已检验gate收益目标、参数兼容、bf16原部署公式逐位相同、5次合成更新含尾4、两rank参数SHA、CPU严格重载与<=38GiB/卡。初版/诊断修订均保留engineering，不把合成通过称患者训练。

`DIAGNOSTIC_RESULT.md`报告108正常/base/AP错配三路。V1概率残差存在dtype统计问题，V2按两侧fp32修正；累计216次病例主干前向、432参照枝路、参数更新0。首次诊断缺独立prelaunch源码冻结如实补记；正式训练之前另需全新执行冻结。

## 有限执行

新入口为`python -m scripts.race_v2_20261029.coordinator`，依次等待/审计验证载荷、诊断、972载荷、按完整合成吞吐与新剩余时间冻结最多80新增轮、再启动`launch train`。不运行任何旧协调器。训练每轮972各一次、global8/微批2×每rank本地4/122更新，原始0.5阈值108每轮选优，baseline新增0轮可入选。e164 best没有optimizer，本次重新建AdamW；不是无缝续训。

完整合成稳态约1.282s/update，冻结预算用`(122*实测速度+60s验证)*1.5`每轮及45min额外余量，最多80轮约7.22h。`RECIPE.json`为预定配方：LR2e-5，新增前10轮恒定后cosine到2e-6，.05参照收益校准、.002收益死区，继承.05类型对比。训练更新前的实际轮数与source SHA见`BUDGET_DECISION.json`、`EXECUTION_FREEZE.json`。

每次GPU队列全程持有总控canonical两个fcntl锁。每轮保存完整optimizer latest到worker/tmp/run，best也在此。每epoch从确定seed重新构造global972顺序和两rank RNG；恢复仅从完整epoch、配方不变、最多2次工程恢复。首完整真实epoch需`STARTUP_VERIFIED.json`；该分数只是早期结果。最后checkpoint后若仅出版失败，只恢复统计/出版，不重复更新。

## 恢复与推理交接

主节点`/tmp/aflow_race_v2_training_backup_20261029/`异步备份watch已启动。它先在worker将原子checkpoint inode硬链接为固定scratch快照，再SCP到主节点临时文件、核SHA/bytes和optimizer步数，最后原子替换主节点best/latest。实际完成收据看`RECOVERY_BACKUP.json`，仅启动watch不代表已有有效备份。

仅完整预算终态后导出final best一个固定推理快照；不在40轮增加模型推理或270访问。`HANDOFF.json`含构造参数、固定路径/SHA、严格CPU重载、入选best新增轮/更新、完整阶段新增轮/更新、code/recipe冻结、108逐例结果、跨split画布和参照矩阵共轭约定。A/B须等总控登记后读取固定快照，不得读正在覆盖latest。训练agent不执行270；最终候选仍由总控限制至最多2项。

本目录含文本和少量纯模型权重累计<=0.50GiB；全阶段新增<=0.75GiB，并始终保留持久空闲>=20GiB。大数组/完整optimizer只放worker/tmp和主节点/tmp；不删除历史权重或数据。
