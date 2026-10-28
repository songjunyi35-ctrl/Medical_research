# RACE-Seg实验接续与状态核对

本轮用户授权双A10040GB、24小时自主运行。截止时间由[AUTHORIZATION.json](AUTHORIZATION.json)的deadline_unix控制；后续交互不自动重置预算。主状态见[PIPELINE_STATUS.json](PIPELINE_STATUS.json)。

## 节点与进程

已核实worker1152470，两张A100-SXM4-40GB。连接地址以`mlx worker list`为准；2026年10月28日核实的地址为`fdbd:dccd:cde2:100b:7888:88ec:133c:25a5`、端口11239、用户tiger。SSH主机记录为主节点`/tmp/aflow_race_20261028_known_hosts`。

代码与持久记录位于本仓库；缓存、优化器与日志位于worker的`/tmp/aflow_race_20261028`。主节点与worker的`/tmp`不是同一位置。原数据准备PID1015，新协调器PID1184；应先核对当前命令行，不能只凭历史PID判断存活或执行kill。

## 当前流程

1. `scripts.race_seg_20261028.prepare`每例N4一次，同时导出128³与原生数据；逐SHA匹配历史后写DATA_READY。
2. `scripts.race_seg_20261028.coordinator`等待数据门，通过合成吞吐和剩余时间冻结共同轮数及微批。
3. control使用GPU0，race使用GPU1；正式PID写入TRAIN_PIDS。每组日志、HEARTBEAT、STATUS、BEST与逐轮108报告分别归档。
4. 完成后自动运行finalize，报告患者配对区间、固定收益门、权重与报告SHA，并更新根文档、运行四项文档校验。若发生异常，应保留真实未完成状态。

当前完整代码已推送到Medical_research的提交`3306992ca9ece9e2b6faf5b4bc10dacd4b88be12`；本次提交存档实现与合成证据，不包含未来患者训练结果。8项CPU测试、完整双卡合成更新、尾批与严格重载已通过；最终修正保证当前特征与参照使用同一value投影空间，辅助随机采样不改变骨干dropout随机数。

## 安全接续边界

先读PREPARE_HEARTBEAT、DATA_READY、PIPELINE_STATUS及两组直接HEARTBEAT/STATUS，确认真实阶段。不得把准备阶段当作患者训练，不得重启旧width、distill或nnU-Net协调器。

训练最多各200轮，共同实际轮数由BUDGET_DECISION决定，不能因分数延长或增加候选；每组工程恢复最多2次，只从完整epoch的model/optimizer接续。缓存准备已启动，禁止直接重跑prepare覆盖或重复N4；准备失败应先检查原始失败记录与已分配载荷。

本轮无新test270/outer/confirmation/外部模型访问。原972/108及患者隔离、原生载荷SHA、WT标签1/2/3排除4、阈值0.5固定。持久新增<=4GiB、空闲>=20GiB；大载荷仅worker临时盘。
