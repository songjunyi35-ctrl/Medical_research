# RACE-Seg实验接续与状态核对

本轮用户授权双A10040GB、24小时自主运行。截止时间由[AUTHORIZATION.json](AUTHORIZATION.json)的deadline_unix控制；后续交互不自动重置预算。主状态见[PIPELINE_STATUS.json](PIPELINE_STATUS.json)。

**当前优先：用户取消新对照，改为双卡共同训练RACE-Seg，累计200轮。** 原两组分卡与两组200轮迁移已被覆盖。新对照已完整4轮/488更新停止，禁止自动恢复；下文原两组流程与173轮启动均为保留历史。实际切换状态见[DDP_MIGRATION.json](DDP_MIGRATION.json)，双卡计划与执行冻结见[DDP200_PLAN.json](DDP200_PLAN.json)、[DDP200_FREEZE.json](DDP200_FREEZE.json)。

## 节点与进程

已核实worker1152470，两张A100-SXM4-40GB。连接地址以`mlx worker list`为准；2026年10月28日核实的地址为`fdbd:dccd:cde2:100b:7888:88ec:133c:25a5`、端口11239、用户tiger。SSH主机记录为主节点`/tmp/aflow_race_20261028_known_hosts`。

代码与持久记录位于本仓库；缓存、优化器与日志位于worker的`/tmp/aflow_race_20261028`。主节点与worker的`/tmp`不是同一位置。原数据准备PID1015、首次协调器PID1184均为历史；当前协调器见[COORDINATOR_DDP200_PID.json](COORDINATOR_DDP200_PID.json)，训练torchrun见TRAIN_PIDS。应先核对当前命令行，不能只凭历史PID判断存活或执行kill。

## 当前流程

1. `scripts.race_seg_20261028.prepare`每例N4一次，同时导出128³与原生数据；逐SHA匹配历史后写DATA_READY。
2. `scripts.race_seg_20261028.coordinator`等待数据门，通过合成吞吐和剩余时间冻结共同轮数及微批。
3. control使用GPU0，race使用GPU1；正式PID写入TRAIN_PIDS。每组日志、HEARTBEAT、STATUS、BEST与逐轮108报告分别归档。
4. 完成后自动运行finalize，报告患者配对区间、固定收益门、权重与报告SHA，并更新根文档、运行四项文档校验。若发生异常，应保留真实未完成状态。

首次实现已存档于Medical_research提交`3306992ca9ece9e2b6faf5b4bc10dacd4b88be12`；后续修订以仓库main和EXECUTION_FREEZE为准；本次提交存档实现与合成证据，不包含未来患者训练结果。8项CPU测试、完整双卡合成更新、尾批与严格重载已通过；最终修正保证当前特征与参照使用同一value投影空间，辅助随机采样不改变骨干dropout随机数。

## 安全接续边界

先读PREPARE_HEARTBEAT、DATA_READY、PIPELINE_STATUS及两组直接HEARTBEAT/STATUS，确认真实阶段。不得把准备阶段当作患者训练，不得重启旧width、distill或nnU-Net协调器。

训练最多各200轮，共同实际轮数由BUDGET_DECISION决定，不能因分数延长或增加候选；每组工程恢复最多2次，只从完整epoch的model/optimizer接续。缓存准备已启动，禁止直接重跑prepare覆盖或重复N4；准备失败应先检查原始失败记录与已分配载荷。

本轮无新test270/outer/confirmation/外部模型访问。原972/108及患者隔离、原生载荷SHA、WT标签1/2/3排除4、阈值0.5固定。持久新增<=4GiB、空闲>=20GiB；大载荷仅worker临时盘。

2026年10月28日启动核验：共同预算已冻结173轮；两组第1轮与检查点重载通过，参见[启动收据](STARTUP_VERIFIED.json)。主节点另有仅负责汇总结果存档的守护，不运行或重启模型训练。

## 用户要求对齐200轮后的执行修订

用户随后明确“我建议直接训200轮吧，对齐一下”。本次将两组目标从患者前选出的173轮改为各200轮、24400保留更新；原173轮决策与启动收据保持历史事实，不能称200轮已在患者前完整冻结。当前接续状态见[BUDGET_MIGRATION.json](BUDGET_MIGRATION.json)，授权与附加源码冻结分别见[RESUME200_PLAN.json](RESUME200_PLAN.json)、[RESUME200_FREEZE.json](RESUME200_FREEZE.json)。原EXECUTION_FREEZE及数值训练源码保持不变。

原监督器先退出，两组分别在下个完整epoch的model/optimizer保存、STATUS和HISTORY写入后暂停。核对epoch×122优化器计数、有限状态、best一致性及SHA，在worker临时盘保留原检查点副本后接续下一轮。若边界后已经发生未保存工作，明确作为丢弃的未完成epoch工作记录，不冒充无中断或相同实际FLOPs。此为用户预算修订的计划迁移1次，工程异常自动恢复计数仍为每组最多2次。

新监督入口为`scripts.race_seg_20261028.coordinator200`，训练入口`resume200`复用原`train.train(..., resume=True)`，保留完整模型和AdamW、有效batch8、微批2、122更新/轮及原本就按200轮定义的学习率。唯一执行优化是将父训练进程CPU线程2→8；合成整批8例增强的图像、标签、类型、有效区及参照矩阵字节核对一致，见[CPU_PROFILE.json](CPU_PROFILE.json)。合成CPU阶段提速不等于实际整轮提速，后续以新HISTORY为准。

原24小时截止仍为AUTHORIZATION控制的10月29日北京时间12:06；200轮是目标，若时间不足仍保留真实未完成状态，不自动延长资源或缩减每轮数据。早期约7.2分钟/轮预计较24小时窗口偏紧；CPU优化后的真实耗时要通过接续完整epoch检查。当前仅早期数轮，不能预称200轮必然收敛、验证收益或完成训练；判断收敛需结合后期训练目标、验证曲线和200轮学习率终点。无新270/outer/confirmation。

[训练速度对照证据](TRAINING_SPEED.json)保留旧128³双卡、原生16通道双卡、近期32通道单卡及本轮两组的实际耗时来源。当前算法相对同期对照多约半分钟，不能把全部历史速度差归因新算法。

## 最新修订：取消对照，双卡同训算法

用户随后明确“不需要对照了吧，双卡直接训就好了”，覆盖上节的两组200轮安排。control已在完整第4轮/488更新停止，完整model/optimizer与best/latest保留；没有任何继续训练许可。旧`migrate200`已终止，`coordinator200`和`resume200`未启动，不能重启。新监督为`coordinator_ddp200`，数值入口`ddp200`，结题入口`finalize_ddp200`。

算法单卡阶段在下个完整epoch后保留全部模型、AdamW状态、best记录和计数，再接双卡DDP；不会从随机初始化重新训练。新增原生整脑双卡纯合成门检查五次真实更新、全局尾批4、两rank参数SHA严格一致、完整optimizer重载及每卡reserved<=38GiB，门失败时不进入患者训练。当前合成/迁移是否完成必须读SMOKE_DDP200与DDP_MIGRATION，不把计划称为启动。

两个rank各取全局8例中不重叠的4例，每卡微批2、累积两次，DDP梯度平均后各进行一次AdamW更新；尾批全局4时各取2。原始CPU增强以全局8例构建后再分半，以复用原增强规则与随机流；缓存读取和CPU增强有执行复制，但每轮模型监督严格合计972个唯一扫描、无重复/丢尾，仍122更新，累计24400更新。使用8 CPU线程/训练进程。训练目标、损失权重、原生图像与标签SHA、原200轮学习率曲线和阈值均不变；DDP归约及dropout轨迹改变，不声称逐位同轨迹或相同实际FLOPs。

108验证在两卡各54例独立前向后统一汇总，不重复完整108验证。第一接续完整epoch与最终200轮须核验CPU重载、optimizer计数和双rank参数SHA。每组工程异常最多2次现在只适用于剩余算法，旧对照不得借异常恢复启动。原24小时截止和存储边界不变。

原同期control/race收益门因取消control不能评估。新结果只与已固定的历史32通道200轮参考（最佳171轮、108 Scan0.753276500）复用既存逐例报告作描述性比较，原历史报告SHA在DDP200_PLAN固定。保留原+0.005、患者区间正、Subject/Point不降作为“历史提升门”并单列，不能改称同期消融或单独算法因果收益。最终归档通过新的CPU/报告审计后由已有主节点守护推送；无新增270/outer/confirmation，不恢复任何旧实验。

正式DDP患者训练前的工程修订均保留在engineering/ddp_preflight_v1至v3。首次合成计算通过但CUDA/DDP对象析构滞留；清理滞留进程和重叠的下一合成实例后，新增完成收据+有界退出，在成功计算/检查点提交后结束进程，不以收尾失败重新做训练。最终revision4合成5次真实更新、尾批、重载、两rank参数SHA、增强预取逐值一致和进程退出码0全部通过，见[SMOKE_DDP200.json](SMOKE_DDP200.json)、[进程退出门](DDP_GATE_PROCESS_EXIT.json)。此前均0患者DDP更新，原第5轮/610更新检查点不变；事件见[DDP_SYNTHETIC_CLEANUP_EVENT.json](DDP_SYNTHETIC_CLEANUP_EVENT.json)。旧bootstrap_ddp200已结束，不可重跑；launch_ddp200从已验证检查点接管，当前coordinator_ddp200持续监督。

revision4使用NCCL安全的spawn数据进程、持久训练/验证loader和顺序CPU增强预取1批；下一批增强与当前GPU计算重叠。增强规则/种子/顺序不变，合成输入逐值核对通过。SMOKE中seconds_per_update只计准备好数据后的优化器更新，不能将其直接当成端到端训练速度；真实整轮速度看患者HISTORY。
