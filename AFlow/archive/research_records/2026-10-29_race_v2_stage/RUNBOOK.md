# 新阶段总控与交接

总控管理三个子agent：`/root/training`、`/root/inference_a`、`/root/inference_b`。分配见RESOURCE_ALLOCATION.json，只有总控可更改；不同worker的/tmp互不相通，不能据目录/index存在认定payload完整。

- 训练：1154259 GPU0+1，独立源码`scripts/race_v2_20261029/`，记录`../2026-10-29_race_v2_training/`。先数据/参照诊断/合成/预算，再新优化器微调。
- 推理A：1154121 GPU0，源码`scripts/race_v2_inference_a_20261029/`，记录`../2026-10-29_race_v2_inference_a/`。负责共享108 native构建及RACE8视图；旧270只核验，最终冻结前不新评分。
- 推理B：1154121 GPU1，源码`scripts/race_v2_inference_b_20261029/`，记录`../2026-10-29_race_v2_inference_b/`。负责w32八视图与12项统一融合；共享A完整发布的只读缓存。

根授权首次24小时取worker创建时间保守起算：推理至北京时间2026-10-30 12:59:34，训练计算至13:17:37，训练资源整体至14:17:37。用户允许高利用率下延长至96小时；当前冻结不预支条件延长，也不复用旧24小时授权。只询问一次资源时长，回复已存AUTHORIZATION.json。

资源采样入口`python3 -m scripts.race_v2_stage_20261029.resource_monitor`：只读查询两worker，每60秒记录一次；摘要RESOURCE_STATUS.json，完整采样在主节点/tmp/aflow_race_v2_stage_20261029/resource_samples.jsonl。采样活跃比例不等同连续占用，不把CPU预处理称为GPU训练。监控不自动制造负载、不延长授权、不启动旧任务。

每个GPU在对应worker持有`/tmp/aflow_race_v2_20261029/locks/worker<id>_gpu<index>.lock`的fcntl独占锁；元数据含owner、PID和物理UUID。总控核验四个UUID互异。新`governance.guard`核对本轮截止、持久空间和实验新增占用；独立队列可保留其更严格的检查。

训练交接包必须有run_id、架构／构造参数、固定checkpoint路径及SHA、CPU strict reload收据、真实新增轮数／更新数、源码／配方冻结、108逐例结果、输入几何和参照矩阵约定。完整best/latest含优化器保留worker/tmp，并备份主节点/tmp；推理不得读取正在覆盖的latest。

最终选择由总控登记最多两个候选（最强单模型、纯SegResNet融合），各自固定模型SHA、配方SHA、108依据与前向成本；270仅在此收据之后评分。原RACE164270八视图经载荷SHA、形状、有限值及参考矩阵来源审计后可复用，新权重需要自己的预测。缺失须先登记、有界重建，不能称独立确认。

发布采用既有独立checkout `/tmp/medical_research_publish_20261014`，目标`https://github.com/songjunyi35-ctrl/Medical_research`，只推代码与文本证据；不改变本工作区AFlow上游remote，不上传影像、标签、权重、预测数组或凭据。每次实际结果同步方法目录、交接、流水账；完成270后同步两份排行榜并验证报告SHA。
