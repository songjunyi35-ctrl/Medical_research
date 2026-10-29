# 最终固定权重108推理交接队列

入口`scripts/race_v2_inference_a_20261029/snapshot108.py`，另有不可变`SNAPSHOT108_FREEZE.json`。worker1154121 GPU0安静等待总控在`archive/research_records/2026-10-29_race_v2_stage/SNAPSHOT108_QUEUE/`登记最终包。等待期间无GPU负载，不靠虚假前向保资源。40轮中间模型不进入队列。

仅接完整训练预算`DONE.state=complete`且`new_epoch=target_new_epochs`的最终固定best；最多一个新权重任务。必须验证root登记的HANDOFF和DONE文件SHA、run_id、架构构造参数、checkpoint SHA、实际best新增轮/更新及总阶段轮/更新、代码与配方冻结、108逐例身份和输入几何。拒绝可覆盖的best/latest文件、epoch0重复源模型、非终态训练和未登记模型。固定权重在CPU严格加载并逐张量核对后才过GPU完整batch8/尾4门，随后一次864病例前向。

支持`RaceSeg`及本轮参数相同的`RaceBenefitSeg`，仅宽32/dropout0.1/dim16/enabled。108使用历史160×208×160画布，R_view=T@R@T、有效支撑同步变换，几何送入预测前只保留四个允许字段。模型与解码不接GT或病例ID路由。

新权重只报告三个固定参考和本轮已选单模型配方的直接移植。当前选中等权8镜像logit阈值0.5（mode=equal、config=null），本身已是参考，因此实际只有三个唯一解码。不会重新32配置搜索，不会自动270前向或评分。每个结果相对RACE164同配方报告患者配对CI、5患者折指标与全部变差病例。所有临床数组继续留在worker1154121/tmp，不向主节点或其他目的地备份。

总控登记JSON字段为`job_id`、`registered_by="root"`、`approval_scope="final_completed_training_snapshot108_only"`、`run_id`、`handoff_path`、`handoff_sha256`、`training_done_path`、`training_done_sha256`、`registered_at`。路径与SHA必须是已导出CPU严格重载包，不能预先登记尚在写入的文件。

若队列失败，保留JOB_FREEZE和完整journal，不自动重复患者网络前向。数据缺失或节点回收必须交总控登记有界重建，不能冒称存在备份。当前worker截止沿本轮AUTHORIZATION，不沿用旧窗口，也不自动预支96小时。
