# 用户分工澄清后的执行链

1154259 GPU0+1的原RACE-v2训练保持原进程、源码冻结、数据、优化器、学习率与80新增轮／9760更新不变。1154121 GPU0／GPU1分别归A／B推理队列，各自持有原物理卡锁；没有跨worker抢卡、跨节点DDP或第二训练任务。

1. A16配置与B12配置均已完成108验证，新增网络前向0；总控复核32组整数报告与五折选择。两项新机制对各自参照的患者配对区间均跨0，不能称稳定增益。
2. 等原训练完整80轮及原子固定best交接。新优化器、真实新增轮／更新、CPU严格重载、checkpoint与代码／配方SHA、逐例108、原生几何及镜像共轭约定均由既有交接包记录。
3. 最佳新增轮为0时，root写`FINAL_HANDOFF_NO_NEW_MODEL.json`，A／B转移队列写显式跳过标记，沿用旧模型缓存；不会伪造新模型或重复八视图前向。
4. 最佳新增轮大于0时，仅生成最终best的108八视图缓存。旧有界伙伴选择／融合复核后，A／B各固定转移一条108已选的新算法，不重开16／12搜索。所有大数组留在1154121/tmp。
5. [新冻结选择政策](FINAL_SELECTION_INNOVATION_POLICY_FREEZE.json)比较至多四种单模与四种双模选项，以完整108 Scan为主，精确同分保留先前选项。只生成一个单模、一个纯SegResNet双模的不可变`FINAL_TEST_SELECTION.json`。政策在既有与新机制108结果后修订，继承test-tuned历史；尚未访问本轮新270分数。
6. 只执行最终两个270候选。RACE164现有2160视图已逐SHA、形状、finite审计；直接复用。选中新模型及需要的w32必须各自产生预测，同模型不同解码共享缓存，部署8／16前向每例。患者前向不自动重试。
7. 推理worker上的两个队列完整重载540份最终掩膜复核TP／FP／FN、Dice、Point与SHA。训练worker不访问270。主节点只收文本统计，将mask-audit receipt绑定实际canonical raw SHA和wrapper全部270行，独立计算Scan／Subject、患者配对区间和所有变差病例。
8. 单模与双模分别写榜单、方法目录、交接及流水账，通过四个文档／榜单检查后推送代码和文本。新发布入口还包含archive中的冻结旧Python源码，保留来源SHA映射；不上传图像、标签、预测数组或权重。

新selector PID2079633，结题V2 PID2082077，统一只读状态PID2082805。A原snapshot1642／新固定转移12036／新最终12037；B原snapshot9260／新固定转移11776／新最终11777。旧空闲最终等待器被新入口替换，原训练PID未动。PID是本次启动收据，实时状态与[完整登记](EXECUTION_REGISTER.json)、[统一进度](INNOVATION_STAGE_STATUS.json)为准。

[新增108结果](INFERENCE_INNOVATION108_RESULT.md) · [结题V2冻结与4项错配回归](FINAL_PUBLICATION_INNOVATION_FREEZE_V2.json) · [原执行链，完整保留](EXECUTION_CHAIN.md)
