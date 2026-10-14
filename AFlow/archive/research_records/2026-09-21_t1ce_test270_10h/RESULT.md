# 10小时单T1ce定向优化结果

状态：10小时监督结束；未达到0.80且至少+0.03目标。

270 scans /119 subjects，原始1mm crop，排除填充。总体选panel_10：Scan **0.775891493**，Subject **0.761694199**，Point 261/270；Scan患者簇95%区间[0.7518267700847958, 0.7967372880713604]。最佳单模型panel_02 Scan 0.772829645，Subject 0.758800132；单模型不等于单次前向。

固定五模型最强参考ref_native173 Scan 0.768573106；总体差+0.007318387，患者配对95%区间[0.0031155479307933965, 0.012786231698480971]。相对固定193同栅格差+0.014756423，区间[0.008780445478531582, 0.02123841888829019]。所有区间仅描述性，不消除反复选择偏差。

本轮明确test-tuned：270标签用于错误诊断、门审、模型/阈值选择；仅原972参与分割训练；108用于分支检查点选择。不是独立泛化、盲测或外部确认，无新outer/confirmation。保留传统榜和全部失败门审，不能用本榜冒充传统最佳。

方法直觉、单因素和控制见[七问](IDEA.md)。训练状态：{"A": {"utc": "2026-09-21T15:17:30.837573+00:00", "state": "completed", "arm": "A", "last_complete_epoch": 200, "partial_next_epoch": false, "completed_updates": 24400, "best_epoch": 90, "best_scan": 0.7521689044265755, "reason": "epoch budget", "elapsed": 6033.229177504778, "test_tuned": true, "new_outer_confirmation_access": false}, "B": {"utc": "2026-09-21T13:52:06.092212+00:00", "state": "stopped", "arm": "B", "last_complete_epoch": 25, "partial_next_epoch": false, "completed_updates": 3050, "best_epoch": 0, "best_scan": 0.7496144051100061, "reason": "epoch25 gain and progress below budget gate", "elapsed": 1134.8059021383524, "test_tuned": true, "new_outer_confirmation_access": false}, "C": {"utc": "2026-09-21T13:52:30.280731+00:00", "state": "not_started", "reason": "initial FNR-FDR below0.05", "test_access": false, "completed_updates": 0}}。输出预算5.332个128³前向等价量，阈值0.4；实际耗时和重载检验见DEPLOYMENT_CHECK.json。持久磁盘新增0.276GiB；临时缓存不作为持久交付。负结果仅评价本实现与预算，不证明一般方向无效。

[机器结果](RESULT.json) · [逐例总体报告](reports/panel_10.json) · [单模型报告](reports/panel_02.json) · [选择收据](FINAL_SELECTION.json) · [部署清单](../../../workspace/BraTS/segresnet_t1ce_wt/test270_10h_20260921/deployment/manifest.json) · [访问账本](ACCESS_SUMMARY.json) · [协议](PROTOCOL.json)


<!-- COMPLETION_SUPPLEMENT -->
## 收益来源与诊断

总体0.775891来自**已有native173＋组合166的等权logits融合及统一概率阈值0.4**，不是新训练A带来的总体收益。A额外训练200轮后仍选新增第90轮，最佳单模型0.772830，需要8块前向；总体方案2次native前向约5.332个128³等价量。[输出方法完整解释](output_panel/RESULT.md)。

无增强、同训练/验证原生栅格诊断：最终流程训练972/444 Scan 0.817523、验证108/50 Scan 0.754431；纯193为0.809540/0.742672，native173为0.810327/0.746099。训练病例本身也存在明显剩余误差；这些数据不能单独证明具体过拟合原因、标签病因或T1ce性能上限。诊断在最终配置固定后执行，没有回流选择，也没有新270访问。[诊断](TRAIN_FIT_DIAGNOSTIC.md)。

暖启动合成GPU基准：总体batch1约0.236秒/例、相对128³约5.36倍；最佳单模型约0.351秒/例、约7.99倍。仅测模型/融合/阈值，排除N4、磁盘、启动和权重加载；不是临床端到端耗时。[测量](DEPLOYMENT_BENCHMARK.json)。训练进程墙钟折算约3.667 GPU小时，不把10小时监督窗口写成10小时满负载训练。

混合精度下batch1部署与batch2评分不是逐位相同：预设4例重载检查最大掩膜不一致比例0.00001896；榜上成绩对应冻结batch2。与历史参考的微小数值差保留于[数值记录](NUMERIC_COMPARABILITY.json)，旧分数不改写。

C虽然没有模型前向，其是否启动使用了270错误诊断；不能把C的“无候选测试”误读为全流程未看270。早期缓存字节哈希访问与随后native构建访问的边界见[范围补记](ACCESS_BOUNDARY_CLARIFICATION.json)，未倒填日期。部署打包字段错误已经修复，原12项报告和选择完全保留，没有重新评分；见[工程修订003](ENGINEERING_AMENDMENT_003.json)。

[总体部署清单](../../../workspace/BraTS/segresnet_t1ce_wt/test270_10h_20260921/deployment/manifest.json) · [最佳单模型清单](../../../workspace/BraTS/segresnet_t1ce_wt/test270_10h_20260921/deployment/best_single_manifest.json) · [训练曲线](LIVE_CURVES.png)
<!-- END_COMPLETION_SUPPLEMENT -->


<!-- WORKER_AVAILABILITY -->
## 单卡节点可见性事件

单卡1105022最后心跳为19:06:25 UTC，19:08起SSH连接失败；复查平台活动清单已无该worker，仅剩1104883。不能把连接失败本身冒充PID退出证明，也不能声称单卡全程在线。B训练在13:52已停止，C未启动；全部持久模型和报告核验完整。B的270份临时预测与剩余双卡上的ref_pretrained179逐文件SHA完全一致，已现场核验，无须重跑模型。

单卡事件后主节点与双卡曾继续监督；随后双卡也失去可见性，见下节。主节点仍持续记录端点状态至原截止；没有启动替代GPU、恢复训练或新增评估。10小时指整体科研监督窗口，单卡可见性确有中断；其磁盘数字仅代表最后可观测时刻。

[节点事件](WORKER_VISIBILITY_INCIDENT_001.json) · [平台原始清单](PLATFORM_INVENTORY_AFTER_SINGLE_LOSS.txt) · [缓存冗余核验](SINGLE_CACHE_REDUNDANCY_VERIFIED.json) · [监控修订](MONITORING_AMENDMENT_004.json)
<!-- END_WORKER_AVAILABILITY -->


<!-- DUAL_WORKER_AVAILABILITY -->
## 双卡节点可见性事件与剩余监督

双卡1104883最后心跳为21:06:12 UTC，21:06:43起SSH连接失败；重复查询的平台活动worker清单为空。其远程PID当前不可直接观察，不能仅凭SSH错误宣称具体进程退出状态。A训练15:17已完成，最终面板、部署检查和拟合诊断也已完成；3个部署权重、21份270原始报告及3份固定数据清单在节点不可访问后再次核验完整、SHA一致。

两节点临时盘此后不可访问；不声称临时预测或影像缓存永久可用。已有缓存SHA、源码、模型和报告保存在持久目录，没有因为节点离线重新构建或推理。余下监督由主节点持续监测持久化终态、端点状态和磁盘直到23:04:52 UTC；没有启动替代GPU或恢复研究。10小时是整体科研监督窗口，不是每个worker全程在线或10小时满载训练。

[双卡事件](WORKER_VISIBILITY_INCIDENT_002.json) · [空活动清单](PLATFORM_INVENTORY_AFTER_DUAL_LOSS.txt) · [持久产物再核验](PERSISTENT_ARTIFACTS_AFTER_DUAL_LOSS.json)
<!-- END_DUAL_WORKER_AVAILABILITY -->
