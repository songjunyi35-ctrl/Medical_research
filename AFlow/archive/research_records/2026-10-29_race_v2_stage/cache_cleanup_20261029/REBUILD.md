# 两组旧缓存的删除与重建边界

2026-10-29用户明确授权清理旧缓存并保留构建脚本。本次只删除PLAN.json中两组1134个NPZ，不执行重建、模型前向或评分。原始输入、全部模型、索引、冻结记录、逐例分数和最终报告保留；当前两worker的训练与108/270缓存保留。

## 已关闭684／180支线的864份预处理数组

构建函数保留在`archive/reviews/2026-09-16_remove_684_branch/closed_material/scripts/sammed3d_graymean_pair.py.disabled`的`cache_one`，内部调用`preprocess_row`，产物仅`image.npy`与`label.npy`。索引保留在`archive/reviews/2026-09-16_remove_684_branch/closed_material/pair_technical/CACHE_INDEX.json`，每例原SHA与bytes完整；本次逐一核对后删除。

将来如明确需要重建，应先恢复原清单和输入路径、核原始image／label SHA，再在新worker/tmp运行隔离的纯预处理入口并核864个输出SHA。旧完整`build_cache`还依赖当时的冻结和合成门，不能直接重启已关闭的684训练流水线；本次不恢复已删除的SAM权重，也不声称旧训练环境仍可直接执行。

## nnU-Net525的270份连续margin预测缓存

构建源码保留在`scripts/nnunet_resenc_20261013/finalize_v3.py`中270预测与`np.savez_compressed(...,margin=margin)`段；原`TEST_INFERENCE_JOURNAL.json`保留270路径、输出SHA与窗口数。模型检查点、计划、原始来源与全部报告保留，本次NPZ逐SHA匹配且ZIP元数据确认为单一`margin.npy`后删除。

未来恢复需按保留的525权重、原生1mm几何与nnU-Net预处理／滑窗／还原参数，在新worker/tmp重新生成该固定模型的270输出；需另登记真实模型访问和前向成本，不能凭索引声称已有缓存，也不能把重建当成新独立确认。不直接运行过期的整套finalize／训练协调器。本次不重新推理nnU-Net，也不将其引入当前SegResNet路线。

## 留存与审计

- [删除白名单与来源SHA](audit/b/PAYLOAD_SHA_AUDIT.json)
- [模型前后完整SHA核对](audit/a/SUMMARY_AFTER.json)
- [删除结果](RESULT.json)

两处原目录均写入`PAYLOAD_REMOVED_20261029.json`。目录或索引仍存在不代表数组仍完整。8MiB旧SAM smoke缓存缺少当时的权重和上游载荷依赖，未纳入本次确认可重建删除白名单。Git对象、原始MRI、模型和最终掩膜／分数证据未清理。
