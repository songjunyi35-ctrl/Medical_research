## 保留原始子区标签的fit数据视图 ✅ 工程完成，未训练新模型

**一句话：** 已保留现有128栅格中的原始类别信息，供未来严格比较区域编码。

**出发点：** 旧开发漏画约92%来自标签2，而错误最高点约38%—65%在切除腔；仅WT/RC二值视图不能直接研究前景内部类别。

**具体做法：** fit684/313原生0—4标签按原image-only crop/nearest-exact128保存uint8；WT=[1,2,3]、TC=[1,3]、ET=3、RC=4。当前模糊的原标签/模型/损失/采样完全不改。Torch构建后用NumPy中心索引独立重建全部病例。

**结果：** 684/684源SHA、旧WT/RC字节和独立完整类数组匹配。包含各类0—4的病例数=[684, 264, 684, 481, 561]；各区域病例数={'whole_tumor': 684, 'tumor_core': 482, 'enhancing_tissue': 481, 'resection_cavity': 561}。这是数据工程结果，没有新模型Dice/Point或CI。

**是否支持直觉：** 支持能无损保留当前栅格类别信息，不证明区域编码会提高指标、不恢复原生缩放细节。旧RC辅助0.1负结果保留，后续编码模型需另注册匹配对照；不能照抄nnU-Net重新编号示例的核心标签。

**代价/下一步：** /tmp缓存1.336GiB，4 CPU；构建主体29.6秒、独立重建主体30.3秒（均不含前置哈希核查），0 GPU/模型前向。无dev/test标签新读取，当前模糊训练不接入新cache。[七问](archive/research_records/2026-09-10_fit_subregion_labels/IDEA.md)、[结果](archive/research_records/2026-09-10_fit_subregion_labels/RESULT.json)、[缓存收据](archive/research_records/2026-09-10_fit_subregion_labels/CACHE_RECEIPT.json)、[独立验证](archive/research_records/2026-09-10_fit_subregion_labels/VERIFICATION.json)。

**一手依据边界：** 原论文的区域对照并非当前WT-only基线追加辅助区域；本工程只准备数据，不证明新增区域目标会提高指标。下一步先做匹配对照设计。[原文适用边界](archive/research_records/2026-09-10_resumed_eight_hour/PRIMARY_SOURCE_CHECK.md)。
