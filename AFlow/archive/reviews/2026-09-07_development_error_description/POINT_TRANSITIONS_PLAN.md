# 患者采样的Point转移描述（结果后审计）

本分析在正式结果揭示及联合门失败之后提出，明确post-hoc，仅用于解释保存的逐例结果，不是新的效果检验或预注册子组成功门。

只读取已冻结RAW_DEVELOPMENT的两seed candidate/reused_control行。每seed全180例按control→candidate Point布尔值分为保持命中、失去命中、获得命中、保持未命中；所有类别都报告。核对lost/gained差等于原Point净差，并列出所有失去或获得命中的病例、原/新Dice及计数，不筛选其中“好看”的例子。

另外描述lost病例中Dice是否改善，以及两seed lost/gained病例集合是否重叠。只作计数，不做新CI、p值、非劣效门或病例路由，不改变原Point≥−1成功门，不取平均Point或选择seed来覆盖失败。

这能区分净变化与实际发生变化的病例数，也能直接观察整块mask Dice与最高分点是否发生方向分离。但原报告没有完整logit，不能判断置信度排序变化的具体机制或量化并列最大值；本次不为解释而重跑模型。0新forward、0影像/标签/checkpoint数组读取、0保护队列访问。
