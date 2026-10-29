# 固定纯SegResNet融合270恢复结果

原270扫描/119患者、单T1ce、原生1mm完整原crop评分。固定RACE164+w32e171各8镜像，appearance_k1_lambda0.5，不改原FINAL5b74。

Scan **0.789070115**、Subject **0.776314624**、Point **254/270**，未超过0.8000，也未高于历史RACE full0.789263。全部270保存mask TP/FP/FN/Dice/最终score Point独立复数审计exact。

患者配对10000簇bootstrap seed20261029、Subject配对区间、对原raw/历史full/本轮single的全部变差病例见COMPARISONS.json；均是继承test-tuned历史的描述性证据，不称独立确认。部署2模型16网络前向+每类5EM。

原1154121已失联导致/tmp缓存丢失。原80训练完成保留后，root将1154259闲置GPU1分配给本恢复；A重新N4构建四payload且实际SHA全等历史，新index不冒充旧byteSHA。A历史RACE八镜像重建2160全旧atomSHAexact；B w32实际新2160，batch8/尾6，272calls/238.447秒；固定GMM解码87.787秒。总实际恢复患者前向4320，不冒充零成本复用，不重复研究或改配方。

新RAW_FUSION与MASK_AUDIT在recovery/b，临床数组仅新worker/tmp、不跨目的地备份。原训练/历史权重/旧源码/冻结均未动。唯一队列21240退出、canonical GPU1物理锁free、该卡0MiB/0%已只读核验；不自动续训、扩搜或重测。
