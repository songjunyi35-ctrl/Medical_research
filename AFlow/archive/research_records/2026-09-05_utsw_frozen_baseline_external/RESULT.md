# UTSW362冻结baseline外部基线：来源标签域门失败

状态：在UTSW模型推理前关闭当前v1实现。

两checkpoint的synthetic smoke全过；第一次724路径单连接因Aspera source数上限失败，透明Amendment改为每copy固定8批后，362图像+人工标签双下载724/724官方size/SHA exact。

首次cache构建在BT0007被预注册raw label域`{0,1,2,4}`门拒绝。独立全362诊断得到union=`{0,1,2,3,4,5}`：label3出现116例/2,762,246体素，label4出现270例，二者共现86例；label5出现5例/434体素。FeTS一手标准只定义1/2/4三个肿瘤区，因此不能在当前实现里事后把3/5自动并入WT。

没有CACHE_RECEIPT，partial稀疏cache仅占约4KB apparent blocks并保留；UTSW模型forward=0、external metric=0、旧cohort未访问。

后续按源诊断计划固定35个异常病例（30例label3无4，加5例label5），双下载其非`r`原始manual标签35/35 exact。非`r`union仍为`{0,1,2,3,4,5}`，所以3/5不是`r`全图交付独有的重采样现象；官方论文只定义三种肿瘤区且无3/5数值说明，语义继续阻塞。当前UTSW模型路线关闭，未来须发布者映射，不能删5例、排除3/5或用模型分数决定。

证据：[IDEA](IDEA.md)、[PREREG](PREREG.md)、[首次传输失败](EXECUTION_01_FAILURE.json)、[批传输修订](AMENDMENT_01_TRANSFER_BATCHING.md)、[双下载](SOURCE_DOWNLOAD_AMENDMENT_01.json)、[cache失败](CACHE_EXECUTION_01_FAILURE.json)、[标签域诊断](LABEL_DOMAIN_DIAGNOSTIC_AMENDMENT_01.json)、[非r探针](NONR_ANOMALY_PROBE.json)、[机器结果](RESULT.json)。
