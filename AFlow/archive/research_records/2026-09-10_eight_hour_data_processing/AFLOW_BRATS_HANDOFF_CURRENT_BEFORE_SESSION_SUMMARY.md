# AFlow BraTS 当前状态

更新时间：2026-09-10 UTC

## 2026-09-10 八小时自主科研（当前）

10:36 UTC：占比标签两候选开发完成并独立核验，Scan0.746659/0.739852（对native−.000104/−.004325），Subject也未过门，Point+5/+4。原配方与其单模型test分支关闭，0新增test；所有训练都已退出。剩余时间显式增加独立推理研究“0.5患者采样模型+0.5占比模型raw logits”，不修改/重开两成员失败配方，不称1×。这是基于已揭示dev的新提案，已另七问/预注册；只一个权重、双seed，必须超两成员Dice并保最佳Point−1才晋级；失败不扫权重。高层“此前不再追加方案”安排在10:31明确修订，旧实验门保持不变。当前仅新fusion fit8 smoke会话70912，尚未读融合dev。见[占比结果解释](archive/research_records/2026-09-10_fractional_occupancy/INTERPRETATION.md)、[融合七问](archive/research_records/2026-09-10_data_recipe_ensemble/IDEA.md)、[融合预注册](archive/research_records/2026-09-10_data_recipe_ensemble/PREREG.md)。

10:24 UTC：占比target两个168轮候选完成，监督器39049正常exit0，峰值各20.664GiB/导出error0。全336轮独立抽样/RNG/记录图像byte配对通过，TRAINING_REPLAY_FINAL已保存。统一原hard GT的dev180/78读出会话6188已启动（360新forwards）；尚未有结果，不准入test。当前无后台训练，只有该开发评测；截止仍11:28:19 UTC。

09:28 UTC六小时节点：占比target两个种子95/168，唯一活跃科研作业39049；预计10:25左右训练结束，11:28:19 UTC总截止不变。该新方法dev/test均未访问，训练/开发评测冻结未改。弱形变和小孤岛清理的负结果均已归档，full864弱形变未准入；不重启。后续严格先`audit_fractional_occupancy_training --final`，再统一`evaluate_fractional_occupancy`和`finalize_fractional_occupancy`，全门才允许既定seed11 test；不按结果追加新方案/改seed/改阈值。

08:28 UTC五小时节点：占比标签两候选约17/168轮，唯一活跃作业仍会话39049（PID19199/19200），预计10:25左右完成；截止11:28:19 UTC。前10轮独立核查抽样/原RNG/记录图像byte exact，只有target改变。占比开发评估/归档代码已提前冻结。完成后必须先`python -m scripts.audit_fractional_occupancy_training --final`，再`evaluate_fractional_occupancy`、`finalize_fractional_occupancy`，三者均用python -m scripts.前缀。若两seed全门通过，再且仅再走`freeze_fractional_occupancy_test270 → evaluate_fractional_occupancy_test270 → verify_saved_test270_panel --study fractional_occupancy → archive_verified_test270_panel --study fractional_occupancy`；这些test入口已实现但未准入/执行。不得按最好seed改选，固定seed11，原硬GT/阈值/4输出不变。所有保护队列仍无新访问。

08:18 UTC当前：弱形变两配方已独立负结果归档，其full864/test分支未执行并关闭。占比target的GPU smoke/完整时间门/哈希冻结全过，08:15双seed168轮训练启动，唯一活跃作业为会话39049（PID19199/19200），各约2/168轮，预计10:25左右完成。fit684/313、dev180/78保持，原dev未被训练消费。新target模型尚无dev/test结果；截至目前本8小时没有新增test/outer/confirmation/external。截止仍11:28:19 UTC。见[占比训练前冻结](archive/research_records/2026-09-10_fractional_occupancy/TRAIN_PREACCESS.json)、[GPU检查](archive/research_records/2026-09-10_fractional_occupancy/SMOKE_RECEIPT.json)。

08:10 UTC主结果：四弱形变候选dev180/78全部读出并独立核验，两配方均不准入。native层Scan0.745530/0.744132（自身control差−.001232/−.000046、CI跨0，Point+2/+4）；subject层Scan0.742229/0.751126（自身control差−.011245/−.004780，Scan CI均负，Point+5/−3）。selected_arm=None，关闭固定elastic与其full864/test分支，不扫强度/epoch/seed。旧dev没有被full864训练消费。当前开始按结果前方案给占比target做GPU smoke/预算准入，尚未训练它。见[主RESULT](archive/research_records/2026-09-10_weak_elastic/RESULT.md)、[解释](archive/research_records/2026-09-10_weak_elastic/INTERPRETATION.md)、[验证](archive/research_records/2026-09-10_weak_elastic/VERIFICATION_RECEIPT.json)。

08:07 UTC：弱形变四条168轮训练全部完成，监督器12554正常exit0；全672轮抽样/原RNG和四条形变流独立重放通过，导出全error0，峰值均20.664GiB。`TRAINING_REPLAY_FINAL.json`已保存。统一四模型dev180/78读出会话16086已启动，最多720新forwards；尚未有结果，不提前准入下一训练或test。其余旧阶段状态以下保留历史。

07:28 UTC四小时节点：seed11两臂168轮完成；seed12 native/subject均118/168，唯一运行作业仍为监督器12554。主模型新dev仍0，预计08:10前后读出；清理支线已负结果关闭，占比target的cache/CPU入口准备完成但未训练。最新全repo881 passed/91warnings/48.50s；后续test保存mask与独立核验/两榜更新均已实现并单测，未执行真实test。截止仍11:28:19 UTC。

06:28 UTC三小时状态：当前唯一运行科研作业为弱形变监督器会话12554，seed12 native/subject均40/168（seed11两臂已完成）；没有正在运行的cache构建或其他训练。全repo872 passed/91warnings/47.46s。继续到11:28:19 UTC总截止，不提前用训练Dice宣称超榜。

执行接续：四臂TRAINING_DONE存在后，先`python -m scripts.audit_weak_elastic_training --final`，再单次`python -m scripts.evaluate_weak_elastic`、`python -m scripts.finalize_weak_elastic`；GPU命令用既有venv/授权GPU访问。结果出来立即补解释、当前交接/百科和validators。若selected_arm非空，只走已准备的`smoke_weak_elastic_full864 → freeze_weak_elastic_full864 → run_weak_elastic_full864`，其后才允许单次test前冻结。若弱形变科学门失败且余时够，走`smoke_fractional_occupancy → freeze_fractional_occupancy → run_fractional_occupancy_training`，完成后`evaluate_fractional_occupancy → finalize_fractional_occupancy`。不要重跑已启动或完成的入口；每个入口有不可覆盖收据。

尚未完成：任何新训练模型dev结果、条件性后续训练/bench、最终结果归档。full864 test冻结/评估入口已准备但未执行；通用保存mask核验和归档/两榜更新入口已于07:09准备并通过单测，不能把准备代码当完成。获准test后顺序为`verify_saved_test270_panel --study <weak_elastic_full864|fractional_occupancy>`，再`archive_verified_test270_panel --study ...`（均用python -m scripts.前缀）；其结果新建在对应`2026-09-10_<study>_test270/RESULT.md/JSON`，保留父目录开发RESULT。占比target的条件性seed11 test冻结/评估入口尚未实现。整个session最多一个最终test panel，未访问旧confirmation/fixed validation/outer/external。

07:09状态：seed12两层93–94/168，仍预计08:10左右统一dev。若test获准，额外存四个预测mask共270MiB（0新增forward）便于直接重算Dice，原logits不存故不声称独立重算argmax；相关掩膜/核验/榜单渲染9项测试通过，实际榜单仍未改。历史test已知、流水线继承test-informed背景，本次不以新panel调参，不能称blind/external。

05:44 UTC备用占比训练标签cache已完成并核验：fit684原硬GT byte exact，新soft每例字节/统计exact，float64最大相对质量误差3.36e−16，float16最大4.02e−5；2.672GiB在/tmp，2356.10秒CPU、0新模型/dev。与旧hard硬化重叠0.973320只是标签表示诊断、不是模型分数。cache会话12958已exit0，验证会话97943已exit0，不重跑builder。只在弱形变无合法晋级资格且时间足够才正式训两seed；否则备用不混入。双seed过门后若仍无本session test，事前固定seed11可另冻结同test270，标fit684、不能选最好seed或冒充full864。见[七问](archive/research_records/2026-09-10_fractional_occupancy/IDEA.md)、[cache结果](archive/research_records/2026-09-10_fractional_occupancy/CACHE_RESULT.md)、[验证](archive/research_records/2026-09-10_fractional_occupancy/CACHE_VERIFICATION.json)。

主GPU最新06:04：弱形变seed11两层168轮完成，耗时7767.62/7763.09s，峰值20.664GiB、导出error0；完整168轮抽样/原增强RNG/20328步形变流重放exact，32531/162624样本形变=20.0038%。seed12两层已自动启动（PID11562/11563），约9/168轮；监督器12554仍运行，预计08:10左右统一读出。四臂未完成前不读其新dev。见[重放](archive/research_records/2026-09-10_weak_elastic/TRAINING_REPLAY_055913.json)。记录了[软件/GPU元数据](archive/research_records/2026-09-10_weak_elastic/ENVIRONMENT_20260910.json)：历史80GB control复用，当前40GB；未声称从头bit-exact重训历史control。

seed11新checkpoint：native SHA12bd86129ac5d387c4ffd608012f7aabdccbb30e2b5780b90c24e8f1455683d7，subject SHA273537ab9cc0de7e2bcd97ce1885ec9319436275f4f30ebc897c17ac41775e65；均在weak_elastic_20260910/seed2026090711_*下fixed_epoch168.pth，未做dev前向。备用占比数据入口的fit8 CPU核查也通过：图像/RC exact，仅WT target改变，0model，[收据](archive/research_records/2026-09-10_fractional_occupancy/DATASET_CPU_RECEIPT.json)。

04:17 UTC独立小孤岛清理已完成/关闭：固定26邻接、8体素/最大块1%、最大块/argmax保护，两个旧native模型reused dev180/78新增360forwards。清理后Scan0.746668827/0.744056511，差−.000093489/−.000120907，CI跨0；Subject也微负，Point170/163和全部点坐标不变。只去掉原FP0.3447%/0.3762%却删TP3805/5071，提升直觉未支持，不扫比例，条件性full864清理路线关闭。0新训练、31.63s、额外峰值4.815GiB，无test/outer/confirmation/external。正在训练的弱形变新checkpoint仍未读dev、配方不变。见[清理RESULT](archive/research_records/2026-09-10_peak_protected_cleanup/RESULT.md)、[解释](archive/research_records/2026-09-10_peak_protected_cleanup/INTERPRETATION.md)、[验证](archive/research_records/2026-09-10_peak_protected_cleanup/VERIFICATION_RECEIPT.json)。

新Goal起点03:28:19 UTC，计划11:28:19 UTC结束；用户要求自主提升并争取超榜、不询问。优先弱弹性形变：native和患者均衡两个固定抽样层×两seed，四新增168轮与各自冻结历史control比较。七问/预注册/实现与数据哈希已冻结，5项合成测试和fit前8例GPU smoke通过（初值/RNG匹配、导出error0、峰值20.664GiB）；03:46 UTC正式双卡训练启动，全部四臂结束后再统一dev读出。弱形变尚无新开发结果；另行清理支线已读dev并关闭如上。旧患者均衡Point门失败不翻案。两A100当前40GB，workspace6.8G空余，不建大缓存。

fit684/313与dev180/78保持；只过既定开发联合门后，另冻结全授权864从头refit与一次同test270的identity/八镜像比较，不用test选参数。旧confirmation/fixed validation/外部不访问。当前旧榜overall Scan0.757269662（Full972KD168 Flip8,8×），单次前向Scan0.752003176（GCSAM184）；均reused test270/119，开发分数不能宣称超榜。

当前依据/执行记录：[IDEA](archive/research_records/2026-09-10_weak_elastic/IDEA.md)、[PREREG](archive/research_records/2026-09-10_weak_elastic/PREREG.md)、[SESSION](archive/research_records/2026-09-10_weak_elastic/SESSION.md)。以下历史轮次和已完成Goal不重启。

04:02 UTC阶段核查：正式训练会话12554仍运行，首波前16轮实际采样/原增强RNG/形变随机流与冻结预期exact（形变19.81%）。CPU全fit684标签压力检查完成：整WT无消失，5个微小组件/6体素消失，体积差中位1.65%/最大12.72%，不改参数。0新dev/test。全repo846 passed/91warnings；一次失败因/tmp旧pydicom wheel丢失，已按旧SHA恢复，不改环境版本或断言。见[工程](archive/research_records/2026-09-10_weak_elastic/ENGINEERING.md)、[标签压力结果](archive/research_records/2026-09-10_weak_elastic/FIT_AUGMENTATION_AUDIT.md)、[筛查排序](archive/reviews/2026-09-10_data_processing_priority/RESEARCH.md)。

这是新会话和新agent唯一需要先读的当前科研状态入口。历史长交接、阶段continuation、旧plan和idea检索均已归档；不要从旧文档恢复已判负分支。

## 2026-09-08最高优先级fit空间保真诊断已完成

用户要求直接执行最高优先级后，已完成fit684/313人的标签裁剪→128³→原生还原诊断。固定两种还原、表面误差和26邻接连通块ID追踪，0模型/0开发与测试评测、不建大cache；损失如实报告、不排例、不称模型上限。有效协议见[V2 IDEA](archive/research_records/2026-09-08_fit_spatial_roundtrip_v2/IDEA.md)、[V2 PREREG](archive/research_records/2026-09-08_fit_spatial_roundtrip_v2/PREREG.md)。弹性形变训练尚未启动；旧30小时Goal不恢复。

执行终态：V2会话88466正常exit0，全684/313完成，旧缓存684/684 byte exact。最近邻/三线性指示图还原Scan Dice0.965432/0.964736（非模型成绩）；184/2446个原生26邻接块无缩小取样，涉及124例，消失块总质量469体素=全部WT的0.001185%，最大63体素。1例固定crop外4体素，未排例；126体素以上1136块均有取样。较大掩膜主体保留较好，但边界/微小块/拓扑不无损，不能推断模型上限。20项新合成/归档测试通过；149.7秒/4 CPU、2.83MB raw，无新模型/开发/test。见[RESULT](archive/research_records/2026-09-08_fit_spatial_roundtrip_v2/RESULT.md)、[解释与下一决定](archive/research_records/2026-09-08_fit_spatial_roundtrip_v2/INTERPRETATION.md)、[VERIFICATION](archive/research_records/2026-09-08_fit_spatial_roundtrip_v2/VERIFICATION.json)。不要恢复两个已结束入口。

单位边界：V1因单位声明检查在保存2行后停止，保留失败与partial，未知预取数量仅可界定在fit集合。全684 header数值shape/affine相同，但14例影像与标签均无单位声明；V2只改为原生体素坐标距离，原Dice/组件/插值不变，不猜毫米、不删例。访问日志685次标签读取尝试/完成对应唯一fit684；0开发/test。见[V1失败](archive/research_records/2026-09-08_fit_spatial_roundtrip/RESULT.md)、[V2修订](archive/research_records/2026-09-08_fit_spatial_roundtrip_v2/AMENDMENT.md)。

## 2026-09-08数据处理调研记录

用户质疑是否到上限并要求调研数据处理技术。本轮已核对当前代码与既有N4/Gamma/裁剪/160³负结果，阅读一手论文和官方实现；没有创建/恢复Goal，没有实施候选、训练或新增模型评测。

fit-only元数据新发现：684例裁剪尺寸中位136×174×142，均缩至128³，三轴相对缩放比中位1.2879、最大1.5；仅体素索引尺度，未验证header毫米spacing。缓存整WT最小228体素、684/684非空；不能据此说小连通灶未损失，也不能把几何扭曲当作已证明的精度瓶颈。

优先级：先fit标签/空间往返保真诊断；首个低部署成本训练候选为轻量弹性形变；保长宽比补边作独立输入对照；固定spacing块训练/盲滑窗是更高成本的分阶段路线。Nyúl/GMM/低分辨率/偏置场作有边界的技术储备，不盲目叠加，更不复活已失败的精确配方。完整[调研](archive/reviews/2026-09-08_segmentation_data_processing/RESEARCH.md)、[几何机器审计](archive/reviews/2026-09-08_segmentation_data_processing/FIT_GEOMETRY.json)、[三个候选七问](archive/research_records/2026-09-08_data_processing_candidates/IDEA.md)已归档并更新catalog。七问只是提案，参数/实现/共同评分网格尚未正式预注册，不能直接运行。

## GPT-6再次接手（2026-09-07，原30小时轮次已完成）

最新状态：原30小时科研工作已完成，最终计时检查108,030秒（30小时30秒），不计用户暂停/Fable时段。两项最新实验均完成、两个监督器exit0，两张A100已核为空闲。患者采样有开发Dice正证据，但Point安全门失败，未晋级；辅助监督当前0.1配方关闭。最新全仓814测试通过、三validator通过。完整交付见[30小时报告](archive/reviews/2026-09-07_30h_completion_audit/FINAL_REPORT.md)与[最终核验](archive/reviews/2026-09-07_30h_completion_audit/FINAL_VERIFICATION.json)。

用户批准的复用开发角色保持：原972池排除旧confirmation后864 scans/391人，固定fit684/313与dev180/78，与旧confirmation/fixed validation/test患者均零重叠；并非fresh/blind。见[授权记录](archive/research_records/2026-09-07_reused_development_roles/AUTHORIZATION.md)和[冻结split](archive/research_records/2026-09-07_reused_development_roles/SPLIT.json)。未来新会话不恢复已完成的监督器，不放宽本次门、不重测已关闭队列。

### 12:49 UTC结果：患者等概率采样有Dice收益，Point安全门失败

监督器12286已正常exit0。候选/预指定control：seed11 Scan0.753475/0.746762、Subject0.740633/0.733213、Point167/170；seed12为0.755906/0.744177、0.742725/0.729318、168/163。两seed平均Scan差+0.009221、Subject差+0.010413。Scan配对95%CI为[+.001555,+.012001]/[+.005221,+.019021]，Subject为[+.001531,+.013552]/[+.006627,+.020811]，四个区间都高于0，但唯一失败项是seed11 Point−3越过≥−1门。不能平均Point+1、挑seed12或放宽门后称整体晋级；当前精确策略按预注册关闭晋级，保留“分割改善、定位不稳定”的研究证据。

只改患者采样概率，aux=0、无教师、从匹配初值训练168轮；两seed全部增强RNG/121步匹配，导出error0。完整672个epoch-policy有效抽样SHA独立重建exact，患者覆盖约271→299，小体积抽样约45%→27%。两seed四个预定体积组Scan差都正；seed11 Point净损失3例在最小组，seed12该组反增3，不据子组改方法。720条保存点/真值计数与dev标签全exact，0新增审计forward。

新增训练4.028 GPU-hours、双卡workflow约2.04h、峰值20.664GiB；新增dev360 forwards/12.12秒，对照不重跑，不建新影像cache，部署4.70M/1×。两seed共用180 scans/78位reused development患者，非独立外部/盲确认；未新增test/outer/confirmation/fixed validation/external性能。证据：[RESULT](archive/research_records/2026-09-07_subject_balanced_sampling/RESULT.md)、[机器结果](archive/research_records/2026-09-07_subject_balanced_sampling/RESULT.json)、[验证](archive/research_records/2026-09-07_subject_balanced_sampling/VERIFICATION_RECEIPT.json)、[完整抽样重建](archive/reviews/2026-09-07_fit_subject_sampling/SAMPLER_REPLAY_FINAL.md)、[错误与点核查](archive/reviews/2026-09-07_development_error_description/SUBJECT_SAMPLING.md)。RAW SHA=96785e393fd1033684014fda091f2de1545b42a0fce6fe226f79a8a10630017d；RESULT SHA=cfd44c8e727cae2250898e8b74084025fa370ab5b228c65cc161cfb00eb165e3。不要重新启动已完成的workflow或用这些结果选新的temperature/mixture/epoch/seed。

### 10:44 UTC结果：辅助监督完成并关闭

监督器65466已正常exit0、四条168轮训练/720次开发前向/独立重算归档全部完成。切除腔辅助系数0.1在两seed均未过任何性能门：seed11 candidate/control Scan0.744044/0.746762、Subject0.729224/0.733213、Point166/170；seed12为0.739218/0.744177、0.724989/0.729318、161/163。两seed平均Scan差−0.003839、Subject差−0.004159；四个患者聚类配对Dice CI都跨0，不能称显著下降，但正向收益与Point安全门均失败。只关闭当前0.1配方，不搜权重、epoch或seed。

辅助任务确实学到（末10轮aux loss约0.15，对照约1），但没有转化为主任务开发收益；四条训练工程门和导出error0，故不是接线失败。成本8.064 GPU-hours/双卡两批约4.07h，峰值20.664GiB，缓存1.34GiB，部署4.70M/1×；开发720 forwards、23.12秒。访问仅reused development180/78，未新增fixed validation108/outer/confirmation/test270/external性能。开发不进入旧test榜。

证据：[RESULT](archive/research_records/2026-09-07_structured_background_aux/RESULT.md)、[机器结果](archive/research_records/2026-09-07_structured_background_aux/RESULT.json)、[验证](archive/research_records/2026-09-07_structured_background_aux/VERIFICATION_RECEIPT.json)、[漏画/多画描述](archive/reviews/2026-09-07_development_error_description/AUXILIARY.md)。两seed漏画比例平均均增加约0.005，多画变化方向不一致，不能简化成同一抑制机制。RESULT JSON SHA=a59c919f358890cfe1f7552796d8259b0088cff99522494489dadfa1ecf62c1e；raw SHA=80e9bd449d7c72a61de0ecdbdfbc5bb3a61814e7770bbcc1c983ac097b4cd2c6。后续患者等概率采样按揭示前方案执行并已完成，结果见上节；原30小时工作现已完成。

10:54补充证据：辅助监督720条保存点坐标/真值体素数与180份dev标签全部exact，命中数仍166/170、161/163；0新forward，明确重复读取开发标签，不证明未保存logit的argmax或预测mask重放。见[点核查](archive/reviews/2026-09-07_development_error_description/AUXILIARY_POINT_AUDIT.md)。采样同样核查已于12:50完成，720/720 exact。两正式实验和原30小时工作均已完成，逐项要求见[完成条件清单](archive/reviews/2026-09-07_30h_completion_audit/REQUIREMENTS.md)。

训练阶段的等待日志已移入[历史摘录](archive/reviews/2026-09-07_30h_completion_audit/HANDOFF_TRAINING_STAGE.md)，两个旧监督器65466/12286均已结束，不再恢复。完整结论、六模型同队列比较、成本与推断边界见[本轮研究总结](archive/research_records/2026-09-07_reused_development_program/RESULT.md)。本机额外MRI模态仍不可用，详见[fit模态文件审计](archive/reviews/2026-09-07_fit_modality_availability/RESULT.md)。

## Fable 8.5小时session成果（2026-09-06 13:32:17–22:02:17 UTC，实际连续工作8.5小时）

终态：无后台训练/评测/下载进程；两张A100空闲；磁盘约8.5G可用；全仓`760 passed`（新增6项面板合同测试）、doc layout/catalog/leaderboard三validator通过。/tmp中的评测cache（`/tmp/aflow-ucsd-brats2024test/cache_128`，index已归档为CACHE_INDEX_FINAL.json）与copy B可由workspace copy A+归档builder重建，丢失不影响证据链。下一session第一步仍是读本文件+榜单+方法百科；当前又回到"没有无需新用户权限即可开卷的新cohort"状态，最优先解除项不变（EGD374/CoRe-BT250用户DUA；发布者侧见阻塞登记表#3/4/5/10/14）。

- **新外部cohort完成并关闭：UCSD-BraTS2024test56。** TCIA UCSD-PTGBM V2（2026-01-22）公开了原BraTS-GLI 2024隐藏test的UCSD部分（package 1221，59 studies/42 subjects，V3 2026-03-13修正临床表patient ID）；这是2026-09-02 UCSD75审计中明确"未使用"的包。官方v3映射any-timepoint规则下仅`UCSD-PTGBM-0038`与1,350例union重叠（也是全表唯一跨ID歧义），像素前排除后冻结56 scans/41 subjects，与union及已评UCSD75患者零重叠。七问/预注册冻结后：双独立传输112文件SHA全exact（包无`.sums`）、标签域56/56合规无空WT、零WT crop丢失56/56、首例byte exact、smoke/preaccess全过，唯一504 forwards。结果：Full972 KD168 Flip8/identity/GCSAM184 Scan=`0.732973/0.730002/0.717689`、Subject=`0.733625/0.730740/0.715545`、Point=`56/56、56/56、52/56`。Flip8−identity=`+0.002971` CI跨0（第六个正点估计cohort）；**GCSAM−Full=`−0.012312`，scan CI `[-0.025822,-0.000092]`与subject CI `[-0.031507,-0.000268]`在七个已揭示cohort中首次整体为负，Point−4**。身份：public post-challenge、非hidden blind、patient-disjoint by official mapping、**非institution-disjoint**（UCSD为BraTS2024七站之一），独立分榜。所有统计已独立重算、全部输入重hash exact。证据[结果](archive/research_records/2026-09-06_ucsd_brats2024test_frozen_panel/RESULT.md)、[raw](archive/research_records/2026-09-06_ucsd_brats2024test_frozen_panel/RAW_UCSD_BRATS2024TEST56.json)、[验证](archive/research_records/2026-09-06_ucsd_brats2024test_frozen_panel/VERIFICATION_RECEIPT.json)与[来源审计](archive/reviews/2026-09-06_ucsd_brats2024test_source_audit/RESULT.md)。
- **claim措辞再收窄（2026-09-07）：** GCSAM点估计方向依cohort变化；UCSD56原冻结Scan/Subject CI虽略负，但离线数值精度审计显示20组10,000次Scan区间17组跨0、百万次CI[-.025714,+.000140]；Subject4/20跨0、百万次仍略负[-.031763,-.000223]。原分数/区间/榜单排名保留，不再称两项“稳健确定反向”，也不以事后seed替换主检验。Full该队列Point仍56/56。禁止回流选择；见[bootstrap数值精度](archive/reviews/2026-09-07_ucsd56_bootstrap_precision/RESULT.md)。
- **外部阻塞监测（本轮复查，均未解除）：** Africa仍V1（无2024扩展）；UTSW仍V1无3/5字典；UCSF-PDGM仍v5仅BraTS21映射；UPENN仍V2无官方manifest。BRISC新候选经审为2D JPEG切片集，关闭。
- **磁盘回收：** 按用户既有授权并复核2026-09-02审计（superset/subset index SHA均exact）后，删除已证明byte-exact冗余的`workspace/BraTS/cache/t1ce_wt_n4_braincrop_resolution_20260818/formal_128/`（7,927,512,416 bytes，可由raw+归档builder重建）；可用1.4G→8.8G。972例superset、raw、checkpoint、报告全部保留。
- session记录见[本轮SESSION](archive/reviews/2026-09-06_fable_8p5h_session/SESSION.md)。

## Fable接手快照（2026-09-06 13:25 UTC）

用户在原30小时目标运行约23.27小时后主动暂停Codex，要求改由Fable持续工作、后续可交替接力。当前没有后台训练、评测、下载或GPU任务；两张A100在最后一次检查时空闲。不要寻找或恢复所谓“未跑完的OpenNeuro formal”：该路线已经合法终止。

Fable开始顺序：

1. 先读[`AGENTS.md`](AGENTS.md)、本文件、[`AFLOW_BRATS_LEADERBOARD.md`](AFLOW_BRATS_LEADERBOARD.md)、[`AFLOW_BRATS_METHOD_CATALOG.md`](AFLOW_BRATS_METHOD_CATALOG.md)和[本轮SESSION](archive/reviews/2026-09-05_30h_chain_audit/SESSION.md)。
2. 把[外部阻塞登记表](archive/reviews/2026-09-05_30h_chain_audit/EXTERNAL_BLOCKER_REGISTER.md)当唯一来源队列。当前用户可解除的前两项是EGD374与CoRe-BT250的机构Data Use Agreement（数据使用协议，DUA）；**现有授权不包含代用户填表、发邮件或签协议**。
3. OpenNeuro-GBM334已完成334例全源输入预检并关闭：332例过，BO118/KA86各有4/6个WT体素在image-positive crop外；0 external model/0性能。[结果](archive/research_records/2026-09-06_openneuro_gbm334_frozen_panel/RESULT.md)。禁止运行已准备但未获资格的`freeze/evaluate/finalize_openneuro_gbm334_*`入口；其冻结器会因失败收据自行拒绝。
4. IMAGO release1.0不能作人工WT基准：mask为PICTURE自动输出后由医师accept/reject，且需DUA。[审计](archive/reviews/2026-09-06_imago500_source_feasibility/RESULT.md)。CoRe-BT250标签流程合格但DUA阻塞，[审计](archive/reviews/2026-09-06_corebt250_source_feasibility/RESULT.md)。
5. Immutable-role内部fresh容量仍仅2 subjects/3 scans；未经用户明确改变数据角色，不得启动新train-only方法。不要用已揭示external/test选择loss、阈值、padding、view、router或病例。

最终校验状态：全repo `754 passed, 27 warnings`（仅既有CUDA沙箱与SciPy弃用警告）；root layout通过，根Markdown 9个、相对链接479个；method catalog validator通过；leaderboard validator通过，既有五个外部分榜与internal/validation行一致。交接时没有后台科研进程。

## GPT-6接手：用户明确的30小时长周期授权

- **2026-09-05新30小时轮次进行中：** 起点08:02:39 UTC，计划截止2026-09-06 14:02:39 UTC。N4家族no-N4/Otsu未过Stage A，多条件链关闭；随后完成UCSD75 official-float、Africa95、CPTAC33、WAW preop152与patient-disjoint WAW early-postop31。GCSAM−Full只有Africa `+0.068617` CI全正，CPTAC/WAW preop/postop均小正且跨0；Flip8在postop31 `+0.007312`也跨0。五外部/来源cohort关闭不调参；继续30小时证据审视。进度见[本轮审视](archive/reviews/2026-09-05_30h_chain_audit/SESSION.md)。
- **OpenNeuro-GBM334在全输入零裁剪损失门关闭，0性能：** 2026-08-07 Scientific Data新数据`ds007045:1.0.1`含337名八医院术前原发GBM，BraTS 0/1/2/3标签全部由神经放射科医师复核，87例小修、11例大幅人工修订。版本化未N4 MNI T1-CE与最终mask严格交集334；BO199/NSK07缺mask、BO97缺对应处理影像，在像素前排除。334对约2.03GiB且有git-annex SHA/versionId；八站不在BraTS2024七站，但无patient桥。七问/预注册/5 tests、首例双N4 byte exact及全量12路预检后，332例通过；BO118有4/348742、KA86有6/97700个WT体素落在image-positive crop外。按预注册0丢失门停止，未生成MODEL_PREACCESS，0外部checkpoint/model/性能；不排例、裁GT、GT扩框、搜padding或换variant。见[结果](archive/research_records/2026-09-06_openneuro_gbm334_frozen_panel/RESULT.md)、[输入收据](archive/research_records/2026-09-06_openneuro_gbm334_frozen_panel/INPUT_PREFLIGHT.json)与[来源](archive/reviews/2026-09-06_openneuro_ds007045_source_audit/RESULT.md)。
- **IMAGO500规模大但不是人工WT基准，并受DUA限制：** 阿姆斯特丹首版500人全有术前四模态，381人有14天内术后；但一手论文明确249份术前/190份术后mask全部由PICTURE自动生成，神经放射科医师只做接受/剔除，没有逐体素人工修订；术后模型还只画强化残留。用其算Dice会把另一个模型输出当真值并带选择偏差。EUCAIM/Health-RI又要求用户注册、negotiator与DUA，未代签/发邮件。0清单/payload/model，当前外部榜路线关闭；见[来源审计](archive/reviews/2026-09-06_imago500_source_feasibility/RESULT.md)。
- **CoRe-BT250是新高优先级人工修订候选，但需用户/UW DUA：** 华盛顿大学2023–2025共310名胶质瘤，250人完整术前T1/T1c/T2/FLAIR；完整组mask由BraTS2023优胜ensemble初始化，再由有经验标注者逐例修正edema/enhancing/necrotic错误，可组成WT。UW不在BraTS2024七站但无patient桥。官方要求Google form、联系负责人、pre-approval和用户/机构签署Data Use Agreement；未代提交/发邮件/签署，0清单/payload/model。若用户解除，先审实际mask数、label字典、recurrent混入、overlap和checksum；见[来源审计](archive/reviews/2026-09-06_corebt250_source_feasibility/RESULT.md)。

- **N4家族Stage A完整结果：两候选均关闭，不因局部正向放宽门。** no-N4在E折identity/Flip8 Scan delta=`+0.018044/+0.017634`，E门全过；F折仅`+0.000597/+0.000187`，identity Point`−2`且四个Dice CI下界最低`−0.009649`，所以失败。跨折Scan均值`+0.009320/+0.008910`不能掩盖预注册单折失败。Otsu在E折小正但CI门失败；F identity Scan/Subject=`−0.009235/−0.007148`且CI整体负，F Flip8也负。两候选raw SHA=`71b09584…/f1ae6df1…`；共享/候选各约14.426/6.783 GPU-hours、峰值20.167GiB。只访问E42/17与F42/19 train-only development；fixed validation/outer/confirmation/test270/UCSD/RHUH/UTSW模型均未访问。证据[no-N4结果](archive/research_records/2026-09-05_no_n4_clean_teacher_replication/RESULT.md)、[Otsu结果](archive/research_records/2026-09-05_n4_otsu_clean_replication/RESULT.md)与[对比图](archive/reviews/2026-09-05_30h_chain_audit/N4_FAMILY_STAGE_A_COMPARISON.png)。

- **条件后续全部按门停止：** Full972 admission在剩余14.46h且8h门通过时从raw重算eligible为空，`automatic_training_launched=false`、0CUDA；RHUH no-N4在EXTERNAL_PREACCESS/标签/GPU前exit0；UTSW no-N4在preaccess/其余359例/cache/model前`closed_before_metric`。UTSW来源、主FeTS no-N4/canonical/Otsu固定3例输入桥仍保留为工程资产，但不得换旧checkpoint绕过失败。证据[admission](archive/research_records/2026-09-05_no_n4_clean_teacher_replication/FULL972_ADMISSION_CHECK.json)、[RHUH终态](archive/research_records/2026-09-05_rhuh_no_n4_input_bridge/EXTERNAL_QUEUE_TERMINAL.json)、[UTSW终态](archive/research_records/2026-09-05_utsw_no_n4_paired_external/QUEUE_TERMINAL.json)。

- **UTSW冻结baseline也在模型前因官方标签域关闭：** KD168/GCSAM184预先冻结、synthetic smoke通过；724路径首传因Aspera source上限失败且暴露`None==None`计数bug，固定100路径×8批后双源724/724 exact。cache首例BT0007触发`{0,1,2,4}`门；全362人工标签union实际`{0,1,2,3,4,5}`，3见116例、5见5例/434体素。固定35异常病例双下载非`r`原始manual标签仍同样含3/5，排除“仅r重采样产生”。论文无3/5语义，不能猜positive union；0UTSW模型/指标，partial cache约4KB blocks保留。证据[结果](archive/research_records/2026-09-05_utsw_frozen_baseline_external/RESULT.md)。

- **UCSD75官方float32来源纠正完成且关闭：** 在七问/PREREG、15项新+历史测试、synthetic GPU smoke、75映射/150源SHA和模型前收据冻结后，只把同75例影像从Pittsburgh Fiber Data Hub uint8镜像换为TCIA包1019直接float32；crop/128³标签75/75 exact，128³影像0/75 exact。Official-float上GCSAM184 Scan/Subject/Point=`0.755890/0.755312/70`，Full972 KD168=`0.746255/0.740165/70`；相对mirror分别=`+0.000242/+0.000160/−1`与`+0.000701/+0.001110/0`，四个来源Dice CI均跨0。训练0、cache CPU868.8秒、150 forwards、峰值约4.86GiB。它是同cohort reused source correction、not pristine blind、不是新外部确认；未访问internal test270/fixed validation/outer/confirmation，不回流选择。证据[结果](archive/research_records/2026-09-06_ucsd_official_float_source_correction/RESULT.md)、[raw](archive/research_records/2026-09-06_ucsd_official_float_source_correction/RAW_OFFICIAL_FLOAT.json)与[独立重算](archive/research_records/2026-09-06_ucsd_official_float_source_correction/VERIFICATION_RECEIPT.json)。

- **EGD374人工二值WT来源合格但外部协调阻塞：** 一手论文确认Erasmus Glioma Database有774位术前患者、全四模态，374位由四位观察者之一基于T2/FLAIR勾画binary WT（1 tumor/0 background），400位AUTO明确不能混入人工榜。官方Health-RI XNAT项目metadata可匿名读，但description要求邮件联系service desk、签Data Usage Agreement后建账号；匿名subjects API HTTP403。当前0 subject IDs/影像/标签/模型/GPU，患者重叠也未证明。用户/机构完成签署后应先做manual+quality+BraTS overlap metadata审计，不能由Codex代签或用AUTO绕过。证据[来源结果](archive/reviews/2026-09-06_egd_source_feasibility/RESULT.md)。

- **UCSF-PDGM标签合格但无法证明对BraTS2024患者独立：** 官方501条/495位术前患者的多区分割由BraTS ensemble初始化、放射科医生修订、2位专家批准；v5 metadata明确298条已映射BraTS21。合并官方6组随访后296/495患者至少一时点已映射，表面余199人；但UCSF本身是BraTS2024七个来源站之一，官方只给2021映射，没有PDGM↔2024 patient crosswalk。故不能把空字段当零重叠；0 package token/inventory/NIfTI/标签/模型停止。证据[来源结果](archive/reviews/2026-09-06_ucsf_pdgm_external_feasibility/RESULT.md)。

- **BraTS-Africa95新外部评测完成，揭示强跨域Dice反转：** TCIA package948目录为146×5文件；在payload前按发布者类别冻结`95_Glioma`、排除51 OtherNeoplasms。双源190/190 exact、全95 label `{0,1,2,3}`/geometry/cache/单线程首例复现/模型前收据过后，只运行Full KD168 identity+Flip8和GCSAM184 identity共855 forwards。GCSAM Scan=Subject/Point=`0.591619/84/95`，Full identity=`0.523003/87/95`，Full Flip8=`0.526438/88/95`；GCSAM−Full Dice=`+0.068617`，CI `[+0.044472,+0.092498]`，Point−3；Flip8−identity=`+0.003435`，CI跨0，Point+1。三者普遍低估WT，GCSAM预测范围较大。训练0、source533MiB、cache951MiB、正式17.15秒/峰值5.08GiB。实际95例按datainfo为四中心，不是raw最初写的六中心；显式更正、不改分数。Public post-challenge/not hidden blind、institution-disjoint但无patient-ID bridge；未访问旧cohort，已更新独立榜单并关闭。V3又重hash双源380文件/555.26MB、双checkpoint/evaluator/raw/cache/榜单全过。证据[结果](archive/research_records/2026-09-06_brats_africa95_frozen_baselines/RESULT.md)、[raw](archive/research_records/2026-09-06_brats_africa95_frozen_baselines/RAW_EXTERNAL95.json)、[终态验证V3](archive/research_records/2026-09-06_brats_africa95_frozen_baselines/VERIFICATION_RECEIPT_V3.json)与[来源](archive/reviews/2026-09-06_brats_africa_source_audit/RESULT.md)。

- **Africa95事后误差边界：GCSAM主要改善中大型WT，小病灶未解决。** 按target voxels排序的四分位，GCSAM−Full identity Dice=`+0.0039/+0.0820/+0.0969/+0.0926`，相对优势与log10体积ρ=`0.321`；Point相对Full丢4救1。三模型Africa预测/真实体积比约0.50–0.61，而UCSD约1.09–1.14；跨cohort治疗阶段/标签/采集/grid共变，不能归因单因或调阈值。GCSAM最后head bias还更负，排除简单整体bias抬高解释。禁止在已揭示Africa按体积或中心设计router；证据[误差审计](archive/research_records/2026-09-06_brats_africa95_frozen_baselines/POSTHOC_ERROR_AUDIT.json)与[图](archive/research_records/2026-09-06_brats_africa95_frozen_baselines/POSTHOC_ERROR_AUDIT.png)。

- **Africa95 Dice差主要是recall而非precision交换：** 保存计数重算Full identity precision/recall=`0.823/0.416`，GCSAM=`0.819/0.497`。GCSAM−Full recall=`+0.0809`且事后CI `[+0.0548,+0.1070]`，precision=`−0.0038`且CI跨0；平均多预测11,720 voxels。Flip8 precision=`+0.0333`但recall=`+0.0025`且CI跨0，平均少887 voxels。该分解解释已冻结结果，不是promotion或threshold证据；见[precision/recall审计](archive/research_records/2026-09-06_brats_africa95_frozen_baselines/POSTHOC_PRECISION_RECALL.json)。

- **Africa95多数WT处于训练罕见大范围：** 其target在train972经验分位的中位为95.47%；62/95高于train p90、54/95高于p95、1例超过train最大。按train p75=50,740预定边界，以上72例GCSAM−Full=`+0.0893`、以下23例仅`+0.0040`。这与coverage解释一致，但治疗阶段/标签/采集共变，严禁把train分位变成Africa-tuned router。证据[训练支持审计](archive/research_records/2026-09-06_brats_africa95_frozen_baselines/POSTHOC_TRAIN_TARGET_SUPPORT.json)。

- **Africa预测向训练size prior回缩：** Africa target在train分位中位95.47%，Full/GCSAM预测体积仅落在76.13%/85.70%；真实62例高于train p90，预测只有19/38例。GCSAM扩张更多但仍不足。该模式与learned size prior相容，不是因果；术前all-positive与训练RC-excluded WT也不同。证据[回缩审计](archive/research_records/2026-09-06_brats_africa95_frozen_baselines/POSTHOC_TRAIN_PRIOR_REGRESSION.json)。

- **中心聚类敏感性不改变主区间：** GCSAM−Full删任一中心后的Dice仍`+0.0654–+0.0707`、四中心均值全正；Flip8删NKDC后变负且中心bootstrap跨0。说明GCSAM方向不由一个已见中心独占，Flip8中心不稳；但仅4中心，不能外推所有非洲医院。主证据仍是预注册患者配对CI。证据[中心敏感性](archive/research_records/2026-09-06_brats_africa95_frozen_baselines/POSTHOC_CENTER_CLUSTER_SENSITIVITY.json)。

- **同一1× checkpoint跨cohort模式不单调：** GCSAM−Full Scan在internal/UCSD/Africa/CPTAC/WAW-pre/WAW-post为`+0.000244/+0.009634/+0.068617/+0.001954/+0.010223/+0.001657`，只有Africa CI全正，其余跨0；Point差`−10/0/−3/0/−1/−1`。Africa只支持队列特异checkpoint优势，不能证明优化器配方；六cohort不可混排。见[综合报告](archive/reviews/2026-09-06_cross_cohort_checkpoint_synthesis/RESULT.md)与[V4图](archive/reviews/2026-09-06_cross_cohort_checkpoint_synthesis/COMPARISON_V4.png)。
- **GCSAM checkpoint跨五外部有coverage倾向，但非全面/因果优势：** 预测体积与recall点估计5/5均正，precision仅2/5正、Point从未增加。Recall CI在Africa/CPTAC全正；预测体素CI在Africa/CPTAC/WAW-pre全正。CPTAC precision降`0.0235`、recall升`0.0223`。Coverage V1曾把empty-pred precision记0，V2按既有约定记1，仅把Africa precision差纠正`−0.0143→−0.00380`，其他结论不变。不证明optimizer或授权router/threshold。见[综合](archive/reviews/2026-09-06_gcsam_coverage_synthesis/RESULT.md)与[V2图](archive/reviews/2026-09-06_gcsam_coverage_synthesis/COMPARISON_V2.png)。

- **Africa差不能因果归给GCSAM optimizer：** GCSAM是fit864+inner选epoch184+另一seed，Full是fit972+固定168；训练病例数、选择、epoch、seed、optimizer同时不同。论文证据又是2D分类而非3D分割。Africa只支持该checkpoint的域特异Dice/recall，不改变fixed184双复制失败或授权重训。见[一手理论/因果边界](archive/reviews/2026-09-06_gcsam_external_interpretation/RESULT.md)。

- **内部fresh证据容量已耗尽：** 972 scans/444 subjects中，按E/F既有冻结规则合并历史inner/selector/confirmation后已排除406人；E/F前仅38人/87 scans，E/F互斥使用36人/84 scans。当前未作为评测揭示的只剩2人/3 scans（`02633`一例、`02870`两例），无法建立一个42-scan fold或双复制。除非用户明确改变目标并授权role reassignment，否则不得复用E/F、把旧fit改称fresh validation或启动无新泛化证据的训练。证据[容量审计](archive/reviews/2026-09-06_fresh_evidence_capacity/RESULT.md)。

- **五份冻结cache显示Africa最极端，体积仍不能单因解释准确率：** 相对train972，UCSD/Africa/CPTAC/WAW的WT mean比=`1.16/2.87/2.13/1.47`、median比=`1.21/4.09/2.89/1.72`；全体素std median差=`+0.014/+0.052/+0.025/+0.019`，非零占比差=`+0.020/+0.038/+0.016/+0.013`。Africa三轴最右移；WAW中度，CPTAC target比WAW更大但Dice明显更高。输入支持、强度、标签、阶段、采集和发布者预处理共变，不能用体积解释或校准。证据[V2域审计](archive/reviews/2026-09-06_fresh_evidence_capacity/INPUT_DOMAIN_SHIFT_AUDIT_V2.json)与[图](archive/reviews/2026-09-06_fresh_evidence_capacity/INPUT_DOMAIN_SHIFT_AUDIT_V2.png)。

- **既有native-volume sampler明显偏小WT：** 972例inverse-sqrt体积权重下，最小/最大四分位占采样概率44.0%/12.7%，每例每轮预期1.75/0.506次；最大四分位平均60.4%概率单轮未抽到。它历史上恢复Point，可能与Africa大WT漏分方向相容；但Full/GCSAM都使用该轴，不能解释二者因果差，也不授权按Africa重配权重。证据[采样审计](archive/reviews/2026-09-06_fresh_evidence_capacity/TRAIN_SAMPLING_EXPOSURE_AUDIT.json)。

- 用户要求GPT-6接手后连续科研30小时，不因单次负结果或等待训练而提前停止；两张A100、充足credit和联网检索均获授权。

- **训练源provenance限制仍未解决，development授权不等于官方parity：** 2026-09-05复查`syn60086071`仍为corrected-v2/file handle143397620，匿名下载HTTP403，唯一DOWNLOAD门为SelfSign requirement9606685；不要求certified/validated profile，但必须登录用户自行同意。本机无Synapse CLI/client，未代用户授权或下载。历史用户授权只在development fit/tune边界覆盖旧执行阻塞；本地Kaggle version2三包1350 scans/613 subjects仍未与官方zip MD5=`1d910b17…`及逐member SHA核验。当前E/F/Full972与继承checkpoint必须标“镜像来源、官方字节parity未证明”；不自动授权confirmatory/test。证据[本轮复查](archive/reviews/2026-09-05_brats2024_source_provenance_recheck/RESULT.md)。
- **外部阻塞已统一登记，避免重复空转：** EGD374和CoRe-BT250需用户/机构DUA；UTSW需发布者3/5映射；Africa2024需新公开版本；UCSF-PDGM需2024患者桥；BraTS训练源需用户Synapse自签；ReMIND/MOTUM/OpenNeuro需一致图像支持或独立过门模型链；IMAGO是自动标签且需DUA；RHUH需兼容且过门模型链；内部只剩2人/3 scans。解除主体和第一步见[阻塞登记表](archive/reviews/2026-09-05_30h_chain_audit/EXTERNAL_BLOCKER_REGISTER.md)。该表不授权发邮件、签协议或模型访问。

- **Burdenko-GBM-Progression治疗时点合适但不是WT标签：** 官方180人、1–8随访，前两次约放疗后3周/2–3月；但RTSTRUCT为GTV/CTV/PTV且随访GTV仅子集，MRI原空间、轮廓在topometric CT且无注册文件。放疗靶区不等于BraTS WT，几何容器也不能修复语义；0 manifest/临床表/主体/模型停止。证据[来源审计](archive/reviews/2026-09-06_burdenko_source_feasibility/RESULT.md)。

- **CPTAC33 Africa-informed确认完成，GCSAM强差不复现、Flip8方向独立支持：** 官方映射CPTAC39中Training33有seg、Validation6无seg；原39门失败后像素前另冻33。Package636 `.sums`407,245 MD5 rows，选中66路径官方MD5/size/双源SHA exact；全33 raw `{0,1,2,4}`/cache/首例门过。Full identity/Flip8/GCSAM=`0.841331/0.844577/0.843285`，Point全33。GCSAM−Full=`+0.001954`、CI跨0、16胜17负，Africa强优势仅弱/不确定复现；Flip8−identity=`+0.003246`、CI `[+0.000828,+0.005855]`全正、23胜10负。训练0、cache288秒、297 forwards6.55秒/5.08GiB。Public Training external-to-project/not hidden blind、source-disjoint/no patient bridge；cohort关闭。证据[结果](archive/research_records/2026-09-06_cptac33_africa_informed_confirmation/RESULT.md)、[raw](archive/research_records/2026-09-06_cptac33_africa_informed_confirmation/RAW_CPTAC33.json)、[验证](archive/research_records/2026-09-06_cptac33_africa_informed_confirmation/VERIFICATION_RECEIPT.json)与[来源](archive/reviews/2026-09-06_cptac_brats2021_source_audit/RESULT.md)。
- **固定Flip8已有五cohort同向，三处Scan区间支持，但不是免费收益：** Flip8−identity Scan在internal/Africa/CPTAC/WAW-pre/WAW-post为`+0.005510/+0.003435/+0.003246/+0.010116/+0.007312`，Point=`+3/+1/0/+2/0`；internal/CPTAC/WAW-pre CI全正，Africa/WAW-post跨0。各cohort不可pool；部署仍8×，不得调视图/权重/阈值。见[综合报告](archive/reviews/2026-09-06_flip8_cross_cohort_synthesis/RESULT.md)与[V3图](archive/reviews/2026-09-06_flip8_cross_cohort_synthesis/COMPARISON_V3.png)。
- **Flip8的外部误差分解更像保守去假阳性，而非稳定补漏：** 在有完整保存计数的Africa/CPTAC/WAW-pre/WAW-post中，Flip8−identity precision点估计4/4为正，Africa和两个WAW队列的事后CI全正；recall仅CPTAC CI全正，WAW-post甚至点估计略负。预测体积仅CPTAC点估计增加，WAW两队列明确减少；Point差均不为负。这支持八镜像共识通常减少偶然多画的有限描述，不代表所有Dice显著或免费收益。四队列分列、不pool、不授权调视图/权重/阈值；见[误差机制综合](archive/reviews/2026-09-06_flip8_overlap_synthesis/RESULT.md)与[图](archive/reviews/2026-09-06_flip8_overlap_synthesis/COMPARISON.png)。
- **WAW-GliomAI152新外部完成并关闭：** Zenodo final record于2026-09-01开放、CC BY 4.0，来自Medical University of Warsaw。描述156 preop但final T1c-mask交集152人，原生geometry exact、label union `{0,1,2,3}`、grade2/3/4=`44/34/74`。官方2.95GiB大包MD5/CRC、152源、首例、cache、smoke、模型前门均过后唯一1368 forwards：GCSAM/Full Flip8/Full identity Dice=`0.693601/0.693494/0.683378`，Point=`140/143/141`。GCSAM−Full=`+0.010223` CI跨0、Point−1；Flip8−identity=`+0.010116` CI全正、Point+2。第一次formal字段KeyError在checkpoint前0forward，V2后唯一访问。终态重hash大包/304源/checkpoints/cache/bootstrap/榜单全过。Public/institution-disjoint no patient bridge/not hidden blind；cohort关闭。证据[结果](archive/research_records/2026-09-06_waw_gliomai152_frozen_panel/RESULT.md)、[raw](archive/research_records/2026-09-06_waw_gliomai152_frozen_panel/RAW_WAW152.json)、[验证](archive/research_records/2026-09-06_waw_gliomai152_frozen_panel/VERIFICATION_RECEIPT.json)与[来源](archive/reviews/2026-09-06_waw_gliomai_source_audit/RESULT.md)。
- **WAW事后分解：Flip8主要提高precision，GCSAM机制仍不确定。** 按“空预测precision=1”的项目既有约定，Full identity/Flip8/GCSAM平均precision=`0.801/0.842/0.799`、recall=`0.635/0.636/0.641`。Flip8 precision差`+0.04185`、事后CI `[+0.01957,+0.06881]`全正，recall差近0且跨0，平均少预测763 voxels；GCSAM precision/recall差均跨0。旧V1误把空预测precision记0，V2只修正该描述统计，Dice/Point/正式门不变。WAW target在train972经验分位中位68.31%，44/152高于p90、0超过最大，远不如Africa极端；GCSAM最大体积四分位`+0.0272`但总体ρ仅0.125，grade2/3/4差也变号。禁止按体积/grade/scanner建router；见[误差审计V2](archive/research_records/2026-09-06_waw_gliomai152_frozen_panel/POSTHOC_ERROR_AUDIT_V2.json)与[图V2](archive/research_records/2026-09-06_waw_gliomai152_frozen_panel/POSTHOC_ERROR_AUDIT_V2.png)。
- **WAW尾部暴露T1ce-only的目标可观测性限制：** Full/Flip8/GCSAM分别5/6/5例Dice0、20/19/18例Dice<0.5；共同5个零Dice全为grade2，其中4例只有FLAIR abnormality label2、无enhancing label3/necrotic label1。有/无label3的平均Dice约`0.70–0.71` vs `0.61–0.62`。Point命中时Dice明显更高但仍非同一目标。该分析只说明仅T1增强输入难观察FLAIR定义区域，不授权用真值label/grade选模型、排例或router。见[尾部审计](archive/research_records/2026-09-06_waw_gliomai152_frozen_panel/POSTHOC_FAILURE_TAIL.json)。
- **Train并非没见过无强化，但纯FLAIR-only覆盖明显不足：** Train972 label3缺失269例/27.7%，高于WAW29/19.1%；真正差异是only `{0,2}`：train30/972 scans、18/444 subjects（3.09%/4.05%），WAW26/152（17.11%），约5.54×。Train许多无label3仍有label1或切除腔label4提供结构线索，WAW术前无label4。与共同零Dice相容但不证明因果，也不授权在已见WAW设计重采样。见[可观测性审计](archive/reviews/2026-09-06_t1ce_observability_audit/RESULT.md)。
- **一手文献限定可观测性措辞：** BraTS2024官方SNFH本就由T2/FLAIR异常定义；独立模态消融显示T1CE偏ET/TC，WT的最小充分输入更依赖FLAIR或T1CE+FLAIR。但另一研究T1CE-only WT仍可达较高Dice、加FLAIR再升，故只能说平均信息与纯FLAIR尾部受限，不能说绝对不可能或把别队列分数搬来。见[文献边界](archive/reviews/2026-09-06_t1ce_observability_audit/LITERATURE_CONTEXT.md)。
- **WAW GCSAM点估计受dominant scanner组成影响：** SIGNA Architect112例中GCSAM−Full=`+0.01365`，其余40例仅`+0.00063`；3T/1.5T为`+0.01611/−0.00341`。Flip8在Architect/其他与3T/1.5T均正。Scanner、场强、grade和label组成共变，且全为事后，故只进一步支持GCSAM结果依队列/组成变化，不授权scanner router。见[描述](archive/research_records/2026-09-06_waw_gliomai152_frozen_panel/POSTHOC_SCANNER_DESCRIPTION.json)。
- **WAW early-postop only-patient31完成：** Final160/92中排57 preop-overlap患者；35 post-only里排4 blank-day，再对31人/48 numeric scans取每人最早一张，day0–22、29名grade4、preop patient overlap0。官方postop3.322GB MD5/CRC、31源、cache/smoke/preaccess全过。Full identity/Flip8/GCSAM Dice=`0.591647/0.598959/0.593305`，Point=`15/15/14`。Flip8−identity=`+0.007312` CI跨0，GCSAM−Full=`+0.001657` CI跨0，均positive-but-uncertain。训练0、279 forwards5.76秒/5.08GiB；public early post-resection、非hidden、cohort关闭。证据[结果](archive/research_records/2026-09-06_waw_postop_only31_frozen_panel/RESULT.md)、[raw](archive/research_records/2026-09-06_waw_postop_only31_frozen_panel/RAW_WAW_POSTOP31.json)、[验证](archive/research_records/2026-09-06_waw_postop_only31_frozen_panel/VERIFICATION_RECEIPT.json)与[来源](archive/reviews/2026-09-06_waw_postop_only31_source_audit/RESULT.md)。
- **WAW postop Point miss主要落切除腔：** 离线重建128³多类label并与正式WT cache31/31 exact。Full/Flip8 Point分类均WT15/RC13/普通背景3，GCSAM为14/14/3；约80%以上miss被GT RC吸引。Full/Flip/GCSAM precision=`.567/.587/.570`、recall=`.658/.654/.665`。这解释Dice约.59但Point约.45–.48；GT RC部署不可用，禁止RC避让后处理。证据[审计](archive/research_records/2026-09-06_waw_postop_only31_frozen_panel/POSTHOC_ERRORS.json)与[图](archive/research_records/2026-09-06_waw_postop_only31_frozen_panel/POSTHOC_ERRORS.png)。
- **RC吸引集中在day0–1但仅可描述：** 24个day0–1病例的Full/Flip Point为WT9/RC12/背景3，7个day>1为6/1/0；GCSAM为9/12/3与5/2/0。Dice没有同幅变化，说明Point是独立失败轴。后组仅7例且患者/手术共变，禁止day router或恢复轨迹因果。证据同[posthoc](archive/research_records/2026-09-06_waw_postop_only31_frozen_panel/POSTHOC_ERRORS.json)。
- **WAW pre/post粗域统计近似，不能以体积解释Point崩塌：** Postop31相对preop152 WT mean/median=`1.017×/1.071×`，cache std/nonzero fraction median只差`+0.000334/+0.00249`；但Point低约一半且miss集中RC。两组患者零重叠且grade组成不同，不能作前后因果，只排除简单全局体积/尺度解释。见[addendum](archive/reviews/2026-09-06_fresh_evidence_capacity/WAW_POSTOP31_DOMAIN_ADDENDUM.json)与[图](archive/reviews/2026-09-06_fresh_evidence_capacity/WAW_POSTOP31_DOMAIN_ADDENDUM.png)。
- **Grade4-only仍有pre/post大差，但非配对因果：** WAW pre74/post29 grade4的Full/Flip/GCSAM Dice=`.730/.745/.737` vs `.586/.594/.586`，Point近100%/97% vs45–48%；post target median反小13,546.5 voxels。Grade/大WT不是充分解释；两组患者不同且post28/29有RC、pre0，禁止称手术因果。见[描述](archive/research_records/2026-09-06_waw_postop_only31_frozen_panel/POSTHOC_GRADE4_PRE_POST.json)。
- **一手文献支持RC是独立且易混淆的术后结构：** BraTS2024把recent/chronic RC单列label4；340例术后GBM研究报告RC/CSF/坏死核心混淆，MU数据论文也记录RC内T1高信号被误作肿瘤。与本次argmax落RC相容但非因果；显式RC head/CSF/surgery信息都是新模型，不能在已见31例补。见[文献边界](archive/research_records/2026-09-06_waw_postop_only31_frozen_panel/RC_LITERATURE_CONTEXT.md)。
- **Point是诊断，不是当前direct mask的输入：** 当前无SAM，mask由`logit>0`直接输出；argmax落RC不会反过来改变Dice。它警告未来自动prompt/交互链在早期术后有风险，不能误写成当前Dice由错误Point导致。
- **RC不是训练完全没见过，而是二值监督把它折叠为普通背景：** Train972有814/972 scans（83.74%）含label4，WAW-post31为30/31（96.77%）；当前single-channel WT却把RC和普通背景都编码0。大量暴露不等于显式学会特殊结构，与81–82% Point miss落RC相容。显式RC/CSF/surgery-aware输出仍未验证，需新fresh role；当前不能用GT避让或训练。见[监督缺口](archive/reviews/2026-09-06_rc_supervision_gap/RESULT.md)。
- **MOTUM-HGG29合法来源但当前输入链在模型前关闭：** G-Node DOI快照CC BY4.0，双984,434,960-byte传输SHA/CRC exact；Participants在性能前固定29名grade3/4原发胶质瘤、排除38转移瘤。29×T1ce+双语义mask native geometry exact，21/24层、5.5–6.75mm。旧独立canonical在2例把微小header误差放大越`1e-5`，0cache/模型关闭；另立共享离散方向桥后29/29双进程arrays/shared affine exact且0插值。但新性能协议首例发现image-positive bbox丢892/21007 WT（4.246%）；全29仅此1例。官方HD-BET+FLIRT spline解释负值来源，但`image!=0` bbox仍丢同892，继续只剩GT扩框/事后padding/丢例，均禁止。MOTUM性能始终0，路线关闭。证据[来源](archive/reviews/2026-09-06_motum_hgg_source_audit/RESULT.md)、[旧失败](archive/research_records/2026-09-06_motum_hgg29_frozen_panel/RESULT.md)、[共享桥](archive/research_records/2026-09-06_motum_hgg29_shared_orientation_bridge/RESULT.md)与[最终输入失败](archive/research_records/2026-09-06_motum_hgg29_shared_orientation_panel/RESULT.md)。
- **六cohort/656 scans证据矩阵已冻结：** Internal/UCSD/Africa/CPTAC/WAW-pre/WAW-post保持独立身份，不pool。GCSAM−Full只有Africa CI全正，其余五处跨0；Flip8五处点估计正，internal/CPTAC/WAW-pre CI全正、Africa/WAW-post跨0，Point差均不负但部署8×。没有Dice/Point/成本/全域统一冠军。见[证据矩阵](archive/reviews/2026-09-06_five_cohort_evidence_matrix/RESULT.md)与[V2机器结果](archive/reviews/2026-09-06_five_cohort_evidence_matrix/EVIDENCE_MATRIX_V2.json)。
- **CFB-GBM v3不能作为WT外部：** 2026 v3虽有法国264名GBM、347个binary GTV及五放疗医师验证的模型补全，但GTV只对应强化+坏死核心，不含non-enhancing FLAIR abnormality；专业放疗靶区不等于本项目WT。208GB Aspera主体、label/model均未访问；未来只有独立GTV任务才可用。见[来源审计](archive/reviews/2026-09-06_cfb_gbm_source_feasibility/RESULT.md)。
- **论文/汇报claim边界已机器化登记：** 固定Flip8、GCSAM/Africa、WAW、UCSD、N4、“best”、外部身份、MOTUM、T1ce-only可观测性与早期术后RC Point共11类列可说/不可说。没有全域Dice/Point/成本统一冠军；CI跨0不能写显著，public不能写hidden blind，真值label/RC不能变router/后处理。见[可读登记](archive/reviews/2026-09-06_claim_evidence_registry/RESULT.md)与[机器版](archive/reviews/2026-09-06_claim_evidence_registry/CLAIMS.json)。
- **未来外部样本量不能只看现有N：** 加入WAW-post后，四个single-scan external的paired-delta SD仍差异大。正态敏感性下，GCSAM关心`+0.01`约需313–1149独立人、`+0.02`约79–288；Flip8 `+0.01`约79–353、`+0.005`约314–1411。EGD374可能覆盖部分场景但非保证；必须新数据前冻最小有意义delta。见[规划审计](archive/reviews/2026-09-06_future_external_sample_size_sensitivity/RESULT.md)。
- **下一外部数据最低验收规范已固化：** 覆盖official/mirror parity、患者桥、label字典、人工/自动等级、native+canonical geometry、GT是否落在image-only support、N4线程复现、一次性preaccess、样本量与1×/8×成本。EGD/UTSW/Africa2024/UPENN/MOTUM分别给出解除后第一步，避免再踩来源、语义和GT修框问题。见[验收规范](archive/reviews/2026-09-05_30h_chain_audit/NEXT_EXTERNAL_DATA_SPEC.md)。
- 可自主完成仓库范围内的只读审计、联网一手文献检索、七问、预注册、隔离实现、测试、双卡训练、合法fresh-inner/新外部评测、归档和canonical文档更新；不需要为常规安全科研步骤逐项等待确认。
- “权限全开”不覆盖安全与证据规范：不得用已揭示test270、UCSD75、fixed validation、outer、confirmation或C/D inner选择loss、weight、epoch、view、阈值、checkpoint或后处理；不得把reused/post-test结果冒充blind/external；不得静默删除原始数据、代码、checkpoint或报告。
- 磁盘不足时，用户允许自主删除**已审计、可重建、已有哈希/归档证据的派生缓存**；删除前仍须精确核对目标，保留原始数据、checkpoint、代码和报告，并在交接记录删除内容与可恢复性。当前已知可回收候选是`workspace/BraTS/cache/t1ce_wt_n4_braincrop_resolution_20260818/formal_128`，审计证明756例完全包含于现972例cache，可释放约7.93GB；只有实际需要时再删。
- 执行循环：完整链路审视→用当前数字向初学者解释七问→冻结IDEA/PREREG→新entry point与合成测试→比例GPU smoke→formal→预访问receipt→合法评测→正负结果归档/更新handoff+catalog；若失败，关闭该实现并立即转向下一条未被否定的单因素路线，直到30小时结束或出现真正外部阻碍。

## 交接给下一session：当前停点

- **Otsu正式训练与联合readout完成，Stage A失败关闭：** E/F学生均168轮、工程门全过，checkpoint SHA=`6601d190…/51417050…`，新增6.7832 GPU-hours。E identity/Flip8 Scan delta=`+0.004115/+0.005374`但CI门失败；F identity=`−0.009235`且Scan/Subject CI整体负，F Flip8=`−0.002901`。数值机制0.057679→0.000021仍成立，但没有稳定准确率收益；不进Full972/外部/旧cohort。证据[结果](archive/research_records/2026-09-05_n4_otsu_clean_replication/RESULT.md)。

- **联合开卷顺序/隔离已兑现：** no-N4与Otsu在任何E/F指标前共享冻结canonical、双preaccess与三cache六大文件重新SHA；canonical只评一次，Otsu复用保存行。E/F是同一开发家族而非两份独立验证。结果与区间见[对比图](archive/reviews/2026-09-05_30h_chain_audit/N4_FAMILY_STAGE_A_COMPARISON.png)。

- **数值原因已经收窄：** 原图重复与纯0.5/2缩放排除随机性/严重纯比例问题；float32负背景往返使第三例近零信号相对误差22.1%，N4末级11→10次迭代并放大到0.057679；float64往返与直接缩放exact。当前ITK5.4.6已有旧最高档边界修复，不能归因为旧版本bug。只读train图诊断，不改正在训练方法、不访问E/F或旧cohort模型。证据[诊断结果](archive/reviews/2026-09-05_n4_scale_diagnostics/RESULT.md)。

- **UPENN术后无标签；术前逐文件镜像也不完整：** 官方671 scans/630 patients中60 follow-up全部无corrected seg；术前611中232份有corrected seg。2026复查`data-nih/tcia` release API实际4424 assets/62.87GB：630份T1GD但仅147份seg、少于官方232，网页还称4426；全asset SHA只证明Pittsburgh Fiber Data Hub镜像内部完整，UCSD已证明该provider可量化官方float32。论文另有173例进入旧BraTS，且无2024患者桥。0asset/NIfTI/seg/model；只有官方manifest/parity+患者排重才重开。证据[旧随访审计](archive/reviews/2026-09-05_upenn_source_audit/RESULT.md)与[镜像复审](archive/reviews/2026-09-06_upenn_release_mirror_followup/RESULT.md)。

- **GLIS-RT来源已审计并排除当前WT榜：** TCIA官方有230例MGH术后增强3D-T1及放疗医生手工GTV/CTV，公开646,164-byte manifest为1,915 DICOM series/230 participants；但主体28.26GB明确是NIH Controlled，当前无受控资格，且GTV/CTV分别是可见放疗靶区/含规划边界的临床靶区，不等同BraTS Whole Tumor。没有申请访问、下载DICOM/RTSTRUCT或跑模型；不把不同任务的标签重命名进现榜。证据[来源结果](archive/reviews/2026-09-05_glis_rt_source_audit/RESULT.md)。

- **QIN-GBM Treatment Response来源无体素真值并停止：** 官方54人术后治疗前double-baseline多模态MR，适合ADC/Ktrans/rCBV重测；但Data Types/下载仅MR Images，无SEG/RTSTRUCT/ROI，不能计算WT Dice/Point。T1 postcontrast仅23层×5mm也是明显grid caveat。只读网页，未下载33.51GB主体/manifest/DICOM或跑模型。证据[来源结果](archive/reviews/2026-09-05_qin_gbm_source_audit/RESULT.md)。

- **Brain-Tumor-Progression虽有20人×2治疗后时点和binary mask，但当前WT路线关闭：** 官方40 studies/383 series/8,798 images/3.16GB，含放化疗后90天内与进展时MRI、T1 pre/post/FLAIR/T2/ADC/灌注，域比UTSW更接近；但许可为NIH Controlled，mask只描述为“generated using T1w images”，无人工/专家/QC或FLAIR水肿语义，不能冒充BraTS WT。第三方改版仅11目录且resize256、裁22层/pad24层，不能替代官方原生证据。未申请访问、下载DICOM/mask或跑模型。证据[来源审计](archive/reviews/2026-09-05_brain_tumor_progression_source_audit/RESULT.md)。

- **ReMIND发现43例合法人工标签候选，但本轮输入/标签链关闭：** TCIA官方CC BY 4.0/BWH单站114人，完整metadata在像素前固定43例Glioblastoma/Astrocytoma/Oligodendroglioma，均有唯一preop T1 postcontrast及直接参考该T1ce的manual tumor；34例另有cerebrum。NRRD第二轮SSH失败；DICOM SEG五个下载/重复SHA全过，但fixed1e-5 source gap失败。独立量化source区间191/191通过后，又发现SEG是裁剪/不同分辨率grid且无SOP引用；物理grid路线中tumor/cerebrum frame gap相对声明厚度最大层残差`5.79e-6/2.42e-5`超严格DS区间，在SEG pixel/nearest映射前停止。0GPU/0模型、其余42例无像素；不再加第五种容差。证据[来源审计](archive/reviews/2026-09-05_remind_source_audit/RESULT.md)、[NRRD结果](archive/research_records/2026-09-05_remind_t1ce_source_probe/RESULT.md)、[DICOM SEG结果](archive/research_records/2026-09-05_remind_dicomseg_source_probe/RESULT.md)与[物理grid结果](archive/research_records/2026-09-05_remind_physical_grid_probe/RESULT.md)。

- **UTSW362来源/三种输入桥保留，但两条模型链均已关闭：** 官方625例中362例人工细化；no-N4/canonical/Otsu固定3例输入工程资产保留。No-N4专属queue先因Stage A关闭；随后另立既有冻结baseline并完成双源724/724 exact，却发现全362 manual labels含发布者未定义3/5，0模型关闭。因此不再是“可直接另行评测”的下一步，须发布者数据字典。证据[来源](archive/reviews/2026-09-05_utsw_glioma_source_audit/RESULT.md)、[baseline标签失败](archive/research_records/2026-09-05_utsw_frozen_baseline_external/RESULT.md)与[no-N4终态](archive/research_records/2026-09-05_utsw_no_n4_paired_external/QUEUE_TERMINAL.json)。

- **UCSD75来源核验与official-float纠正均完成：** 原评测输入来自Pittsburgh Fiber Data Hub的data-nih/tcia uint8整理镜像；TCIA主包1019的150文件426.13MB经核验为150/150几何一致、75/75源标签exact、0/75源图像exact、18,156个非零体素在镜像变0。2026-09-06另行预注册同cohort来源单因素纠正；新旧最终crop/128³标签75/75 exact、影像0/75 exact。Official-float GCSAM184=`0.755890/0.755312/70`、Full972 KD168=`0.746255/0.740165/70`；相对mirror Dice只变约`+0.0002–0.0011`且四个来源CI跨0，GCSAM Point少1。旧mirror与新official均保留榜单来源版本；后者是reused/not blind，不是新确认，禁止回流调参或再次重跑。证据[来源核验](archive/reviews/2026-09-05_30h_chain_audit/UCSD_SOURCE_RESULT.md)、[纠正结果](archive/research_records/2026-09-06_ucsd_official_float_source_correction/RESULT.md)与[机器结果](archive/research_records/2026-09-06_ucsd_official_float_source_correction/RESULT.json)。

- **EGD是下一份标签最清楚的外部候选，但需用户/机构签DUA：** Erasmus官方论文给出774位术前患者，其中374位manual binary WT且值`1=肿瘤/0=背景`、每人有T1GD；400位AUTO不得混入人工榜。Health-RI XNAT明确要求邮件联系service desk并签Data Usage Agreement后创建账号，匿名subjects端点403。当前0患者清单/影像/标签/模型；患者与BraTS重叠也未证明。若用户未来完成授权，第一步只能做manual/quality/overlap metadata审计，不直接评测。证据[来源审计](archive/reviews/2026-09-06_egd_source_feasibility/RESULT.md)。

- **UCSF-PDGM当前不能补作独立外部：** 501条/495位患者的标签虽经放射科修订和双专家批准，但官方v5已有298条BraTS21映射；按6组随访合并后296位患者已暴露。剩余199位没有BraTS21映射，而UCSF同时是本地BraTS2024来源站，缺少PDGM↔2024患者桥，不能证明独立。停在约60KB metadata，0主体/模型；未来仅在官方交叉表出现后重审。证据[来源审计](archive/reviews/2026-09-06_ucsf_pdgm_external_feasibility/RESULT.md)。

- **BraTS-Africa95是本轮新增的可运行外部证据，已完成并关闭：** 发布者`95_Glioma`、四个尼日利亚中心、95 scans/subjects；原挑战60/15/20标签现全公开，本项目合并一次性报告且从未用于训练/选择。GCSAM184 identity=`0.591619/0.591619/84`，Full KD168 identity=`0.523003/0.523003/87`，Full Flip8=`0.526438/0.526438/88`。GCSAM−Full Dice CI整体正，但Point差−3；Flip8 Dice CI跨0。Public post-challenge/institution-disjoint/no patient bridge/not hidden blind；独立Africa榜已更新，严禁用结果调阈值或再测。证据[结果](archive/research_records/2026-09-06_brats_africa95_frozen_baselines/RESULT.md)。

- **不额外做N4：E强正、F不稳，Stage A按门失败。** E identity/Flip8 Scan delta=`+0.018044/+0.017634`且门全过；F identity/Flip8仅`+0.000597/+0.000187`，identity Point`−2`、Scan/Subject CI下界最低`−0.009649`。跨折均值虽约`+0.009`，预注册要求每折/每视图都过，不进Full972。共享教师+control14.426 GPU-hours，候选6.784 GPU-hours；无旧cohort访问。证据[RESULT](archive/research_records/2026-09-05_no_n4_clean_teacher_replication/RESULT.md)与[raw](archive/research_records/2026-09-05_no_n4_clean_teacher_replication/RAW_INNER.json)。

- **条件Full972未获准且0CUDA退出：** admission从两个RAW重新计算均false；当时剩余14.46h且8h时间门true，所以停止原因纯粹是性能门，不是时间。eligible空、`automatic_training_launched=false`，没有新Full972 checkpoint或旧cohort访问。证据[admission](archive/research_records/2026-09-05_no_n4_clean_teacher_replication/FULL972_ADMISSION_CHECK.json)。

- **RHUH no-N4输入80/80工程门保留，但条件模型链关闭：** no-N4 Stage A失败后queue exit0；EXTERNAL_PREACCESS、2.36MB标签、label voxel、GPU smoke、cache、model forward全部未发生。不得用旧checkpoint绕过资格。证据[输入结果](archive/research_records/2026-09-05_rhuh_no_n4_input_bridge/RESULT.md)与[终态](archive/research_records/2026-09-05_rhuh_no_n4_input_bridge/EXTERNAL_QUEUE_TERMINAL.json)。

- **RHUH Otsu-N4发布态输入桥接exact重建失败并关闭：** 在Otsu E/F结果前独立冻结同一80例输入QC。80/80首轮background/Otsu/support/finite/z-score门均过，但固定首3例的brain mask与Otsu fit-mask SHA为3/3 exact、最终float image SHA却0/3 exact，故整体失败。报告只留哈希，不能量化差值或推断Dice；不重跑/改线程/放宽门。CPU634.269秒、0GPU/0forward/0标签；只关闭RHUH当前桥接，不改变已冻结BraTS Otsu cache或正在等待的Stage A。证据[结果](archive/research_records/2026-09-05_rhuh_otsu_input_bridge/RESULT.md)。

- **UTSW Otsu固定3例input-only桥通过，但不进入全362模型：** 跨进程image/label/geometry/mask3/3 exact的工程结论保留；Otsu Stage A已失败，所以其余359例、cache和模型均不访问。工程确定性不等于准确率。证据[输入结果](archive/research_records/2026-09-05_utsw_primary_fets_otsu_n4_input_bridge/RESULT.md)与[Stage A](archive/research_records/2026-09-05_n4_otsu_clean_replication/RESULT.md)。

- **背景零点解码工程等价性失败并关闭：** RHUH官方720文件与.sums已获取，80份T1ce450.98MB官方MD5全匹配；但源图80/80含负值使原始正值N4格式门失败。单独预注册的零点解码支持/裁剪框exact，3训练图×2编码中源z-score误差<1e-6，但完整N4链仅4/6过门，最坏max/mean=0.057679/0.002245超0.001/0.00001门。121.5秒CPU、0GPU/0模型，未执行RHUH解码QC、未下载标签/跑外部评测；不放宽门。证据[RESULT](archive/research_records/2026-09-05_background_zero_decode/RESULT.md)。另发现data-nih/tcia自述为Pittsburgh Fiber Data Hub整理镜像，旧“TCIA官方GitHub”措辞已通过顶部全75官方来源核验更正。

- **2026-09-05审计新发现：** 沙箱外2×A100可用；RHUH公开OAuth及package684详情已成功（scope=`external_download_package`），进入官方inventory/校验和恢复阶段。另已确认Flip8 A/B和一致性C/D的fixed教师训练集与inner分别重叠87/72/88/72 scans、39/33/39/37 subjects。“fresh inner”不是教师—学生全链路未见患者；数值保留、间接泄漏风险未量化，新内部研究须解决教师训练/选择角色隔离。证据见[本轮审计](archive/reviews/2026-09-05_30h_chain_audit/SESSION.md)、[教师清单审计](archive/reviews/2026-09-05_30h_chain_audit/TEACHER_INNER_EXPOSURE.json)。

- **Direct locator固定Flip8 TTA已完成并刷新reused test270全面点估计：** 唯一变化是同一Full972 KD168 checkpoint从identity 1×改为全部8种三轴reflection、逆翻后float32等权raw-logit平均；阈值0、Point=argmax不变。A/B fresh inner的Scan/Subject平均提升=`+0.006828/+0.007132`，四个CI下界均为正，全部门通过。唯一一次test270先验证identity完整summary/per-case exact，再得到Flip8 Scan/Subject/Point=`0.757269662/0.741253585/265/270`，相对identity=`+0.005510124/+0.003926478/+3`，相对此前GCSAM点估计=`+0.005266486/+0.003987165/+13`。Scan paired CI `[+0.002751,+0.008022]`为正；Subject CI `[-0.000379,+0.007354]`跨0。当前只可称270 scans/119 subjects的post-test/reused/non-blind/non-external三项点估计第1；不称相对GCSAM显著，不得调参或重测。raw绝对CI helper seed偏差已保留并从保存per-case行离线修正，不影响点估计/配对门。证据见[结果](archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/RESULT.md)、[机器结果](archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/RESULT.json)、[raw test](archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/RAW_TEST270.json)、[Stage A](archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/STAGE_A_INNER.json)与[pretest](archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/TEST270_PREACCESS.json)。
- **镜像前后答案一致训练fresh C/D失败并关闭：** C/D各864 fit/108 inner、各49 subjects，彼此case/subject互斥且排除旧A/B的93个inner subjects。四条fixed168训练全部完成并冻结后才访问inner；两pair初值、sampling、168轮sampler、增强RNG、reflection、首batch/首loss全部exact。Candidate确实把末5轮一致性loss降到Control的`0.636/0.648`，但C Flip8 Scan/Subject/Point=`-0.001773/+0.000643/+2`，D=`-0.002063/-0.003967/-1`；两复制平均Flip8 Scan/Subject=`-0.001918/-0.001662`，identity Scan/Subject=`-0.000115/+0.000735`。机制成立但准确率直觉不成立，`all_gates_passed=false`；不调weight/formula/view/epoch/seed，不访问fixed validation/outer/confirmation/test270/UCSD75。四训练20.873 GPU-hours/双卡10.484h、峰值42.006GB、inner3,456 forwards、workspace110MB。证据见[结果](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/RESULT.md)、[机器结果](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/RESULT.json)、[raw inner](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/RAW_STAGE_A_INNER.json)、[preaccess](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/INNER_PREACCESS.json)与[修订](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/AMENDMENT_01.md)。
- **T1ce固定1–99百分位裁剪Stage A失败并关闭：** 唯一变化是N4与source非零z-score之间按每例脑内固定1st/99th percentile裁剪。A Candidate/Control Scan=`0.721939/0.730148`、Subject=`0.704389/0.714339`、Point=`102/106`，差`-0.008210/-0.009950/-4`；B为`0.731047/0.738568`、`0.724639/0.734514`、`103/98`，差`-0.007520/-0.009875/+5`。四个Dice paired CI均整体低于0，两复制平均Scan/Subject差=`-0.007865/-0.009912`。Cache标签与Control整文件exact、5/5独立重建、双卡smoke、四条168轮训练和A/B全部配对工程门均通过，故是稳定泛化损失而非接线失败。`stage_b_authorized=false`，未训练Full972，未访问fixed validation/outer/confirmation/test270/UCSD75，Leaderboard不变。四条formal共13.55 GPU-hours、双卡墙钟约6.78h；推理仍1×。证据见[结果](archive/research_records/2026-09-03_t1ce_percentile_clip_stability/RESULT.md)、[机器结果](archive/research_records/2026-09-03_t1ce_percentile_clip_stability/RESULT.json)与[raw inner](archive/research_records/2026-09-03_t1ce_percentile_clip_stability/RAW_STAGE_A_INNER.json)。
- **GCSAM fixed184双复制Stage A失败并关闭：** 两个新subject-disjoint复制均为864 fit/108 train-only inner，固定`rho=0.05`与epoch184；四条训练完成且冻结检查点后才各读取一次inner。复制A GCSAM/AdamW Scan=`0.737953/0.739167`、Subject=`0.726095/0.722541`、Point=`104/103`，差`-0.001214/+0.003554/+1`；复制B为`0.767425/0.767249`、`0.746052/0.748282`、`102/104`，差`+0.000176/-0.002230/-2`。A/B四个Dice CI下界均低于`-0.003`，两复制平均Scan差`-0.000519`；初值、sampling、184轮sampler、首增强batch和first loss全exact，故是跨seed泛化失败而非接线失败。`stage_b_authorized=false`，没有运行Full972或访问fixed validation108/outer/confirmation/test270/UCSD75。四条formal共19.14 GPU-hours、阶段墙钟约9.59h，推理仍1×。证据见[结果](archive/research_records/2026-09-03_gcsam_fixed184_replication/RESULT.md)、[机器结果](archive/research_records/2026-09-03_gcsam_fixed184_replication/RESULT.json)与[raw inner](archive/research_records/2026-09-03_gcsam_fixed184_replication/RAW_INNER.json)。
- **UCSD患者隔离外部确认已完成并关闭：** data-nih/tcia整理镜像的单文件release（非TCIA官方GitHub，实际输入75/75为uint8）解除整包33.1GB阻塞。主release精确136 subjects/184 T1-post+BraTS-seg pairs；官方映射揭示109 scans/83 subjects已进入本项目1,350例union，全部按患者排除后冻结75 scans/53 subjects，case/subject overlap=0，另列42-subject/59-study BraTS challenge test包未使用。一次性External结果GCSAM/KD Scan=`0.755647/0.745554`、Subject=`0.755152/0.739055`、Point=`71/75`/`70/75`；GCSAM−KD=`+0.010093/+0.016097/+1`，但paired Dice CI仍跨0。GCSAM只成为UCSD external75三项点估计第1，不与internal test270混排、不宣称显著、不允许回流调参。结果后development-only overlap audit在下载前排除28个test270扫描：81例中76例label bit-exact，5例仅差2–36体素且WT Dice最低`0.999742`，确认映射/标签语义正确并揭示官方release小修订；官方UCSD 256³ FOV与训练BraTS副本182×218×182使brain-crop/128³ tensor存在真实grid/FOV域差异。训练0；下载103.18MB、cache751MB、150 forwards。External已访问一次；本实验未新增fixed validation/outer/internal confirmation/reused test270访问。证据`archive/research_records/2026-09-02_ucsd_ptgbm_external_confirmation/RESULT.md`。
- **外部cohort来源审计：** UCSD论文/TCIA说明BraTS四区seg经人工修正与神经放射科复核；早期审计曾因Aspera整包和磁盘判断暂不可行，后由data-nih/tcia整理镜像逐文件assets（其官方身份旧表述已撤回）解决。MU-Glioma-Post约11GB但标签为自动nnU-Net流程，证据较弱且未访问。来源审计证据`archive/reviews/2026-09-02_external_cohort_audit/RESULT.md`。
- **MU-Glioma-Post复审仍停止：** 官方小表显示203 unique patients，但临床6个时间点列有597个非空值而官方影像口径为596；体积表按非空标签记录、ID/timepoint格式不一致且无BraTS/UCSD bridge。官方GitHub无逐文件release，11GB Aspera package1030会把当前约12GB余量压满；标签又是nnU-Net多模型经STAPLE融合，弱于UCSD人工批准真值。因此仅归档409,321-byte元数据，不下载NIfTI/seg、不跑模型、不建榜。证据`archive/reviews/2026-09-02_mu_glioma_post_followup/RESULT.md`。
- **RHUH-GBM曾是下一高质量外部候选；来源后来恢复但模型链仍关闭：** 官方40 subjects×术前/早期术后/复发=120 scans，专家人工修正三类肿瘤标签；后续已恢复package684官方源并完成80例输入工程，但发布态负值与canonical不兼容，no-N4模型Stage A失败、Otsu桥不exact。以下旧段保留当时Faspex阻塞历史，不代表当前可直接运行；第三方HF镜像仍不可冒充正式来源。最新边界见顶部RHUH条目，早期证据`archive/reviews/2026-09-02_rhuh_gbm_external_feasibility/RESULT.md`。
- **其他外部候选边界：** BraTS 2024一手论文明确UCSF约600例来自既有UCSF-LPTDG/ALPTDG，因此UCSF-ALPTDG是当前BraTS源cohort，不能换ID后冒充external；LUMIERE只有专家RANO临床评价、不是同口径四区voxel segmentation。两者均不进入T1ce/WT外部榜。
- **2026 loss-stabilization适配已完成单因素检验并关闭：** 论文SACA的4-class subspace仍不适配binary WT，Norm arm也不能整包归因；唯一可隔离的1–99 percentile normalization已用`/tmp`临时cache完成双fresh-seed检验。两复制Dice均下降且CI整体低于0，因此不再搜索百分位、顺序或合并voxel contrastive loss。论文审计见`archive/reviews/2026-09-02_loss_stabilization_paper_fit_audit/RESULT.md`，实验结果见`archive/research_records/2026-09-03_t1ce_percentile_clip_stability/RESULT.md`。
- **已证明7.93GB train cache冗余但未删除：** 历史`braincrop_resolution.../formal_128`的756例全部包含于当前972例brain-crop train cache；按case ID重排后image/label均`756/756` byte-exact、最大差0。若用户另行明确授权，只删除该精确目录可释放7,927,508,320 bytes，使未来约7.4GB新cache可保留约12GB安全余量。本轮无删除、无test/confirmation/external/model访问。证据`archive/reviews/2026-09-02_disk_reclaim_audit/RESULT.md`。
- **Gamma增强已关闭：** Stage A fresh inner曾`+0.006151/+0.005570/+1 Point`并过六门，但fresh Full972 final Gamma/Control=`0.729001/0.720571/101` vs `0.732850/0.722882/101`，反向`-0.003848/-0.002312/0`；相对archived seed32为`-0.011011/-0.010695/-2`且两CI整体低于0。全部Gamma比例/范围/统计/label/RNG与配对工程门通过，失败是跨seed泛化。最终11门过2失9，不访问test，不调Gamma。证据`archive/research_records/2026-09-02_full972_gamma_aug_stability/RESULT.md`。
- **Lookahead5已关闭：** 完全配对fit864后，fresh train-only inner Candidate/AdamW=`0.738678/0.716695/104` vs `0.742575/0.721451/104`，即`-0.003897/-0.004757/0`；两项CI下界`-0.009837/-0.011494`，六门过2失4。初值、sampling weights、168轮sampler、首增强batch exact，Candidate 20328 fast/4065 slow/3 trailing/final-slow与全部工程门通过，失败是泛化而非接线。未运行Full972 Stage B，未访问fixed validation/test；禁止调`k/alpha`。证据`archive/research_records/2026-09-02_full972_lookahead5_stability/RESULT.md`。
- **LAWA6已关闭：** Stage A两个fresh seed在train-only inner均三指标同向并过10门；第三个fresh Full972 final也相对自身raw提高`+0.000464/+0.000576/0 Point`，但fixed validation仅`0.732919/0.726370/102/108`，相对archived Full972 seed32为`-0.007093/-0.004897/-1 Point`，两项archived CI安全门失败。说明同轨迹最近六轮平均是可复现的小幅平滑机制，但远不足以克服seed间差异。Final 11门过6失5，不访问test，不调窗口/权重。证据`archive/research_records/2026-09-02_full972_lawa6_stability/RESULT.md`。
- 2026-09-04历史停点没有运行中的训练；2026-09-05已启动顶部所述no-N4新runner。没有已授权待执行的旧cohort模型评测，不得重启任何旧runner。
- 本session先完成整条科研链路复盘并向用户报告5条数字化启发，然后严格按“七问→预注册→合成合同→GPU smoke→formal→门槛”执行三条稳定性/多老师路线。没有创建大cache，没有静默删材料。
- **GCSAM已关闭：** fresh train-only inner点估计略升但两项CI门失败，未跑fresh-seed final。用户随后授权failed-gate exploratory reused test270；fit864 epoch184=`0.752003/0.737266/252/270`，相对Full972 KD168=`+0.000244/-0.000061/-10 Point`且两个Dice CI跨0。它只刷新Scan点估计，不是全面/显著提升、Full972 final、盲测或外部确认，禁止回流调参。
- **Fresh三seed固定共识已关闭：** 共识=`0.746942/0.735130/106`，明显救弱seed32/33，但相对强seed31仅`+0.000530/+0.000063/0`，CI下界`-0.009103/-0.016105`失败；没有进入student/Full972 final，也未访问fixed validation/test。
- **GT最佳老师routing已关闭：** Full972 Candidate/Control/archived seed32 fixed validation=`0.732910/0.725998/100`、`0.736013/0.728973/103`、`0.740013/0.731266/103`。Candidate相对Control `-0.003104/-0.002975/-3 Point`，相对archived `-0.007103/-0.005268/-3 Point`，九项门全失败。Routing确实覆盖seed32/40/41=`47215/63869/51540`，失败是泛化失败而非接线失败；未访问test。
- 两项multi-seed新实验共同收窄了结论：seed多样性可以救弱轨迹，但固定平均不能稳定超过强seed；training-label hard routing会让student学习不一致target并降低validation。**下一session不要继续围绕seed32/40/41设计soft weight、entropy gate、inverse-Dice、per-voxel router或剔除seed。**
- 当前本地`/mnt/hdfs/sjy/datasets`只有同一`brats2024_gli_t1ce_seg_v1`数据源，没有发现额外未揭示外部cohort。真正提高证据等级优先需要新外部/独立cohort；若仍做内部方法研究，必须新建fresh证据且离开“多seed怎么加权”轴。

关键证据：

- GCSAM：[结果](archive/research_records/2026-08-31_gcsam_full972_stability/RESULT.md)、[机器结果](archive/research_records/2026-08-31_gcsam_full972_stability/RESULT.json)、[raw test](workspace/BraTS/sammed3d_t1ce_wt/gcsam_full972_stability_20260831/GCSAM_INNER_E184_TEST270.json)。
- Fresh三seed共识：[结果](archive/research_records/2026-09-01_fresh_multiseed_consensus_distillation/RESULT.md)、[机器结果](archive/research_records/2026-09-01_fresh_multiseed_consensus_distillation/RESULT.json)、[raw inner](workspace/BraTS/sammed3d_t1ce_wt/fresh_multiseed_consensus_20260901/TEACHER_CONSENSUS_INNER.json)。
- GT最佳老师routing：[结果](archive/research_records/2026-09-01_gt_best_teacher_routing_distillation/RESULT.md)、[机器结果](archive/research_records/2026-09-01_gt_best_teacher_routing_distillation/RESULT.json)、[raw validation](workspace/BraTS/sammed3d_t1ce_wt/gt_best_teacher_routing_20260901/GT_BEST_ROUTING_VALIDATION.json)。
- 当前内部test总体三项最高点估计是Full972 KD168固定Flip8（8×）：`0.757270/0.741254/265/270`，checkpoint SHA-256 `f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b`。1×范围最高Scan仍是GCSAM fit864 epoch184；1×最高Subject/Point仍是Full972 KD168 identity。

## 下一session第一步（严格顺序）

1. 完整阅读[`AGENTS.md`](AGENTS.md)、本文件、[`AFLOW_BRATS_LEADERBOARD.md`](AFLOW_BRATS_LEADERBOARD.md)、[`AFLOW_BRATS_METHOD_CATALOG.md`](AFLOW_BRATS_METHOD_CATALOG.md)及准备延伸方向的自包含`RESULT.md`。
2. 先读UCSD mirror/official、BraTS-Africa95、CPTAC33、WAW preop152与WAW postop31结果；均已访问并关闭，不得用于checkpoint、threshold、预处理、病例/时间点或新方法选择，也不得重跑。Africa的GCSAM强差在其余队列均未确定复现，不能回流复活训练配方。
3. 当前没有第二个可立即运行且未揭示的人工标签新外部队列：EGD需DUA；UCSF-PDGM缺患者桥；RHUH模型链失败；UTSW标签3/5未解释；MOTUM-HGG29有一例WT超出图像支持bbox；ReMIND/Brain-Tumor-Progression等也因几何、权限或标签语义关闭。不得代签、猜标签、用GT扩框/事后padding或第三方hash不明镜像绕过。
4. 用户已于2026-09-07授权复用开发角色：排除旧confirmation后864 scans/391人，固定fit684/313与dev180/78，明确reused development。辅助监督与患者采样两个workflow均已完成，先读两份RESULT及本轮综合报告，不恢复旧监督器或重跑dev。患者采样有Dice正证据但Point门失败，不能据此挑seed、调mix或放宽门；旧fresh仅2人/3 scans的事实不阻塞已授权复用开发，也不使复用数据成为fresh。其他fixed validation/outer/confirmation/test/外部继续保护。
5. 新idea必须离开已关闭的容量、全图160³、旧loss/feature-KD、难例聚合、固定/自适应多seed融合、LAWA6/Lookahead/Gamma、镜像一致性、percentile clip、当前no-N4/Otsu200轴；仍需七问与预注册后才实施。
6. 原30小时轮次已完成。后续新研究先审视证据缺口和可行性，再按用户的新目标/时长与现有授权写前瞻协议；不得恢复本轮已关闭的精确配方、放宽门或重复已揭示cohort。

## 当前任务

- 数据：BraTS 2024 post-treatment GLI，单通道T1ce。
- 目标：二分类WT=`{1,2,3}`，排除原标签4。
- 主系统：image-only brain crop、true-128³、4.70M参数plain SegResNet locator。
- 当前最高总体推理：固定Flip8 TTA，8种三轴reflection逆翻后float32等权raw-logit均值，阈值`>0`、最高均值logit为自动point；无SAM或其他后处理。1× identity行继续单列比较。
- internal test270已经多次揭示，只能作为复用内部报告集，不能用于调参或宣称盲测/SOTA。
- 2026-08-25 Full972 random-init KD168：test Scan/Subject/Point=`0.751760/0.737327/262/270`，相对旧ensemble和KD200的Dice paired CI均整体为正。它现在是1×范围最高Subject/Point点估计、1× Scan第2（总体含8×时rank3）；仍是failed-gate exploratory reused test，不是pristine blind或外部确认。证据`archive/research_records/2026-08-24_voxel_kd_fulltrajectory_972/RESULT.md`。

- 2026-08-31用户授权GCSAM failed-gate exploratory reused test270：fit864 inner-selected epoch184=`0.752003/0.737266/252/270`，相对Full972 KD168=`+0.000244/-0.000061/-10 points`，两项paired CI均跨0。它只在1×范围Scan点估计上排第1；1× Subject/Point仍由Full972 KD168领先，总体三项已由固定Flip8刷新。非Full972 final、非盲测/外部确认，禁止据此调rho、epoch、seed、阈值或后处理。证据`archive/research_records/2026-08-31_gcsam_full972_stability/RESULT.md`。

- 2026-08-25 post-test multiseed结果：seed40/41完整168轮；seed32/40/41固定各1/3 ensemble validation=`0.745841/0.735684/104/108`，相对seed32=`+0.005829/+0.004418/+1`，但Subject CI下界`-0.005660`失败`-0.003`门。按预注册不调权重、不剔除seed、不蒸馏、不访问新test。证据`archive/research_records/2026-08-25_full972_multiseed_ensemble_distill/RESULT.md`。

- 2026-08-28 Full972 width24 final结果：两臂已用新seed重训到inner冻结的epoch166。Fixed validation108上matched width16=`0.739335/0.731320/105/108`，width24=`0.736464/0.727795/104/108`；width24相对matched为Scan/Subject `-0.002872/-0.003525`、Point-1，四个点估计门与四个CI门均失败。说明inner收益未跨seed复现，关闭width24，不访问outer/confirmation/test。证据`archive/research_records/2026-08-26_full972_width24_fulltrajectory/RESULT.md`。

## 当前Leaderboard摘要

| 范围 | 方法 | Scan Dice | Subject Dice | Point hit | 成本 |
|---|---|---:|---:|---:|---:|
| UCSD75 official-float Scan/Subject点估计 | GCSAM fit864 inner-selected epoch184 | `0.755890` | `0.755312` | `0.933333` | `1×` locator |
| UCSD75 mirror-uint8 Point点估计 | GCSAM fit864 inner-selected epoch184 | `0.755647` | `0.755152` | `0.946667` | `1×` locator |
| Africa95总体/1× Dice点估计 | GCSAM fit864 inner-selected epoch184 | `0.591619` | `0.591619` | `0.884211` | `1×` locator |
| Africa95 Point点估计 | Full972 KD168固定Flip8 | `0.526438` | `0.526438` | `0.926316` | `8×` locator |
| CPTAC33总体点估计 | Full972 KD168固定Flip8 | `0.844577` | `0.844577` | `1.000000` | `8×` locator |
| CPTAC33最高1×点估计 | GCSAM184 identity | `0.843285` | `0.843285` | `1.000000` | `1×` locator |
| WAW152总体/最高1× Dice点估计 | GCSAM184 identity | `0.693601` | `0.693601` | `0.921053` | `1×` locator |
| WAW152 Point点估计 | Full972 KD168固定Flip8 | `0.693494` | `0.693494` | `0.940789` | `8×` locator |
| UCSD-BraTS2024test56总体点估计 | Full972 KD168固定Flip8 | `0.732973` | `0.733625` | `1.000000` | `8×` locator |
| UCSD-BraTS2024test56最高1× | Full972 KD168 identity | `0.730002` | `0.730740` | `1.000000` | `1×` locator |
| WAW postop31总体Dice点估计 | Full972 KD168固定Flip8 | `0.598959` | `0.598959` | `0.483871` | `8×` locator |
| WAW postop31最高1× Dice | GCSAM184 identity | `0.593305` | `0.593305` | `0.451613` | `1×` locator |
| WAW postop31最高1× Point | Full972 KD168 identity | `0.591647` | `0.591647` | `0.483871` | `1×` locator |
| 内部test总体三项最高点估计（post-test） | Full972 KD168固定Flip8 TTA | `0.757270` | `0.741254` | `0.981481` | `8×` locator |
| 内部test最高1× Scan点估计 | GCSAM fit864 inner-selected epoch184 | `0.752003` | `0.737266` | `0.933333` | `1×` locator |
| 内部test最高1× Subject/Point | Full972 random-init voxel-KD epoch168 identity | `0.751760` | `0.737327` | `0.970370` | `1×` locator |
| 历史单模基座 | true-128³ plain epoch168 | `0.741204` | `0.723179` | `0.929630` | `1×` locator |

完整、严格同口径排名见[`AFLOW_BRATS_LEADERBOARD.md`](AFLOW_BRATS_LEADERBOARD.md)。

UCSD75、Africa95、CPTAC33、WAW preop152、WAW postop31与reused internal test270是不同cohort，表内不可横向按数值排名。UCSD official-float是同75例来源纠正；Africa为public post-challenge四中心；CPTAC为BraTS2021 public Training；WAW preop/postop是同Warsaw source但患者零重叠，postop仅day0–22。所有外部均非服务器hidden blind；机构/来源差异不等于有跨库patient-ID桥。

## 已确认的成功主线

1. Image-only brain crop：去除黑背景后再缩放，旧匹配协议Scan约`+0.054`。
2. Direct plain locator：直接使用locator mask，避免强locator接旧SAM后退化。
3. Late-cosine长时程：epoch99→168将test Scan从`0.725820`提高到`0.741204`。
4. 固定logit ensemble：证明历史模型与EMA的输出错误互补；旧总体最高，后续已被Full972单模超过。
5. Voxel-logit KD：把ensemble知识压回单模型；Full972完整轨迹identity仍是当前1×最高Subject/Point；同checkpoint固定Flip8是8×总体点估计最高。

通俗解释、失败原因和待尝试方法见[`AFLOW_BRATS_METHOD_CATALOG.md`](AFLOW_BRATS_METHOD_CATALOG.md)。

## 已关闭/判负方向

- 全图true-160³：计算/显存近2倍，test反而下降；大cache已在保留报告后删除。
- Full972 width24完整轨迹：参数4.70M→10.57M，inner正向但新seed final validation反向；matched width16 `0.739335/0.731320/105`，width24 `0.736464/0.727795/104`。容量扩展不成立，不进test。
- CAPG中心头、灰度/intensity residual、旧SAM再解码、prompt-local refiner。
- Cross-seed parameter soup：两个seed不在同一参数盆地，validation灾难性失败。
- EMA单模型：能改善同seed raw，但没有超过历史最佳Dice。
- ReCo-KD轻量区域蒸馏：同epoch小收益，不超过最佳base voxel-KD，不进test。
- NeurIPS 2025 LoMix-3D三尺度适配：工程门全部通过，但fixed validation selected epoch200仅Scan/Subject/Point `0.732479/0.721607/0.935185`；相对同epoch control为`-0.001126/-0.002065`，相对KD200为`-0.001905/-0.002827`，promotion失败。validation已访问；outer/confirmation/test均未访问。只否定当前raw-logit、未归一化Softplus、32轮continuation实现。完整证据见[结果归档](archive/research_records/2026-08-19_lomix3d/RESULT.md)和[raw report](workspace/BraTS/sammed3d_t1ce_wt/plain_lomix3d_20260819/LOMIX3D_VALIDATION.json)。
- MICCAI 2026 DCD方向细节蒸馏适配：raw detail loss下降约86%，但fixed validation selected epoch200仅Scan/Subject/Point `0.733603/0.723798/0.935185`，相对base KD200为`-0.000781/-0.000636/0`；promotion失败。本方法的新validation已访问，outer/confirmation/new test均未访问。只否定当前同容量、T1ce/WT、32轮DCD-on-voxel-KD适配。完整证据见[结果归档](archive/research_records/2026-08-19_dcd_detail_distillation/RESULT.md)和[raw report](workspace/BraTS/sammed3d_t1ce_wt/plain_dcd_20260819/DCD_VALIDATION.json)。
- CVPR 2026 KAS-on-voxel-KD适配：完整双卡pairing通过，但fixed validation selected KAS232仅Scan/Subject/Point `0.732915/0.723199/0.916667`，相对same-epoch control为`-0.001219/-0.001419/0`，相对KD200为`-0.001469/-0.001235/-0.018519`；promotion失败。本方法的新validation已访问，outer/confirmation/new test均未访问。完整证据见[结果归档](archive/research_records/2026-08-20_iad_kas_assimilation/RESULT.md)和[raw report](workspace/BraTS/sammed3d_t1ce_wt/plain_iad_kas_20260820/KAS_VALIDATION.json)。
- 2025 GCSAM paper-form适配：fresh train-only inner点估计正向但两项CI门失败，未跑fresh-seed final/fixed validation。用户授权exploratory test后，fit864 epoch184取得Scan/Subject/Point `0.752003/0.737266/252/270`；相对Full972 KD只Scan `+0.000244`，Subject近乎持平、Point少10且两项CI跨0。已访问train-only inner与reused test270；outer/confirmation未访问。证据见[结果归档](archive/research_records/2026-08-31_gcsam_full972_stability/RESULT.md)。
- 2025 hard mL1-ACE前景校准适配：20-bin loss确实把selected epoch224 macro/official-micro ACE从same-epoch control的`0.230865/0.197534`降到`0.125927/0.071770`，Point从100/108升到103/108；但Scan/Subject仅`0.723870/0.713278`，相对control为`-0.010544/-0.011392`且95% CI整体低于0，相对KD200也下降，promotion失败。新fixed validation已访问；outer/confirmation/new test均未访问。完整证据见[结果归档](archive/research_records/2026-08-20_hard_ml1ace/RESULT.md)和[raw report](workspace/BraTS/sammed3d_t1ce_wt/plain_hard_ml1ace_20260820/HARD_ML1ACE_VALIDATION.json)。
- 2025/2026 VM Dice困难体素适配：gamma0/gamma1双臂完整配对，selected gamma1 epoch224 Scan/Subject/Point=`0.732178/0.721938/0.953704`，低于same-epoch gamma0 `0.732818/0.722534/0.953704`及KD200；boundary-top27小幅`+0.000903`，但KD200-bottom27为`-0.001495`，promotion失败。首次smoke只读诊断触发PyTorch quantile元素上限，在反传前停止并修为exact min/max后双臂重跑通过；方法公式未改。新fixed validation已访问；outer/confirmation/new test未访问。证据见[结果归档](archive/research_records/2026-08-20_vm_dice/RESULT.md)和[raw report](workspace/BraTS/sammed3d_t1ce_wt/plain_vm_dice_20260820/VM_DICE_VALIDATION.json)。
- 2025 source-inspired Batch-CVaR50病例尾部聚合：新uniform/CVaR双臂完整配对，selected CVaR208 Scan/Subject/Point=`0.731801/0.721922/0.925926`，相对same-epoch uniform为`-0.001393/-0.001412/-0.018519`且Dice CI整体低于0；KD200-bottom27=`-0.002275`、boundary-top27=`-0.001551`，tail机制失败。GPU parity曾暴露FP32与legacy autocast loss差`3.35e-4`，在formal前恢复历史autocast语义并实现loss/gradient精确0差，双臂全部重跑。新fixed validation已访问；outer/confirmation/new test未访问。证据见[结果归档](archive/research_records/2026-08-20_batch_cvar50/RESULT.md)和[raw report](workspace/BraTS/sammed3d_t1ce_wt/plain_batch_cvar50_20260821/BATCH_CVAR50_VALIDATION.json)。
- MICCAI 2025 single-bottleneck DMRFB dilation adapter：same-parameter双臂、epoch0 identity和two-step gradient门均通过；selected candidate216 Scan/Subject/Point=`0.734242/0.724742/0.925926`，相对同轮control `+0.000245/+0.000146/-0.009259`，相对KD200 `-0.000142/+0.000308/-0.009259`。Boundary-top27 `+0.000107`、bottom27 `+0.001184`，机制方向为正；但KD Scan和Point门失败，不进test。首次validation第一批因输出缺`center_logits=None`接口停止，无指标/选择揭示；只读property修复后V2完成。新validation已访问；outer/confirmation/new test未访问。证据见[结果归档](archive/research_records/2026-08-21_dmrfb_bottleneck_adapter/RESULT.md)和[raw V2](workspace/BraTS/sammed3d_t1ce_wt/plain_dmrfb_adapter_20260821/DMRFB_ADAPTER_VALIDATION_V2.json)。
- KD200–DMRFB216同盆地weight interpolation：固定alpha0.25/0.5/0.75，无训练；selected alpha0.5 Scan/Subject/Point=`0.734683/0.724934/0.925926`，同时超过KD200和DMRFB216的Scan/Subject，boundary/bottom27相对KD也为正，但Point仍100/108，低于101门，故不进test。首次evaluator preflight因metadata字符串不一致在started前停止，纯合同修复后完成。新validation已访问；outer/confirmation/test未访问。证据见[结果归档](archive/research_records/2026-08-21_dmrfb_weight_interpolation/RESULT.md)和[raw report](workspace/BraTS/sammed3d_t1ce_wt/dmrfb_weight_interpolation_20260821/INTERPOLATION_VALIDATION.json)。
- 固定计算量的972例voxel-KD数据扩容：用同一cache新跑756/972双臂，每轮均756 draws、94 steps。Selected candidate184 Scan/Subject/Point=`0.734049/0.724694/0.916667`（99/108）；相对same-epoch control为`+0.003297/+0.004287/-0.009259`，两项Dice 95% CI均高于0，明确支持数据多样性机制。但相对历史KD200为`-0.000335/+0.000260/-0.018519`，KD Scan和Point101门失败，不进test。新fixed validation已访问；旧selector/confirmation已明确回收为train且不再具有独立证据身份；test未访问。证据见[结果归档](archive/research_records/2026-08-21_voxel_kd_data972/RESULT.md)和[raw report](workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_data972_20260821/DATA972_VALIDATION.json)。
- 972例原生crop体积感知voxel-KD：train-only工件对重叠756例的体积和权重完exact，原生权重将Point恢复到101/108。Selected candidate200 Scan/Subject/Point=`0.734060/0.724703/0.935185`，相对same-epoch native756 control为`+0.000689/+0.000666/-0.009259`；相对KD200为`-0.000324/+0.000269/0`。只失败严格超过KD200 Scan一门，不进test。工件在formal前expose并修复full-FOV/crop定义、LAS/RAS坐标和Python布尔拼写问题，均在输出/模型/validation前停止且透明归档。新fixed validation已访问；outer/confirmation已回收为train；test未访问。证据见[结果归档](archive/research_records/2026-08-21_voxel_kd_data972_native_sampling/RESULT.md)和[raw report](workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_data972_native_20260821/NATIVE_DATA972_VALIDATION.json)。
- MICCAI 2025 diversity/uncertainty sampling启发的覆盖优先小病灶回放KD：972病例、相同752有效draw/94步双臂，candidate用680互补coverage+72原生体积replay。Selected epoch200 Scan/Subject/Point=`0.734158/0.724950/98/108`，相对same-epoch replacement control=`+0.001481/+0.001890/-1 point`，支持减少replacement重复的局部机制；但相对KD200=`-0.000226/+0.000516/-3 points`，KD Scan CI下界`-0.003098`也失败，故不进test。18项sampler测试、5项validation合同测试、V4 smoke、双卡formal和32轮覆盖审计全过；全部实现/证据接线失败都在相应访问前归档。新fixed validation已访问；outer/confirmation未访问且已回收为train；test270未访问。证据见[结果归档](archive/research_records/2026-08-24_coverage_small_lesion_replay_kd/RESULT.md)和[raw report](workspace/BraTS/sammed3d_t1ce_wt/coverage_small_lesion_replay_kd_20260824/COVERAGE_REPLAY_VALIDATION.json)。
- 972病例配额一致voxel-KD：两臂都用972病例与原生体积权重，只把每轮有效draw/step从752/94增加到968/121。Selected candidate200=`0.734259/0.724956/100/108`，相对same-epoch control=`+0.000779/+0.000954/0`，证明972池确实受训练预算限制；但相对KD200=`-0.000125/+0.000522/-1 point`，KD Scan和Point门失败，不进test。Formal control/candidate 32.1/41.3min，每卡20.29GiB。新fixed validation已访问；outer/confirmation/test270未访问。证据见[结果归档](archive/research_records/2026-08-24_voxel_kd_972_draw_budget/RESULT.md)和[raw report](workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_972_draw_budget_20260824/DRAW_BUDGET_VALIDATION.json)。
- Domain-SAGE、error gate、boundary辅助、adaptive routing的机制主张未成立。

不要在已揭示test上继续调这些方法的权重、epoch、阈值或后处理。

## 下一session必须先审视整个科研链路

这是继续科研前的硬门，不是建议。新session禁止读完当前最高分后直接找模块、改loss或开训练；必须先完成下面的链路审视，并先用最直白的中文向用户报告“看到了什么、由此得到什么启发”。完成审视前，不创建新cache、不训练、不访问新的validation、outer、confirmation或test。

### 必须审视的完整链路

1. **数据与角色：** 梳理756→972病例、train-only inner、full retrain、fixed validation108/50 subjects和reused internal test270/119 subjects分别承担什么角色；确认哪些集合已经被揭示，哪些不能再当盲证据。
2. **预处理与输入：** 对比全图、image-only brain crop、true-128³和true-160³，判断收益来自保留肿瘤细节还是去除无效背景；不能把160³负结果误读成“所有多尺度方法无效”。
3. **训练轨迹与数据利用：** 串起epoch99→168长时程、late continuation、972 draw budget、coverage/replay和Full972从随机初始化参与，区分“更多病例”“更多更新步”“更早参与表示学习”的作用。
4. **知识来源与目标：** 串起history/EMA logit ensemble、voxel-logit knowledge distillation、ReCo/DCD/KAS/LoMix等负结果，判断teacher信息、监督loss和学生容量谁是真正限制。
5. **模型与容量：** 对比plain width16、160³、width24、DMRFB和其他adapter。必须解释为何“更大模型/更高分辨率/更多模块”多次没有转化为泛化收益。
6. **采样与病例异质性：** 对比972 native-volume、coverage replay、CVaR和multiseed，检查小病灶、难病例、重复抽样和seed方差分别影响Scan Dice、Subject Dice与Point的方式。
7. **评测与证据强度：** 同时看Scan、Subject、Point、paired CI、改善/变差case数和跨seed复现；不能只看一个点估计。明确validation和test均已多次使用，后续结论必须标注开发性/探索性边界。

### 审视后必须先形成的产物

- 先画出或文字列出`raw data → manifest/split → cache/preprocess → train/teacher/loss → inner选epoch → fresh-seed full retrain → validation gate → test report`，为每一环标出当前证据、失败案例和可能瓶颈。
- 至少提出3条有数字支撑的启发；每条必须说明“哪个实验支持”“哪个负结果限制了它”“还缺什么证据”。
- 把准备采用的启发写入新实验`archive/research_records/<date>_<method>/IDEA.md`中的“链路审视与启发”部分，并先向用户用小白语言说明。不能只写论文名或缩写。
- 新方法必须来自链路中尚未解决的瓶颈；必须说明为何现有负结果没有已经否定它。若需要2025+前沿idea，先检索并引用论文/官方代码等一手来源，再决定是否缝合；不能为了“前沿”而堆模块。
- 只有完成上述复盘，才进入七问说明、预注册、合成测试、GPU smoke和formal实验。

## 当前链路已经给出的初步启发

这些是交接起点，不替代下一session自己的审视：

1. **最大的可靠收益来自数据表示和训练时机，而非盲目堆算力。** Image-only brain crop旧匹配协议约`+0.054` Scan；epoch99→168约`+0.015`；新增216例从随机初始化开始参与产生当前内部test最高点估计。相反，160³近2倍体素计算失败，width24参数2.25倍也失败。
2. **当前主要瓶颈之一是稳定性。** Width24在inner相对width16曾有Scan/Subject `+0.010634/+0.012193`，但fresh-seed final变成`-0.002872/-0.003525`。三seed ensemble点估计提高，但Subject CI门仍失败。后续不能把单个seed或单次inner胜利当成真实机制成立。
3. **Point和Dice不是同一个目标。** 多个采样、校准和adapter实验会提高Point却降低Dice，或Dice略升却丢Point。下一方案必须事先声明主要目标，并同时冻结Scan/Subject/Point门，不能在结果后挑有利指标。
4. **当前4.70M width16不像明显欠容量，更可能是监督信息、病例异质性或泛化稳定性受限。** Width24训练Dice/inner更好但final validation更差，说明继续width20/32网格搜索价值很低；新的模型改动必须针对明确错误模式，而不是单纯变大。
5. **内部test分数不能替代远域证据。** Full972 KD168 internal test约`0.752`，在Africa95却只有`0.523`；GCSAM内部几乎tie，Africa却高`0.0686`。现有internal/UCSD/Africa均已揭示且不可反向选择loss、epoch、阈值、ensemble或后处理；新的内部fresh容量又只剩2人/3 scans。下一方法必须等待新数据或用户明确role reassignment。

## 下一条前沿候选

2026-09-07更新：当前首条研究是已授权的训练期切除腔辅助监督，正在双卡配对训练。TCIA公开UCSD BraTS-GLI 2024 test份额已由Fable完成并关闭；BraTS-PEDs257输入链和DFCI无标签等外部边界继续保留。

当前可以在固定864扫描的reused-development范围内按七问/预注册推进方法开发；先完成正在运行的两seed辅助监督实验及统计归档，再决定下一个不同机制。新独立外部证据仍受登记表限制：No-N4/Otsu Stage A、RHUH条件链、UTSW标签语义、MOTUM与OpenNeuro图像支持均关闭；IMAGO为自动accept/reject标签；所有已评外部cohort继续封存。当前开发授权不包含对旧外部/test选择超参、输入variant、padding或路由。

独立外部证据优先级见[外部阻塞登记表](archive/reviews/2026-09-05_30h_chain_audit/EXTERNAL_BLOCKER_REGISTER.md)：EGD374的Health-RI DUA、CoRe-BT250的UW申请/DUA仍需用户；发布者可解除UTSW3/5数据字典、Africa2024新公开版等。外部这些限制没有因内部开发授权而消失；各未获权外部路线只读审计，不用第三方镜像、自动mask或猜标签绕门。

GCSAM184在Africa95/WAW Dice点估计最高但Point较低、训练配方双复制失败；Full972 KD168 Flip8在internal/CPTAC总体和Africa/WAW Point更高。没有单一三项/全域冠军，所有已揭示cohort都禁止继续选择超参。

每个新方法必须先按[`EXPERIMENT_IDEA_TEMPLATE.md`](EXPERIMENT_IDEA_TEMPLATE.md)用小白语言写清：问题、直觉、选择原因、做法、唯一变化、门槛、成本。2025+文献检索归档见
`archive/handoffs_20260819/AFLOW_BRATS_FRONTIER_IDEAS_2025PLUS_20260819.md`。

## 数据与资源

- Raw NIfTI：`/mnt/hdfs/sjy/datasets/brats2024_gli_t1ce_seg_v1`
- 756-case train manifest SHA-256：`a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16`
- 保留的true-128 train cache：约7.4G。
- 固定validation108 cache：约1.1G。
- 新保留的LoMix-3D代码、四个checkpoint、训练/validation报告与完整归档：workspace约144M；selected negative checkpoint为epoch200，SHA-256 `f9bfcca36a904341f4eeebf1c71f218b997ab3e753be4815daa2789b2c7245fd`。
- 新保留的DCD代码、四个checkpoint、训练/validation报告与完整归档：workspace约72M；selected negative checkpoint为epoch200，SHA-256 `9354a0f6ef9e6747729ca5be61e3faa357b76f558898940e52e8f7b274f543dd`。
- 新保留的KAS paired代码、8个checkpoint、双臂训练/validation报告与完整归档：workspace约217M；selected negative KAS232 SHA-256 `845a6448a5d472c643146ecd11b43e03f440a29eae34ec029344f47d7acda409`。
- 新保留的hard mL1-ACE代码、四个checkpoint、训练/validation报告与完整归档：workspace约76M；selected negative hACE224 SHA-256 `e773e50f6f5941cdfcf37a60e9915c76b0a86acaec29f3afa4dff9e1789b71ab`。
- 新保留的VM Dice代码、8个checkpoint、双臂训练/validation报告与完整归档：workspace约145M；selected negative gamma1 epoch224 SHA-256 `15b9c20b8239b82c32180ae4e2cd97fe36bd051711191ff86730d6907d11a017`。
- 新保留的Batch-CVaR50代码、8个checkpoint、双臂训练/validation报告与完整归档：workspace约145M；selected negative CVaR208 SHA-256 `791ca9d00ed1fc319ceaa34a3ccec7e97c691233ce71e2dc4c77c9cd94bfafb1`。
- 新保留的DMRFB代码、8个modified checkpoint、双臂训练/validation报告与完整归档：workspace约146M；selected candidate216 SHA-256 `df53d9bd61f1489746ae1240fffba1f8a3897d0de1416122a389e1d420def684`。
- 新保留的DMRFB同盆地插值3个checkpoint、generation/validation与完整归档：workspace约55M；selected alpha0.5 SHA-256 `0db67cb81ae3b8aa76449918910b110207b25233dc5b962070caab6f3939195a`。
- 新保留的972例voxel-KD代码、8个checkpoint、双臂训练/validation报告与完整归档：workspace约181M；selected candidate184 SHA-256 `0f69e049aba20e734474be0e9675b0af374c9a50ee662eb61eb7f4f3a97a34c3`。
- 新保留的native-volume 972例voxel-KD代码、433KB数据工件、8个checkpoint、双臂训练/validation报告与归档：workspace约217M；selected candidate200 SHA-256 `b57693c6af1cded8de4c4e576f462ff203c942ec7cd7a78a3711b70394470c4b`。
- 新保留的coverage-small-lesion-replay KD代码、8个formal checkpoint、smoke/训练/validation报告与完整归档：workspace约253M；selected negative candidate200 SHA-256 `4a02e9ba0beb704a3b1a6429394df39bff5a706f418c12c10436230f7f658ae4`。
- 新保留的972 draw-budget KD代码、8个formal checkpoint、训练/validation报告与完整归档：workspace约181M；selected candidate200 SHA-256 `543e8cfbd80c63fec011453bd91a26beb20fc5a0486640e7846cf877de3a5fcb`。
- 失败160³ train/test/smoke派生cache已删除，可从raw和归档脚本重建。
- 2026-09-02末磁盘检查：总125G、已用约114G、可用约12G（91%）；新增UCSD raw+cache+报告workspace约897M。已证明`workspace/BraTS/cache/t1ce_wt_n4_braincrop_resolution_20260818/formal_128`是当前972例主cache的756例byte-exact子集，整目录7,927,508,320 bytes；**尚未删除**，只有用户明确授权才可按该精确路径回收。其余118个≥50MB文件共62.85GB无整文件hash重复组。
- 机器实际可见2×NVIDIA A100-SXM4-80GB，2026-09-05本轮已用于no-N4隔离教师训练；历史预注册仍常用24GiB/arm安全门。不要因为实际80GB就扩大搜索网格。Full972 width16/routing formal peak约20.29–20.37GiB，width24约30.28GiB。

创建大cache前必须重新运行`df -h`；删除任何材料前遵守[`AGENTS.md`](AGENTS.md)的授权与归档规则。

## 强制完成流程

当前流程：完整链路审视与七问 → 预注册 → 隔离实现/合成合同 → GPU smoke → formal → 全模型冻结后新train-only inner一次性门 → 全过才固定Full972重训与新外部另行预注册 → 正负归档/方法百科/必要的Leaderboard与校验。历史fixed validation/outer/confirmation/test270/UCSD75均不再用于新方法选择或模型重测；既有来源身份核验必须单独标明不是模型评测。

任何test测评完成但未更新Leaderboard时，任务不算完成。

## 当前权威入口

- 仓库流程规范：[`AGENTS.md`](AGENTS.md)
- 当前状态：本文件
- 一目了然榜单：[`AFLOW_BRATS_LEADERBOARD.md`](AFLOW_BRATS_LEADERBOARD.md)
- 小白方法百科：[`AFLOW_BRATS_METHOD_CATALOG.md`](AFLOW_BRATS_METHOD_CATALOG.md)
- 新实验模板：[`EXPERIMENT_IDEA_TEMPLATE.md`](EXPERIMENT_IDEA_TEMPLATE.md)
- 结果账本：[`AFLOW_BRATS_RESULTS.md`](AFLOW_BRATS_RESULTS.md)
- 周报：[`AFLOW_BRATS_WEEKLY_REPORT.md`](AFLOW_BRATS_WEEKLY_REPORT.md)
- 历史长交接：[`AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md`](AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md)
- 自包含实验归档：`archive/research_records/`
- 最新LoMix-3D负结果：[`archive/research_records/2026-08-19_lomix3d/RESULT.md`](archive/research_records/2026-08-19_lomix3d/RESULT.md)
- 最新DCD负结果：[`archive/research_records/2026-08-19_dcd_detail_distillation/RESULT.md`](archive/research_records/2026-08-19_dcd_detail_distillation/RESULT.md)
- 最新KAS负结果：[`archive/research_records/2026-08-20_iad_kas_assimilation/RESULT.md`](archive/research_records/2026-08-20_iad_kas_assimilation/RESULT.md)
- 最新hard mL1-ACE负结果：[`archive/research_records/2026-08-20_hard_ml1ace/RESULT.md`](archive/research_records/2026-08-20_hard_ml1ace/RESULT.md)
- 最新VM Dice负结果：[`archive/research_records/2026-08-20_vm_dice/RESULT.md`](archive/research_records/2026-08-20_vm_dice/RESULT.md)
- 最新Batch-CVaR50负结果：[`archive/research_records/2026-08-20_batch_cvar50/RESULT.md`](archive/research_records/2026-08-20_batch_cvar50/RESULT.md)
- 最新DMRFB部分正向结果：[`archive/research_records/2026-08-21_dmrfb_bottleneck_adapter/RESULT.md`](archive/research_records/2026-08-21_dmrfb_bottleneck_adapter/RESULT.md)
- 最新DMRFB interpolation Dice正向/Point失败结果：[`archive/research_records/2026-08-21_dmrfb_weight_interpolation/RESULT.md`](archive/research_records/2026-08-21_dmrfb_weight_interpolation/RESULT.md)
- 最新972例voxel-KD数据机制正向/未晋级结果：[`archive/research_records/2026-08-21_voxel_kd_data972/RESULT.md`](archive/research_records/2026-08-21_voxel_kd_data972/RESULT.md)
- 最新native-volume 972例voxel-KD Point恢复/未晋级结果：[`archive/research_records/2026-08-21_voxel_kd_data972_native_sampling/RESULT.md`](archive/research_records/2026-08-21_voxel_kd_data972_native_sampling/RESULT.md)
- 最新coverage/replay数据机制正向/Point与KD Scan失败结果：[`archive/research_records/2026-08-24_coverage_small_lesion_replay_kd/RESULT.md`](archive/research_records/2026-08-24_coverage_small_lesion_replay_kd/RESULT.md)
- 最新full972完整轨迹development冠军/CI门失败结果：[`archive/research_records/2026-08-24_voxel_kd_fulltrajectory_972/RESULT.md`](archive/research_records/2026-08-24_voxel_kd_fulltrajectory_972/RESULT.md)
- 最新972预算机制正向/极近KD200但未晋级结果：[`archive/research_records/2026-08-24_voxel_kd_972_draw_budget/RESULT.md`](archive/research_records/2026-08-24_voxel_kd_972_draw_budget/RESULT.md)
- 最新Full972 width24跨seed未复现/容量扩展失败结果：[`archive/research_records/2026-08-26_full972_width24_fulltrajectory/RESULT.md`](archive/research_records/2026-08-26_full972_width24_fulltrajectory/RESULT.md)
- 最新GCSAM inner/test探索性结果：[`archive/research_records/2026-08-31_gcsam_full972_stability/RESULT.md`](archive/research_records/2026-08-31_gcsam_full972_stability/RESULT.md)
- 最新fresh三seed共识Stage A负结果：[`archive/research_records/2026-09-01_fresh_multiseed_consensus_distillation/RESULT.md`](archive/research_records/2026-09-01_fresh_multiseed_consensus_distillation/RESULT.md)
- 最新GT最佳老师routing validation负结果：[`archive/research_records/2026-09-01_gt_best_teacher_routing_distillation/RESULT.md`](archive/research_records/2026-09-01_gt_best_teacher_routing_distillation/RESULT.md)
- 2026-08-19阶段交接归档：`archive/handoffs_20260819/`
