# AFlow repository instructions

<!-- RACE_V2_STAGE_20261029 -->
## 2026-10-29：新授权RACE-v2与双队列推理阶段

用户新授权覆盖此前“一台worker分卡”安排及已关闭组不自动新训边界，仅本新阶段。从RACE164 best新建优化器微调（不是无缝续训），首选80新增轮，必须在972/108独立载荷SHA、双卡完整合成/尾批/重载与预算冻结后开训。1154259 GPU0+1仅训练；1154121 GPU0仅RACE校准、GPU1仅w32互补融合，四物理UUID已核验互异，独占锁且只有总控改分配。首阶段先1训练候选、最多2；初始推理2模型/16前向每例。A32配置、B12融合，统一患者五折与10000患者整簇bootstrap；108选择后最多2固定候选270，旧RACE270八视图完整SHA通过则复用，新权重须自身预测。保留test-tuned历史，单模型和纯SegResNet融合分报，禁GT病例路由、nnU-Net和额外模态。

用户资源“24小时起步，高利用率可96小时，低利用率5小时查杀”；当前只冻结创建时间+24h的保守窗口：推理北京时间10月30日12:59:34，训练计算13:17:37／整体14:17:37，条件延长不预支。新持久总<=0.75GiB且实时空闲>=20GiB，大缓存/完整优化器各worker/tmp，保留best/latest和恢复备份。不重启任何旧协调器。正式新优化器微调已启动，数据／数值／预算及首完整epoch／108验证／CPU重载通过，无新270成绩。[本轮总控与分工](archive/research_records/2026-10-29_race_v2_stage/RUNBOOK.md)。
<!-- END_RACE_V2_STAGE_20261029 -->

<!-- RACE164_TEST270_20261029 -->
## 2026-10-29：RACE164叠加既有最佳推理已完成

原270/119，RACE原始Scan 0.783374，固定8镜像＋解码＋后处理Scan 0.789263、Subject 0.775438、Point 256/270；同权重差+0.005889，描述性95%区间[0.0036050984607035896, 0.008361956075480034]。训练0、2160病例前向，两个输出共用8视图。继承test-tuned历史，无当前新搜索或outer/confirmation。本轮结束不自动重测或续训。[结果](archive/research_records/2026-10-29_race_e164_inference_test270/RESULT.md)。
<!-- END_RACE164_TEST270_20261029 -->

<!-- RACE_DDP_FINAL_20261028 -->
RACE-Seg双卡单算法complete，累计200轮，最佳164轮108 Scan 0.763266。历史参考差+0.009990，描述性区间[0.0009791717071146928, 0.021314055377539866]；历史提升门True。同期对照已由用户取消，不能单独确认算法因果收益。无新270/outer/confirmation，本轮结束不自动恢复。[结果](archive/research_records/2026-10-28_race_seg/RESULT.md)。
<!-- END_RACE_DDP_FINAL_20261028 -->

<!-- AUTO_INFERENCE12H_20261028 -->
## 2026-10-29：12小时无训练推理研究完成

本轮至少12小时自动推理研究已完成。固定w32第171轮/T1ce/原270/119，网络权重训练0。最高Scan **0.784382113**、Subject **0.771378811**、Point **260/270**，8次前向；对原始差+0.011154073。完成7,896,659有限全局元组，全部test-tuned，选择后区间未校正搜索，无独立泛化确认。270缓存掩膜审计及API神经复核范围见[最终报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/RESULT.md)。不自动扩搜或重启旧实验，独立RACE不受影响。
<!-- END_AUTO_INFERENCE12H_20261028 -->

<!-- INFERENCE_SELECTED_TEST270_20261028 -->
## 2026-10-28五镜像＋体积解码固定270测试已完成

用户明确“你去270个test上推理就行了”，仅授权验证已选同一w32第171轮五镜像＋温度2体积解码。270/119 Scan0.780995、Subject0.768306、Point255/270；对同权重原始推理差+0.007767，配对95%区间[0.004495999901909861, 0.011791830233831117]，配对区间支持本复用队列的探索性正差。 单A10080GB，原生载荷/index逐SHA相同，batch8/尾6，每例五视图恰一次；训练0，无本次test调参或新outer/confirmation。内部复用test-informed探索，完整评估结束，不自动扩搜/换参/重测。旧验证搜索的false访问标志保留为历史阶段，本次新测试访问见新归档；独立RACE训练不受影响。[结果](archive/research_records/2026-10-28_segresnet_inference_selected_test270/RESULT.md)。
<!-- END_INFERENCE_SELECTED_TEST270_20261028 -->

<!-- INFERENCE_SEARCH_20261028 -->
## 2026-10-28固定w32第171轮推理搜索已完成

用户授权worker1152682无训练推理穷举，并明确选32通道171轮单模型。32037配置完成原108/50验证；108验证原始0.753271→搜索最高0.761779（差+0.008508），Subject0.750409，Point96/108；患者分折0.761204。 全部复用验证选择后的描述性证据，无新270/outer/confirmation；有限搜索结束，不自动扩搜或测试。独立RACE双卡训练不受本批终态影响。[结果](archive/research_records/2026-10-28_segresnet_inference_search/RESULT.md)。
<!-- END_INFERENCE_SEARCH_20261028 -->

<!-- RACE_DDP_AUTH_20261028 -->
## 2026-10-28 最新用户授权：取消对照，双卡同训RACE-Seg累计200轮

**双卡真实启动已核验：** 算法从完整5轮/610更新接第6轮，双rank严格恢复、完整第6轮/732更新及108验证/CPU重载/SHA同步通过，当前进入第7轮；首个完整双卡epoch198.5秒，旧单卡430.5秒。早期分数非最终收益，当前看DDP_STARTUP_VERIFIED与PIPELINE_STATUS。最终门后无新测试或自动续训。

用户明确“不需要对照了吧，双卡直接训就好了”，覆盖此前两组分卡和两组200轮迁移。新control在完整第4轮/488更新停止、完整model/optimizer和best/latest保留，禁止自动恢复。旧migrate200已终止，coordinator200/resume200未启动，禁止启动。算法保留单卡完整epoch模型/AdamW后接双卡DDP，不从头训练；全局有效8、每卡微批2/本地4/累积2、122更新/轮，累计200轮/24400保留更新及原200轮LR不变。当前切换真实状态见[archive/research_records/2026-10-28_race_seg/DDP_MIGRATION.json](archive/research_records/2026-10-28_race_seg/DDP_MIGRATION.json)，新监督coordinator_ddp200、数值入口ddp200与结题finalize_ddp200；须新双卡纯合成五次更新/尾批/严格重载/参数SHA及每卡<=38GiB门通过后进入患者训练。旧EXECUTION_FREEZE不改，新增DDP200_FREEZE。原972/108、单T1ce、原生SHA固定；两卡各54验证例统一汇总。原同期收益门不再可评估，只以冻结历史w32的108报告作描述性比较；不称相同随机轨迹或算法因果消融。原24小时截止/持久<=4GiB/空闲>=20GiB不变，算法工程异常最多2次，无新270/outer/confirmation。见[archive/research_records/2026-10-28_race_seg/RUNBOOK.md](archive/research_records/2026-10-28_race_seg/RUNBOOK.md)。
<!-- END_RACE_DDP_AUTH_20261028 -->

<!-- RACE_AUTH_20261028 -->
## 2026-10-28 新授权 RACE-Seg 算法实现与24小时匹配实验

**13:03正式启动核验（更新）：** 12:56正式训练已启动；两组都完成第1轮122真实更新、108完整验证和best/latest严格重载，当前最新进度见新PIPELINE_STATUS与两组直接HEARTBEAT。患者模型前按硬件预算冻结各173轮/21106更新、微批2/有效8；共同初始化SHA匹配、原生与128载荷SHA完全一致。首轮为随机初始化早期结果，不代表最终收益。见`archive/research_records/2026-10-28_race_seg/STARTUP_VERIFIED.json`。

准备阶段历史记录：

用户要求调研后实现自己的算法，提供双A10040GB，并明确“24小时，你自己一直干不需要我授权”。当前worker1152470两卡空闲核验通过，源码实现与原生整脑GPU合成门通过，缓存准备进行中，尚无新患者模型训练结果。新目录`archive/research_records/2026-10-28_race_seg/`，数值入口`scripts/race_seg_20261028/`及`sammed3d/race_seg.py`。
本轮control/race两匹配新组、相同随机初始化32通道骨干，原972/108与患者隔离不变；最多各200轮/24400更新、有效batch8、微批仅依合成吞吐冻结。局部跨半球参照、类型对比和参照失效训练是候选组合，不预称首创或保证提升。两卡各一组，不加载已结束171轮权重续训；所有旧组保持终态。
原生与128载荷SHA数据门、显存与预算门后由新coordinator启动训练；24小时截止以AUTHORIZATION.json为准，工程恢复每组最多2次，持久新增<=4GiB且空闲>=20GiB，大缓存/优化器仅worker/tmp。新组仅训练和108验证，无自动270/outer/confirmation/外部测试授权。真实状态见新PIPELINE_STATUS/HEARTBEAT，不把合成通过称为患者训练或新精度结果。
<!-- END_RACE_AUTH_20261028 -->

<!-- WIDTH32_TEST270_20261028 -->
## 2026-10-28 最新用户授权：32通道171轮一次270测试

用户明确“去测评一下吧，2张A10040G，速度拉到最快”，覆盖本候选此前无自动test边界，仅已选w32第171轮SHA02a60b2d…一次270/119评测，不恢复训练、不新增16模型测试。固定171轮270评测完成：Scan0.773228、Subject0.760027、Point256/270；对多尺度166差+0.001842，95%CI[-0.003712439459196132, 0.007115013465642496]。配对区间跨0，没有确认稳定增益或等效。[结果](archive/research_records/2026-10-28_segresnet_w32_e171_test270/RESULT.md)。 worker1152470双A10040GB，固定单T1ce、native1mm/160×208×168去填充、阈值0.5、bf16、主输出每例一次，两卡偶/奇槽位各135例，batch只按合成硬件吞吐选。每例N4一次同时生成两栅格，四载荷SHA须匹配历史。MODEL_PREACCESS独占保护，失败不自动重做前向，raw完整只恢复统计/出版。旧验证门失败保留，test-tuned内部复用探索，无本次test调参或新outer/confirmation。持久新增<=2GiB/剩余>=20GiB，大缓存只worker/tmp，4小时预算。当前真实状态与结果在archive/research_records/2026-10-28_segresnet_w32_e171_test270/。
<!-- END_WIDTH32_TEST270_20261028 -->

<!-- WIDTH_RESUME_20261027 -->
## 2026-10-27 最新用户授权：撤销早停后接续累计200轮

用户在确认早停后明确“还是和之前一样训练完看看”，授权本新阶段恢复两组累计200轮预算，覆盖此前禁止后续接续的约束，仅限本次两组。两组各200轮完成；w16最佳166轮108 Scan0.747497；w32最佳171轮108 Scan0.753276；32−16差+0.005779，收益门False。[结果](archive/research_records/2026-10-27_segresnet_width_resume/RESULT.md)。 新归档archive/research_records/2026-10-27_segresnet_width_resume/；旧width目录85/50停止记录保持历史终态。w16从85轮10370更新接86，w32从50轮6100更新接51；完整model/optimizer恢复，微批8/4、有效8、122更新/轮、累计24400更新和原LR不变。只复用冻结train_v3数值代码，不加新算法/预训练/种子，不从头训。沿用worker1149729两A10080GB和原截止10月28日17:30训练/18:00整体，每组异常工程恢复最多2次、用户预算恢复单列；持久新增<=4GiB/剩余>=20GiB。无新270/outer/confirmation。原前50未过筛事实保留，当前状态看新PIPELINE_STATUS，禁止重启旧coordinator/guard。
<!-- END_WIDTH_RESUME_20261027 -->

<!-- WIDTH_FIRST50_USER_20261027 -->
## 2026-10-27 最新终态：用户前50轮筛选后已停止

用户选择“第50轮仍无优势就停止两组”，覆盖原200轮继续预算。32前50最佳108 Scan0.700532904（46轮）低于16前50最佳0.710403040（45轮），差−0.009870137，描述性患者配对95%区间[-0.018579325,-0.002314227]。已于15:03停止：w16实际85轮/10370保留更新，w32实际50轮/6100更新，模型/优化器完整checkpoint与报告保留；15:04核验两卡0MiB/0%、训练及V4协调器均退出。禁止自动恢复、续训、扩组或新270测试。最终见[结果](archive/research_records/2026-10-27_segresnet_width/RESULT.md)。

此前微批提速是从完整75/45轮接续76/46轮，不是重新随机初始化；微批改8/4，有效8/122更新和LR不变，数值归约/dropout轨迹可能不同。前50比较包含32组46–50轮的新微批执行，必须披露。此次是在看过47轮后新增的预算筛选，不能证明200轮最终排序或32通道普遍无效。旧V2/V3失败为已修复准备/合成事件，当前终态为stopped_by_user_first50_screen，无新270/outer/confirmation。
<!-- END_WIDTH_FIRST50_USER_20261027 -->

<!-- WIDTH_20261027 -->
## 2026-10-27 用户授权16/32通道容量对照（当前优先）

用户明确验证32通道。用户授权前50轮筛选后停止：w16前50最好0.710403、实际结束85轮；w32前50最好0.700533、实际结束50轮。32未超过同期16，撤销剩余200轮预算，不是完整训练终态；不证明32通道普遍无效。[结果](archive/research_records/2026-10-27_segresnet_width/RESULT.md)。 worker1149729两张A10080GB，沿用48小时窗口；本批w16/w32均随机初始化，不加载334或972旧权重。唯一宽度16/32；单T1ce、原972/108、原生1mm、200轮/24400更新、有效8/合成吞吐决策微批累积、同优化器和匹配增强；不加蒸馏/VAE/深监督。数据SHA/合成/预算门后开训，旧蒸馏三组保持complete不恢复。持久新增<=4GiB/空闲>=20GiB，缓存只读复用worker/tmp；每组工程恢复最多2次，训练截止10月28日北京时间17:30、整体18:00。无新270/outer/confirmation。协议与当前状态：archive/research_records/2026-10-27_segresnet_width/。
<!-- END_WIDTH_20261027 -->

<!-- DISTILL_20261026 -->
## 2026-10-26 新授权蒸馏三组（当前优先）

用户授权按讨论开展蒸馏，确认worker1149729两张A10080GB、48小时。三组各200轮完成；control最佳189轮108 Scan0.745119；plain最佳183轮108 Scan0.743886；reliable最佳158轮108 Scan0.738286。明确收益门False。[结果](archive/research_records/2026-10-26_segresnet_teacher_distill/RESULT.md)。 本批control/plain/reliable三组公开334初始化SegResNet，冻结原nnU-Net525为训练内教师，仅972离线软输出与108验证，200轮/24400更新/有效batch8不缩减。数据原生SHA匹配、双卡合成门和预算门后启动；大缓存只worker/tmp，持久新增<=4GiB/空闲>=20GiB。每组工程恢复最多2次，不按分数调参或续训；训练截止10月28日北京时间17:30、整体18:00。无新270/outer/confirmation，不恢复旧已关闭组。真实状态与完整七问见archive/research_records/2026-10-26_segresnet_teacher_distill/。
<!-- END_DISTILL_20261026 -->

## 2026-10-22 随机初始化完整550轮B方案（新授权）

- **23:27正式训练启动核验（最新）：** worker1145433原972/108参考及原生载荷SHA均匹配历史、1080标签几何准确；23:24数据/预算门通过，首轮62/62真实更新和108完整验证及best/latest保存完成，23:29第2轮累计113真实更新。CPU重载检查点有限权重、优化器状态、计数均通过。首轮Scan0/Subject0/Point4/108只是随机初始化早期结果，不代表最终收益。global12/每卡6、62步/轮、550轮目标及最新截止不变；人工“守到启动”任务已完成，V3自动协调器继续，无新270/outer/confirmation。当前看STATUS/HEARTBEAT/PIPELINE_STATUS，旧“尚无训练”仅历史准备阶段；详见STARTUP.md/STARTUP_VERIFIED.json。

- **22:01新batch已选：** worker1145433三档完整合成均通过，global12/每卡6最快（6.518块/秒，reserved约63.16GiB/卡）；按用户授权每轮62步/744原始块，550轮34100更新尝试/409200原始块，相对旧412500块少0.8%，非相同更新预算。纯更新约17.44小时、含25%及2h余量约23.80小时，不含数据重建；初始LR仍0.01。当前V3等待新数据SHA门，尚无患者模型训练，见BATCH_DECISION与BATCH_SELECTION.md。

- **21:55最新用户修订（优先）：** 原1144973在13:01因canonical_cache相对路径记录错误退出数据准备，患者训练0轮；随后旧节点不在活动列表且登录不可用。用户确认新worker1145433双A10080GB，并将资源延长48小时、明确要求增大batch；用户选择每轮约750原始训练块、相应减少更新步数的550轮口径。旧global3/250步/137500更新与旧截止已被本条覆盖。
- 新V3先合成测global batch4/8/12（每卡2/4/6），按原始训练块/秒选最高吞吐；较小batch若达到最高吞吐97%则优先较小者。每轮round(750/batch)步，即4→188步/752块、8→94步/752块、12→62步/744块。不称相同更新预算/等FLOPs，初始LR仍0.01，原B训练目标和预热不变。只有硬件吞吐选batch，无患者分数参与。
- 修复入口native_builder_v2只改/tmp绝对路径元数据，原init/native_arrays/worker函数AST完全一致，历史源与失败保留。当前bootstrap_v3在新节点重建数据，coordinator_v3执行batch合成门，尚无患者模型训练。训练最迟2026-10-24北京时间20:00、整体21:00；其余单T1ce/原972/108/270、SHA、550轮、验证门、持久<=4GiB及空闲>=20GiB、无新270/outer/confirmation边界不变。最新方案见AMENDMENT_002_WORKER_BATCH.md，旧协调器不可重启。

- 用户在评估后明确同意尝试重新审视的B，提供新worker1144973双A10080GB，资源24–48小时。当前只开一个新实验、两卡共同训练；从随机初始化ResEnc L开始，seed20261013/global3（2+1）/550轮×250更新/SGD初始LR0.01。不加载旧e525或SegResNet334权重，不恢复已结束的150轮A/B。
- 新配方为局部真实标签监督从第1轮开始、系数0.5；一致性前50轮关闭，第51–100轮线性0.002→0.1，之后固定0.1。每更新两个训练视图，部署仍原全脑滑窗、概率0.5；训练计算量高于历史普通550轮，不称等FLOPs或新鲜配对消融。原972/108/270、单T1ce和患者隔离不变。
- 新归档`archive/research_records/2026-10-22_nnunet_local_full550/`，旧缓存已删除，先在新worker/tmp重建972/108的128³参考及原生缓存；最终原生载荷SHA必须与旧组一致。当前硬件CUDA、6项CPU及完整双卡合成门通过，45/45实际更新、两rank参数hash一致、严格重载与几何通过；约0.746秒/更新，纯更新28.5小时、含余量约37.6小时。数据准备进行中，尚无新患者模型训练。只有数据/合成/吞吐预算门通过才自动进入完整550轮；预计超过最长48小时则停在准备，不擅自缩轮数。
- 双卡数值合成通过后，旧runner在数据加载器退出阶段滞留；新增V2有界清理入口及协调器接管，只改成功后的进程收尾，数值源码/合成结果保留，患者训练0轮；当前V2等待数据门，旧协调器不可重启。
- 训练最迟2026-10-24北京时间10:00，整体10:30；持久新增<=4GiB、空闲>=20GiB，大缓存仅worker/tmp。新组工程恢复最多2次，不按分数调配方或扩组。108验证1/每25/550轮选优；明确收益门较原e525至少+0.01、患者配对95%区间下界正、Subject与Point不降。本阶段无新270/outer/confirmation像素或模型访问。
- 旧2026-10-21两组已完成各150轮：A选0轮、训练后最好40轮0.783914；B选120轮0.785394，对e525+0.000722、95%区间跨0且Subject下降，两组门均未通过。旧目录complete属于真实终态，不自动重启。

## 2026-10-21 用户授权清理旧缓存（已完成）

- 用户明确“磁盘可以帮我清空下，之前什么缓存可以删了，反正留着建设缓存的脚本就行了”。已清除旧128³训练972/验证108/测试270，以及UCSD75/WAW术后31的输入图像/标签缓存，持久盘共14.21875GiB；另删除主节点/tmp中先前转移的公开334缓存载荷3.26171875GiB。保留原索引、构建源码及SHA、原始源数据、全部模型/历史预测/报告；148个既存模型文件核对仍在。持久空闲约37.44GiB。
- 旧15:20迁移收据只描述当时完整备份；现在/tmp公开334备份载荷已按最新授权删除，不能声称仍可直接复制恢复。现有缓存目录仅保留索引/删除标记，不得把index存在当作payload存在。重建先按原清单生成到新临时目录，核对历史SHA；将来原生缓存重建须先补回其依赖的128³参考缓存。不得借重建自动恢复已关闭的实验或新增测试评分。
- 删除前已检查worker1143438两主进程及其子进程，共14个，候选文件无打开/映射。当前worker的`/tmp/aflow_error_balance_20261021`及150轮两算法运行资料完整保留，未中断训练；清理没有数组解码或新模型/评测执行。详情和重建命令见`archive/reviews/2026-10-21_unused_cache_cleanup/RESULT.md`与`REBUILD.md`。

## 2026-10-21 nnU-Net e525基线上的算法候选与24小时资源

- **18:26 V6仅接管汇总监控：** V5汇总器17:08因I/O错误退出，但原训练PID130482/130483持续更新，没有训练中断或重载。V6附着同一两个PID恢复汇总与结束归档，不启动训练、不增加已用2次工程恢复。18:25阶段A完成122轮/正在123轮、训练后最好0.783914；B完成61轮/正在62轮、60轮最好0.784031，均未超过0轮0.784665。150轮配方/截止不变，无新270/outer/confirmation；当前看V6汇总和两组直接HEARTBEAT。详见AMENDMENT_005_MONITOR_ATTACH.md。

- **15:20 V5第二次/最后一次工程恢复：** 两组V4分别在A44轮/11000尝试（10997实际）、B19轮/4750更新后因空闲<20GiB停止，旧incomplete仅失败阶段。未用公开334派生缓存已逐SHA验证备份至主节点/tmp并转移，释放3.2617GiB、余量约23.2GiB；原始数据、全部历史权重及源index保留。V5保持原run_v4数值代码，A接续45轮、B接续20轮，各150轮预算/LR/算法/截止不变，每组恢复计数现为2，不再自动恢复或扩组。当前状态以V5及两组HEARTBEAT为准。A训练后最高验证0.783914、B0.781598，均未超过0轮0.784665；无新270/outer/confirmation。转移路径、SHA及限制见AMENDMENT_004_DISK_RESERVE_RECOVERY.md。

- **12:33 V4工程接续：** 数据门12:10通过，正式预算已冻结各150轮/37500尝试。V3导出重复CPU共享参数导致存储限额，A完整8轮/2000实际更新、B完成起点验证但0更新后退出；去重导出逐张量完全一致，391MiB，原膨胀副本/失败报告/源码均保留。V4已让A从完整8轮接续第9轮、B复用原起点报告后开始第1轮。两组各计一次工程恢复，最多仍2次，配方和截止不变；当前以V4监督、两组HEARTBEAT为准，不把failed_stage_v3的incomplete当当前终态。详见AMENDMENT_003_STORAGE_RECOVERY.md。

- **用户最新修订：取消普通微调对照，改跑两个算法组。** control只有历史25次合成更新，无患者训练/模型验证；旧v2协调器已停止，禁止重启旧配对入口。新V3只允许A体积预算困难背景竞争（GPU0）和B训练时局部放大视图+全局停止梯度一致性（GPU1），分别与已有e525比较。A的5项损失CPU检查、B的6项局部视图CPU检查及V3两组完整配方GPU门均通过；A/B每步约0.500/0.994秒、reserved约29.06/31.70GiB，真实训练状态看V3收据。具体新七问及修改后的前瞻门见`AMENDMENT_002_TWO_ALGORITHMS.md`，优先于旧IDEA中的普通微调对照要求。
- 用户要求在单模型0.811283的nnU-Net ResEnc L基线上开展算法改进，并提供worker1143438双A10080GB、约24小时；本轮不恢复旧SegResNet或旧550轮任务。新归档为`archive/research_records/2026-10-21_nnunet_error_balance/`。取消普通微调后，没有单独识别额外训练与新增算法的贡献，不得称同预算因果消融。
- 历史V1控制/算法GPU合成门均通过（该control已取消）；当前V3两算法GPU门均通过，每卡batch3。数据门已通过且两组开始/接续训练；正式状态以V4 PIPELINE_STATUS、FREEZE及两组STATUS/HEARTBEAT为准，不能把旧0.811283称本批成绩。
- 固定单T1ce、原972/108/270、患者隔离、旧1mm/L计划和e525初始化SHA；仅新增训练目标，训练体积参照只由972确定。最多200轮/每轮250更新，患者模型前按合成吞吐冻结共同预算；每组最多2次完整checkpoint工程恢复，不因分数扩组/改参。训练截止2026-10-22北京时间08:00，整体截止10:30；本阶段不新增270/outer/confirmation像素或模型访问。
- 算法及相关技巧只称候选设计，困难挖掘/自适应损失已有先例，不称已确认首创或保证提分。新大缓存与完整优化器checkpoint仅worker/tmp，持久新增<=1.6GiB、保留>=20GiB空闲；不删除历史权重。旧270已test-tuned，当前属于test-informed开发，不能称独立泛化。

<!-- NNUNET_RESENC_20261013 -->
## 2026-10-13残差编码器组最终完成（2026-10-14）

- 550轮完成，108选525轮Scan0.784672，验证门通过。270单模型Scan0.811283，最佳三模型融合Scan0.812965、Subject0.797512、Point266/270；相对旧榜首+0.037074，95%CI[+0.030140,+0.044262]。达到本批0.8均值目标，全部test-tuned，非独立泛化。
- 单T1ce、原972/108/270和患者隔离固定；137500次更新尝试、137449次实际更新，预算受限550轮参考，不称默认1000轮完整复现。唯一525权重SHA31f979ac…，270原生载荷与历史完全一致。
- 原worker1132660完成训练；字段归档失败发生于270访问前，原节点随后不可连接。用户新提供1133854仅接续评测，V3已完成。旧源码和失败记录不改；当前访问以MODEL_PREACCESS、TEST_ACCESS_ADDENDUM和最终报告为准，训练STATUS中的false只描述训练阶段。
- 最佳融合为新模型0.75、旧173及组合166各0.125、统一阈值0.4；固定六项全保留。全部test-tuned，未将270标签加入梯度，无新outer/confirmation。本组结束，不自动增加候选、续训、换epoch或重测。[报告](archive/research_records/2026-10-13_nnunet_resenc_reference/RESULT.md)。
<!-- END_NNUNET_RESENC_20261013 -->

## 2026-10-13 当前方法目录改为单T1ce SegResNet主线

- 用户明确要求删去现行方法目录里的旧交互分割“方法1”，从SegResNet路线讲述。固定比较基线仍是原972纯SegResNet第193轮：128³验证108 Scan0.740032，内部测试270 Scan0.759179；原生1mm、分块及融合结果按同栅格和推理成本单列。当前采用SegResNet作为研究主线，不把现有最好分数称为模型架构的理论极限。
- `AFLOW_BRATS_METHOD_CATALOG.md`现为精简的SegResNet入口；改版前全文和旧方法的正负结果已保留在`archive/reviews/2026-10-13_segresnet_method_catalog_restructure/`。本次只整理文档，无新训练、验证、test、outer或confirmation访问；不恢复旧实验。下方2026-09-16“当时唯一SAM主线”属于已被9月17日方向切换覆盖的历史安排。

## 2026-10-10 有效区域监督单组：用户最新授权

- **本轮终态：** 选175轮108 Scan0.746077、Subject0.734457、Point98/108；对旧多尺度166差-0.007015、95%CI[-0.023860,+0.003203]。未达到事前明确验证收益门；只评价当前固定配方，不证明所有有效区域策略无效。 已完成200轮/24400更新及严格重载，仅108验证、无新270/outer/confirmation。本组不自动恢复/扩组/测试。

- 用户明确“ok开始干活吧，先检验有效区域监督”，授权worker1127648双A10040GB先合成检验、数据门后执行一组原972/108单T1ce原生多尺度SegResNet。唯一变化为图像几何有效区域损失；原334只复用150轮预训练，不加入回放，不开nnU-Net或其他组。
- 从原公开预训练150轮初始化，原global8/每卡4/122更新，固定200轮上限/24400更新；不设旧50轮分数早停，不恢复原166已测试模型续训。验证收益门比多尺度166至少+0.01、配对95%区间正、Subject/Point不降；本批无新270/outer/confirmation像素/模型访问。
- CPU合成通过；双卡完整批量须reserved<=38GiB且尾批/参数同步/AdamW计数通过。新worker/tmp原生缓存须图像/标签逐文件SHA与旧载荷一致；持久新增<=8GiB/剩余>=20GiB。14小时训练墙钟、完整epoch工程恢复最多2次，不按分数改配方或扩组。
- 真实阶段见`archive/research_records/2026-10-10_segresnet_valid_region_supervision/PIPELINE_STATUS.json`和对应workspace STATUS；CPU与双卡完整批量合成门已通过，reserved峰值约35.75GiB/卡，v2 bf16全1损失/梯度误差0；原972/108缓存载荷SHA已完全匹配旧组，2026-10-10北京时间12:21正式启动训练，真实轮数见新workspace STATUS；旧0.771386不是本组新成绩。

## 2026-09-29 多尺度166一次270测试：用户最新授权

- **本次终态：** 原270扫描/119患者，native1mm裁剪栅格排除填充，Scan 0.771386、Subject 0.757817、Point 248/270。对同格旧组合166差+0.005115、患者配对95%CI[+0.000590,+0.009713]；复用队列的95%配对区间为正，但97.5%区间仍跨0；这是小幅收益的探索性证据，尚不能认定稳健或独立泛化提升。原验证门未过，test-tuned开发/复用内部探索；此次唯一166模型已访问270，无当前test调参或新outer/confirmation。评测与归档结束，不自动重测或新训。[报告](archive/research_records/2026-09-29_segresnet_multiscale_e166_test270/RESULT.md)。
- 唯一166模型SHA0645c5e7…；270载荷SHA与历史完全一致，135批每例恰一次/主输出全脑前向；旧参考只复用报告。本组训练FINAL_RESULT中的false是历史阶段事实，以独立测试及TEST_ACCESS_ADDENDUM为当前访问范围。不自动换epoch/阈值/融合或重测。

## 2026-09-29 多尺度辅助监督50轮续训到200轮：用户最新授权

- **本轮终态：完成200轮、24400更新，选166轮108 Scan0.753092。** 同格旧166差+0.001913、配对CI[-0.004999,+0.010082]，冻结提升门未通过。训练/验证阶段完成时无270/outer/confirmation新模型访问；后续唯一166的用户授权270测试已完成，见最新条目，不自动续训/扩组/重测；见本轮RESULT和PIPELINE_STATUS。


- 用户明确“可以训练完看看”，授权从原第50轮latest完整模型、辅助头和AdamW状态续训第51–200轮；撤销本次后续阶段的50轮预算停止，不改原50轮未过门事实，不追加到200轮以上、不改配方或增加其他组。旧50轮源码/报告/权重完整保留，新归档`archive/research_records/2026-09-29_segresnet_native_multiscale_resume50/`。
- 原50轮最佳49轮验证0.718758502较旧同预算0.716710192仅+0.002048310，尚不构成稳定提升。此续训是看过验证后的自适应预算修订，仍为test-informed开发；保持原最终108提升门，无新270/outer/confirmation模型访问。
- 旧worker1109780已不在活动清单，现worker1114913双A10080GB真实CUDA通过。保持每卡4/global8/122更新及原学习率曲线，不借新显存扩大batch；硬件迁移不声称逐位同轨迹。完整50轮checkpoint SHA93a969ad…、全部优化器步6100已核验；新的原生训练/验证缓存需逐文件SHA匹配旧载荷。
- **正式续训已开始：** 新worker训练/验证原生载荷SHA与旧阶段一致，恢复合成检查通过；第51轮完整完成新增122更新，AdamW计数6100→6222且原50轮检查点未改，51轮验证Scan0.712444。继续第52–200轮，不再执行50轮预算停止。真实最新轮数看新PIPELINE_STATUS/HEARTBEAT及新workspace STATUS；旧目录complete仅描述50轮历史阶段。持久新增<=8GiB且剩余>=20GiB，大缓存只放新worker/tmp。

## 2026-09-24 原生SegResNet多尺度辅助监督：用户授权新一组训练

- **本轮终态：按50轮规则停止、6100更新，选49轮108 Scan0.718759。** 同格旧166差-0.032420、配对CI[-0.040790,-0.024388]，冻结提升门未通过。训练/验证完成、无本组270/outer/confirmation新模型访问，不自动续训/扩组/测试；见本轮RESULT和PIPELINE_STATUS。


- 用户最新明确要求“再仔细想其他方法，然后开展训练”，覆盖前一轮“当前没有候选不新训”的结论，但不恢复任何已关闭旧任务。本轮只新增一组单T1ce、原972/108/270、原生1mm整脑SegResNet训练：相对公开预训练＋原生分辨率旧组合，唯一研究变化为训练时1/2与1/4尺度辅助监督，部署仅主输出1次前向。七问与冻结边界见`archive/research_records/2026-09-24_segresnet_native_multiscale_supervision/IDEA.md`。
- **正式训练已启动：** worker1109780双A10040GB真实CUDA通过；训练972和验证108原生缓存图像/标签载荷均与旧SHA完全一致，全局8/尾批4合成通过。2026-09-24 06:22:27 UTC已完整结束第1轮122更新并完成108/50验证（Scan0.611496），当前监督与训练仍运行。真实最新轮数以本轮PIPELINE_STATUS/HEARTBEAT和持久训练STATUS为准，不能把首轮当最终成绩；自动监督到50轮门/200轮上限后归档，不自动新增270模型访问。
- 一组最多200轮，每轮972各一次、两卡全局8/122更新，108原生栅格逐轮选Scan；第50轮前50最佳若低于旧组合前50最佳0.716710192加0.003则停，最终提升门为验证Scan至少旧166的0.751179+0.01、50患者配对区间下界正、Subject/Point不降。无本候选新270模型推理或outer/confirmation，虽受既往test影响须标test-informed。持久新增<=8GiB、空闲>=20GiB，大缓存只放worker/tmp。

## 2026-09-24 270测试特征审视训练数据：最新用户授权及终态

- 用户明确允许直接从270测试数据特征出发，判断如何处理原972训练集；此许可覆盖本批早先的“无新270访问”边界，仅对应新归档，不恢复已关闭旧实验。已经访问270的既存128³图像缓存和标签缓存作诊断，复用固定native173逐例报告；**全部结论test-tuned**。没有把270标签加入训练、没有新模型前向或新270评分、没有outer/confirmation。
- 诊断完成：14项固定影像特征的训练/测试患者分组5折区分AUC0.554；所有单项中位数差<训练IQR0.16倍，与旧173逐例Dice的相关绝对值<0.19。128³肿瘤体素中位数训练25363/test23975.5；低于训练p10的比例10.08%/8.52%；1–5体素碎片占总阳性体素0.00472%/0.00482%，外沿5体素阳性比例0.217%/0.284%。这些是不同概念，不能把128³外沿误报成原生裁剪丢失。训练/验证/测试原生图像自身裁剪框外标签历史聚合分别8/4/4体素。
- 本轮决策：没有足够的目标域差异支持新病例加权、直方图匹配、自动标签碎片清理或扩大脑区框；既有B目标域加权25轮验证最好为0轮。因此本批仅完成分析、不新开训练或改训练集，不称数据处理方向被普遍否定。真实范围、访问、复核见`archive/reviews/2026-09-24_test270_image_profile/RESULT.md`。worker1109780两张A100正常，但无本批GPU运行任务。旧最高0.775891仍是test-tuned，不作为独立泛化。

## 2026-09-24 病灶均衡选块：用户授权诊断后条件训练

- **本轮终态：诊断完成，未过准入，不启动训练。** 原972/444、20轮模拟，3350区域中预设小区域881个（1–5体素）；新选块覆盖仅+0.008417518（0.842个百分点），低于0.05门；全区域+0.003236167。原生标签SHA与旧缓存一致，独立复核通过。训练0轮/0更新、模型前向0，无新验证/test/outer/confirmation。不能把覆盖增量叫Dice或把1–5体素区域直接叫临床小病灶；绝对覆盖仅有界限，不冒充实测。见本轮RESULT；本实现关闭，不放宽门或自动扩组。

- 用户明确“先开始诊断然后开始训练”，指定worker1109780（双A10040GB）；两卡真实CUDA计算已通过。按已说明方案先做原972训练标签覆盖诊断，仅诊断准入通过后启动一组同166起点、最多200轮训练与108验证，不增加第二组，不恢复旧A组。
- 新归档为`archive/research_records/2026-09-24_segresnet_lesion_balanced_patch_proposal/`；真实阶段看PIPELINE_STATUS。只改变肿瘤分支的选块规则，保持每扫描每轮一128³块与八块原生推理。50轮同预算提升不足0.003停止；最终108验证较旧A至少+0.01且配对区间为正、Subject/Point不降。
- 诊断旧小区域覆盖须<0.95、新旧覆盖增量>=0.05、全部区域下降<=0.01；不达则不训练、不改门追分。利用两方案相同脑区随机分支严格抵消，先用原生标签计算精确覆盖差；必要条件失败即可终止，绝对覆盖未测则只能报告界限，不能冒充实测。
- 原972/108/270与单T1ce固定；本轮无新270/outer/confirmation像素或模型访问授权。旧test-informed历史保留；大缓存只在worker临时盘，持久新增<=8GiB且保留20GiB空闲。旧10小时批次已结束。

## 2026-09-21 10小时test270定向优化（最新授权，执行中）

- **21:06双卡可见性事件：** 1104883最后HB21:06:12，SSH复查失败，平台活动清单为空。科研计算早已完成，部署权重/原始报告/固定清单已再验SHA完整。剩余窗口由主节点监督持久化终态与端点，不重建、不重训、不新增评测；两临时盘当前不可访问，不冒充全程GPU在线。见WORKER_VISIBILITY_INCIDENT_002和MONITORING_AMENDMENT_005。

- **19:08单卡可见性事件：** 1105022最后HB19:06:25，SSH重复失败且平台活动列表已无此worker；B/C研究已早先结束，不恢复训练。持久权重/报告完整，B270预测与双卡ref179逐文件SHA相同且现场复核通过。主节点/双卡继续原截止；不可声称单卡全程在线，单卡磁盘数仅最后观测。见本轮WORKER_VISIBILITY_INCIDENT_001和MONITORING_AMENDMENT_004。

- **本轮优化阶段已结束，监督仍到23:04:52 UTC：** A200轮/24400更新选新增90轮，单模型8块270 Scan0.772830；B25轮/3050更新停止、最佳为初始化0轮；C根据270错误诊断不启动（无C模型前向）。固定12项输出面板最高为已有native173+组合166等权logits、概率阈值0.4，Scan0.775891、Subject0.761694、Point261/270；对当前同格173差+0.007318、描述性95%CI[+0.003116,+0.012786]。未达0.80及+0.03门，全部test-tuned，不称独立泛化。禁止自动恢复/扩组；剩余窗口为冻结模型拟合诊断、归档和监控。

- 用户批准三候选A分块覆盖/B目标影像域权重/C条件召回损失及最多19项受限输出组合；执行窗口13:04:52–23:04:52 UTC，22:04:52停止训练。此次覆盖旧本批次不扩组/不test条款，仅适用于新归档，不恢复历史关闭任务。
- 原972/108/270、单T1ce和患者隔离固定，270标签允许诊断与选择，不进分割训练。全部成绩test-tuned单列，不作为独立泛化。
- 预算、七问、门审及真实状态见 archive/research_records/2026-09-21_t1ce_test270_10h/；未启动前不称训练中。项目新增<=8GiB，保留20GiB空闲；临时缓存仅worker /tmp。


These rules apply to every Codex run in this repository and are completion requirements.

## 2026-09-21组合166与患者等权172的270测试授权（当前优先）

- **本次终态：** 本次两个固定候选各一次270扫描/119患者测试均已完成。组合166在native grid的Scan 0.766272、Subject 0.750958、Point 236/270；对原native173差-0.002306、95%CI[-0.009078,+0.005791]。患者等权172在128³的Scan 0.759967、Subject 0.745647、Point 250/270；对预训练179差-0.001682、95%CI[-0.008389,+0.005860]。不同栅格不直接相减。验证门未过的事实保留；本次为用户授权、复用内部队列探索，无本批test调优或新outer/confirmation。[组合测试](archive/research_records/2026-09-21_segresnet_pretrain_native_e166_test270/RESULT.md) · [患者等权测试](archive/research_records/2026-09-21_segresnet_patient_equal_e172_test270/RESULT.md)。 真实总状态见共同归档PIPELINE_STATUS/COMPLETION_AUDIT；两次模型访问均完成，不自动重跑。此终态仅对应166/172固定评测，不覆盖其他授权批次。

- 用户明确“我感觉可以去270个test上测一下 因为有提升”，授权前述两候选各一次固定270测试。组合唯一166轮SHA b2a37a7e…，患者等权唯一172轮SHA46ad9f1c…；不重训/换epoch/阈值/后处理。原验证提升门未通过的事实保留，属于用户授权的探索性测试，不事后改写晋级门。
- 共同预注册与代码冻结在`archive/research_records/2026-09-21_segresnet_followup_test270/`，两结果分别在`2026-09-21_segresnet_pretrain_native_e166_test270/`和`2026-09-21_segresnet_patient_equal_e172_test270/`。真实总状态看新目录PIPELINE_STATUS，不沿用旧训练/验证状态。
- 组合在与旧native173相同的160×208×168画布、原始裁剪栅格评分，缓存载荷SHA须与旧收据一致；患者组在原128³缓存评分。不得跨栅格相减。旧native173和193同格逐例报告复用；固定预训练179额外做一次128³→native参考映射，属同一已测试模型重评分、不是第三训练候选，结果也须明确记录双榜。
- 两候选及参考各由独占MODEL_PREACCESS保护，一次前向；raw先落盘，失败不自动重复模型推理。使用bf16 batch2、raw logit>0、确定性评分，10000患者整簇bootstrap。无当前test调优或新增outer/confirmation。完成后更新双榜、四文档/方法卡、访问补记与四校验，保留原训练RESULT中的false为历史阶段，不否认新测试访问。

## 2026-09-21新组合训练与单卡患者等权并行（当前有效）

- **最新终态：两组均完成，当前无运行训练。** 组合200轮/24400更新，选166，原始栅格108/50验证Scan0.751179、Subject0.739474、Point89/108；对同栅格预训练179为+0.001551、95%CI[−0.005990,+0.008267]，对native173+0.005063、CI跨0，对纯193+0.008497、CI[+0.000827,+0.017430]。未达到“比更强单项至少+0.01且两项97.5%区间为正”门，交互差也跨0，不能称协同或全面改善。组合快照SHA `b2a37a7ed4ea73922a454f5f60995d3a8807f18d122539ba0c85d9b5e62c8a18`。
- 患者等权组200轮/24400更新，选172，128³验证Scan0.748017、Subject0.735655、Point100/108；对预训练179差+0.001166、95%CI[−0.006032,+0.009311]，未过明确门。快照SHA `46ad9f1ccbb85a2c8f7790a2d225265ddeb50c667f8091e08bf8006e4039083d`。两组均仅固定108验证，无test/outer/confirmation访问，不能把这些分数当270成绩或跨栅格横比；当前两个实现不自动恢复/延长、换epoch或新增测试。终态与COMPLETION_AUDIT见各自归档，旧“执行中”条目为历史过程。
- 用户已明确“可以那就开始吧 开始训练”，随后授权用另一张空闲A100并行另一idea。当前新增仅两组：公开预训练＋原分辨率组合、公开预训练＋患者等权损失；不自动增加第三组/seed/参数搜索。原先语义分解与预训练128³两组已结束，不能恢复。
- 组合组在worker1104883双A100执行，归档`archive/research_records/2026-09-21_segresnet_pretrain_native_res/`。复用公开334预训练150轮final SHA3726eac6…，在原972的1mm裁剪/160×208×160画布微调200轮，每卡4/global8/122更新；不从已在972微调过的179模型继续加训。匹配原native主要训练配方；本组与固定193/预训练179/native173参考验证统一确定性数值模式，差异已事前披露。
- 单卡组在worker1105022执行，归档`archive/research_records/2026-09-21_segresnet_patient_equal_loss/`。同公开150轮初始化、原128³缓存，固定w_i=972/(444*n_subject)乘每例Dice+0.1BCE后batch平均；不在batch内重归一、不改采样。每轮972各一次，单卡batch8/尾批4/122更新、200轮；单卡与旧双rank的随机数差异已披露。原108验证仍以未加权Scan Dice选优，不能切指标追分。
- 两组均保持单T1ce、SegResNet、原972/108/270与患者隔离；此次执行范围为训练及108验证，无新test/outer/confirmation阶段，历史test定向许可未在本轮使用。正式源代码已各自FREEZE，新增入口不改历史冻结脚本。
- 双卡组合须先重建约26.8GiB临时缓存、同栅格参考评分后训练；单卡复用128³缓存可先运行。真实状态分别看各目录PIPELINE_STATUS/HEARTBEAT及对应workspace训练STATUS。工程恢复最多2次，不因中途分数改变配方或预算。旧两候选的LIVE_STATUS只描述旧批次，不能代表当前新批次无运行任务。

## 2026-09-21 两个SegResNet实验与test定向优化许可（最新）

- 两组结束后，用户询问“预训练结合原始分辨率”是否有依据；本轮仅讨论并登记新候选，未启动第三组。提案在`archive/research_records/2026-09-21_segresnet_pretrain_native_res_proposal/IDEA.md`；CPU结构核对通过，但无组合训练/患者前向/新test。建议起点为公开预训练150轮final（不是已完成972微调179轮再续训），再匹配原native配方；同原始栅格比较两单项与固定193，不能相加跨栅格涨幅或先称协同/原创。此讨论不恢复旧已关闭实验。
- **两组最终完成，当前无本轮运行任务：** 第二候选公开预训练150轮+原972微调200轮已完成，验证选179，checkpoint SHA `7006d44a2a2f765562dc0155f914c967a27b22ffba476cea5e966f01d69f7ad0`。108/50验证Scan0.746851，相对193+0.006818、CI跨0，未过至少+0.01明确增益门；一次270/119同128³测试Scan0.761649、Subject0.745859、Point246/270，对193 Scan差+0.002470、95%CI[−0.004343,+0.009627]，97.5%区间也跨0。它是当前128³自动总体及1× Dice最高点估计，不是显著优越，不与native grid的0.768578直接相减。原193仍为用户固定对照，不自动改部署。本候选test已访问，训练STATUS的false仅训练阶段；无新outer/confirmation或当前test调优。语义组也已完成未提分；两组不自动重训、换epoch或重复测试。完整结果与核验在`archive/research_records/2026-09-21_segresnet_external_pretrain/`；旧“微调中”条目均为历史阶段。
- **第二候选已接续微调：** 1105022公开334例单卡预训练150轮/6300更新已完成，耗时约39分钟；固定final模型SHA `3726eac66918b7afd32ae93500a554534d27528bc86a0fe9dd2a626ce5ac6b76`与微调实际初始化来源一致。1104883双卡已自动进入原972微调200轮；单卡阶段结束，不重复预训练。真实微调进度见`workspace/BraTS/segresnet_t1ce_wt/two_ideas_20260921/finetune_pretrained/STATUS.json`。本候选尚未test，不把公开训练Dice当泛化分数。
- **语义分解已完成关闭：** 200/200轮、24400更新，验证选第158轮，Scan0.739689（原193为0.740032）。一次270/119同128³测试Scan0.757718、Subject0.743434、Point238/270，对193差−0.001461、患者配对95%CI[−0.006823,+0.003567]，未提分，不能称等效或确定有害。快照SHA `8fe36fdb6ca68e0593b0b3230876b2751ade91e5f4b9dfd9734446205d8e432c`，报告在`archive/research_records/2026-09-21_segresnet_semantic_regions/`，双榜及四校验通过。该组test已访问，不得沿用训练STATUS的false否认；无新outer/confirmation或test定向训练。本实现不自动重训/换epoch。第二组按既定方案继续；旧顺序队列的源准备failed不代表被新协调器接管的第二组失败，真实总状态看LIVE_STATUS。
- **最新硬件接续：** 用户新增1105022（1×A10040GB），CUDA/UVM实际检查通过。已冻结`HARDWARE_AMENDMENT_003.json`，公开334预训练改单卡直接batch8/150轮/每轮42更新，与1104883语义组并行；之后等语义监督结束回1104883按原双卡/global8微调972例200轮。单卡预训练RNG/浮点轨迹差异已披露，不声称逐位同轨迹。缓存跨节点三SHA一致；单卡合成峰值约20.16GiB。旧公开恢复等待器PID4852已停止，交接收据`OLD_EXTERNAL_QUEUE_STOPPED.json`；新监督`coordinate_pretrain_workers_20260921.py`分别运行single/dual，别重启旧等待器。真实进度看`SINGLE_PRETRAIN_STATUS.json`、`DUAL_FINETUNE_QUEUE_STATUS.json`和各run的STATUS；尚未动用test定向训练许可。
- 用户提供4张A100，允许最多开展两个SegResNet实验，要求认真选择大幅提分方向；不能保证必达某Dice。继续保持单T1ce和原972/108/270，未授权自动加入其他MRI序列或恢复关闭支线。
- 本轮选定公开334例监督预训练后微调、前景语义分解两候选，七问草案分别在`archive/research_records/2026-09-21_segresnet_external_pretrain/`及`archive/research_records/2026-09-21_segresnet_semantic_regions/`。尚无正式执行冻结或训练；不能把历史未执行的预训练四运行脚本直接启动。
- 用户明确允许必要时针对test处理训练集；此许可优先于下方旧的全面禁止test调优条款，但本轮尚未使用。实际动用前记录具体方案、预算和访问，结果标test-tuned并与传统排名分开，保留旧分数与原清单。不得把适配分数宣称独立泛化或隐瞒标签使用。
- worker1104813的4张A100可见，但缺nvidia_uvm、cuInit=999，实际CUDA计算失败；没有启动GPU训练或新患者推理。需要平台修复/更换节点后完成实现、合成检查及执行冻结。详见`archive/reviews/2026-09-21_segresnet_next_two_ideas/RESULT.md`。
- 用户随后更换为worker1104883（2×A100），实际CUDA计算通过。两候选改为顺序执行，各自两卡/全局8不变；新入口与6项CPU检查完成、36文件冻结，监督入口已启动。真实阶段以`archive/research_records/2026-09-21_segresnet_two_experiments/PIPELINE_STATUS.json`及阶段HEARTBEAT为准；先合成与数据门，再语义200轮、公开预训练150轮及972微调200轮。每组仅验证选定的一个模型允许一次270；无自动test调优或第三组。
- 缓存入口第一次因训练清单字段`label_path`误写成`seg_path`，在读取源标签前失败，尚无训练。已新增v2入口和`ENGINEERING_AMENDMENT_001.json`，保留旧源码/收据及空的失败缓存；恢复监督为`resume_segresnet_two_ideas_gate.py`，不要直接重启旧pipeline或因旧`PIPELINE_FAILURE.json`忽略更新的恢复状态。两组双卡合成检查通过，峰值每卡约10.28/10.11GiB。
- 公开334例已下载到worker1104883的`/tmp/aflow_two_ideas_20260921/raw`，不是HDFS。初次外部缓存因旧包装器只接受BraTS编号而全部失败；源像素/N4已访问，但无预训练模型前向。已冻结工程修订002，新增保留外部编号且数值逐位等价的入口，3项CPU测试通过。`repair_segresnet_external_pretrain_queue.py`负责重建并等旧监督器退出后独占接续A；不重启正在训练的语义组。以`EXTERNAL_REPAIR_STATUS.json`为修复状态，勿因旧failed文件自动重复下载/训练。
- 用户随后提供worker1105002并明确允许第二组并行，覆盖先前仅顺序执行的硬件安排；两候选配方及数量不变。该新节点也缺nvidia_uvm、cuInit999，虽有2张A100但不能计算，尚未迁移或启动。继续保留1104883工作；健康节点到位后须记录缓存哈希及旧恢复队列的独占交接，防止第二组在两处重复启动。收据见`archive/reviews/2026-09-21_worker1105002_cuda_preflight/RESULT.md`。
- 再次提供的worker1105007及1105011同样CUDA失败（缺nvidia_uvm、cuInit999），端点IP与1104813/1105002一致，未迁移/训练；须平台修复或不同宿主机健康节点。1104883继续运行，第二组并行授权保持有效。最新收据见`archive/reviews/2026-09-21_worker1105011_cuda_preflight/RESULT.md`。

## 2026-09-20 原始分辨率e173一次test270授权与节点更换（最新）

- 用户2026-09-20明确授权“先把原始分辨率的组去270个test上评测一下”。唯一候选：2026-09-18原始分辨率组验证选出的第173轮（SHA 57d61598…），记录在`archive/research_records/2026-09-20_segresnet_native_res_e173_test270/`，测试画布160×208×168在访问像素前由几何元数据冻结。对照为原193基线在同一原始栅格上的重采样分数；128³栅格的0.759179与本行不可直接相减。
- worker1103669（4×A100）主机缺nvidia_uvm、cuInit返回999，缓存构建完成但模型未接触测试像素即失败；用户重开worker1103728（2×A100）。构建脚本允许在新节点按同一冻结脚本重建缓存，要求图像/标签SHA与第一次收据一致（IDEA 7b与FREEZE amendment）。
- 一次性评测保护以MODEL_PREACCESS.json为界；完成后必须更新双榜、四文档校验，并在榜单行内标注“native grid”。无新增outer/confirmation许可；不得按结果调画布/阈值/epoch或重测。
- 分辨率方向据验证结果（+0.003422、CI跨0）视为已量化并关闭；后续候选方向为单T1ce公开术前数据预训练与nnU-Net参考基线，均需新预注册与用户授权。

## 2026-09-18 原始分辨率整脑SegResNet训练授权（已完成）

- 用户在改进方向讨论后选定第1项并要求“先帮我用一下1……把显卡资源利用率最大化”，随后追问每卡6/8例是否OOM并表示“不会的话就加大吧”。当前唯一新运行：原972/108/270、单T1ce、同N4/裁剪/标准化/初始化seed/损失/200轮计划的纯SegResNet，唯一研究变化为不缩放到128³，保持1mm原始裁剪并零填充到固定160×208×160画布；每卡batch由合成显存探测在8/6/4中取最大可装下者，记录在`archive/research_records/2026-09-18_segresnet_plain972_native_res/BATCH_DECISION.json`，不是研究因素。
- 验证在原始裁剪栅格上评分（排除填充）；对照为原N4纯193重采样到同一栅格的`BASELINE_E193_NATIVE_VALIDATION.json`，不得拿128³栅格的0.740032直接相减。协议、七问、冻结见该目录IDEA.md/FREEZE.json；真实状态看RESULT.json/HEARTBEAT.json。
- 本组没有test/outer/confirmation许可；不得根据验证结果调参、换seed或自动测试270。中断后守护器自动从latest续跑，最多4次，续跑次数记录。缓存放worker1100631的/tmp，模型与报告持久化在`workspace/BraTS/segresnet_t1ce_wt/plain972_native_res_20260918`。
- 无N4对照已按50轮规则停止，不因本组恢复。SAM权重已删除，不恢复。

## 2026-09-18 无N4组按50轮用户规则停止（最新终态）

- 已完成50/200轮、6100更新；50轮与旧N4的固定108验证比较未达到新增明确差异门，按用户要求节省算力停止。保留50轮/前50最佳/latest，训练与监督退出；没有新test。不可称等效或200轮完成，不自动恢复。收据：`archive/research_records/2026-09-18_segresnet_plain972_no_n4/STOP_RESULT.json`。

## 2026-09-18 无N4第50轮节省预算规则（用户最新指令）

- 用户要求观察到50轮对比旧N4，“如果发现没区别就不要训完”。必须执行`archive/research_records/2026-09-18_segresnet_plain972_no_n4/EPOCH50_REVIEW/AMENDMENT.json`的新规则，不能因旧200轮预注册无条件继续。
- 50轮已保存后暂停并比较：同轮绝对Scan差>=0.01、患者配对95%CI不跨0、41–50平均差同方向才继续原200轮；否则停止并保留50轮/当时best/latest。无明确差异不等于证明等效；这是看过早期验证后的自适应预算修改。
- 独立守护器负责停止/继续与结果归档，不修改冻结训练源码，不自动测试270，不按结论新开参数/seed。真实决定以该目录DECISION和STOP_RESULT为准。

## 2026-09-18 SegResNet单主线、清理与无N4训练授权（最新）

- 用户明确锚定单T1ce SegResNet分割，授权运行不含额外强度不均匀校正的基础对照。当前一组新训练为原972/108/270、同初始化/200轮/全局8/每轮122更新的no-extra-N4纯SegResNet；默认只去N4，保留脑区裁剪与标准化，不制造少训弱对照。新协议在`archive/research_records/2026-09-18_segresnet_plain972_no_n4/IDEA.md`，真实状态看该组RESULT/PIPELINE_STATUS。
- 当前0.759179仍是原N4纯193的既有复用test成绩；无N4对照未出来前不能称已证实创新增益，N4自身是已有方法。不得用测试选参数，新组没有270测试许可。
- 用户明确授权删除SAM-Med3D权重和当前不用的派生缓存。已删除1114文件、约37.956GiB逻辑大小；完整哈希收据见`archive/reviews/2026-09-18_segresnet_only_cleanup/RESULT.md`。旧SAM权重不可再恢复，之前“保留权重”是历史状态；报告、源码、原始数据和全部SegResNet权重保留。不得因旧预注册恢复已关闭SAM分支。
- 新训练缓存放worker1100631临时盘，模型与报告持久化项目目录。`sammed3d/`仍含SegResNet公共代码，不能因目录名整删。

## 2026-09-18 单T1ce输入约束（用户确认）

- 用户明确“保持单T1ce，与当前场景一致”。后续SegResNet训练和部署输入固定单T1ce，多MRI序列候选退出当前方向；不能自动加入FLAIR/T2或以改变输入条件取得的指标替代当前基线。
- 当前仅确认研究约束，不构成新训练或测试启动许可；原972/108/270与纯193对照继续固定。

## 2026-09-18 八镜像测试在执行前取消（最新用户决定）

- 用户随后明确“确实我也觉得不需要进入测试”。新建八镜像test270目录只有预注册和入口；测试未启动、无PREACCESS/模型推理/新增test像素访问，终态见该目录CANCELLED.json。禁止因准备好的FREEZE或此前询问自动启动。
- 当前只讨论SegResNet提升方向，没有新训练/测试授权；纯193仍为唯一基线。讨论记录见`archive/reviews/2026-09-18_segresnet_improvement_options/RESULT.md`。

## 2026-09-18 当前唯一比较基线（用户最新指定）

- 用户明确以后所有优化起点和对照均为本次原972纯SegResNet第193轮：验证108/50 Scan0.7400322696、Subject0.7315383841、Point97/108；复用test270/119 Scan0.7591787810、Subject0.7449846235、Point242/270，单次前向。
- 固定模型SHA为157fc5af363fc42b2c993df07639693ed8d9f193b8cadd11810a39f62c0f54a5，基线身份见`archive/research_records/2026-09-18_plain972_e193_flip8_validation/REFERENCE_BASELINE.json`。旧756/864、蒸馏或旧八镜像仅历史参考，不替换当前对照。
- 用户询问旧榜首算法迁移与提升；已完成一次固定八镜像108验证迁移诊断：Scan0.740032→0.741476，差值CI跨0，Point97→95，未通过冻结门；不继续test，不搜索其他视图/权重/阈值。配方和结果见该目录IDEA.md/RESULT.md。没有新训练/seed复制，没有新test/outer/confirmation许可，不能自动重跑270或搜索视图子集、权重、阈值。
- 本轮是既往测试揭示后的开发，不能称独立确认。效用门通过也不自动授权test；不通过记录负结果，不变相换旧对照。用户的速度偏好只决定是否采用8×部署，不把诊断当作已部署。

## 2026-09-17 SegResNet193测试已完成（最新终态）

- 用户明确授权“193最佳模型去测评”，已固定193轮快照完成原270扫描/119患者一次测试。Scan0.759179、Subject0.744985、Point242/270；当前自动总体及1×Dice最高点估计，但与旧8镜像差值CI跨0，Point非最高。
- 原972纯训练已完成200轮/24400更新，训练/测评均结束。当前没有可自动恢复的训练或可自动重复的测试；不得根据本结果调参或换其他epoch再测。
- 本组test访问已发生；训练阶段FINAL_RESULT.json的false标记是历史阶段范围，不得掩盖新独立测试访问。完整材料：`archive/research_records/2026-09-17_segresnet_plain972_e193_test270/RESULT.md`及PREACCESS.json。
- 旧SAM灰度及旧数据支线仍停止；固定972/108/270与患者隔离继续有效。新研究/评估需用户新明确指令及预注册。

## 2026-09-17 固定972纯SegResNet训练授权（当前最高）

- 用户已明确要求“用972 108 270再训练一下基线”，用于SAM-Med3D与SegResNet模型选型。当前唯一新运行是纯SegResNet基线，不恢复任何历史SAM灰度或756/864旧运行。
- 固定原972扫描/444患者训练、108/50验证、270/119测试及三份SHA和患者隔离。每轮972各一次，不有放回采样，不丢尾批；两A100每卡4例、全局8，每轮122次更新。这是SegResNet原batch8配方，不能误继承SAM的每轮2次更新。
- 纯470万参数SegResNet随机初始化；无蒸馏、灰度项、额外中心头、EMA/融合/TTA/后处理。目标200轮，前100 lr2e-4，后100按冻结余弦至1e-5，逐轮固定108验证选Scan Dice最佳。
- 新组协议与访问边界以`archive/research_records/2026-09-17_segresnet_plain972_baseline/IDEA.md`和FREEZE.json为准。本组没有自动test/outer/confirmation许可，旧测试揭示历史保留。
- 两次SAM灰度组继续停止。新SegResNet全自动指标叫scan_dice，不叫scan_final5/best5；与SAM的提示权限和输入栅格差异必须披露，不能称纯架构因果对照。

## 2026-09-17方向返回SegResNet（最新研究方向）

- 用户要求“先回到之前训练segresnet上吧，对比segresnet在我的场景下的baseline大概是多少”。当前方向改为SegResNet基线核对；这不等于自动恢复旧训练或新增测试许可。
- 历史纯模型756训练的test270为0.725820（99轮选中）及0.741204（168轮）；972蒸馏0.751760和8镜像0.757270不是原始纯baseline。不得把旧756/864指标改写为当前972纯模型。
- 未来若用户进一步要求训练，继续严格固定972/108/270及患者隔离、每轮完整972，不恢复撤销的旧划分/旧运行。两次SAM灰度训练停止状态和历史访问保留。
- 核对材料：`archive/reviews/2026-09-17_segresnet_baseline_return/RESULT.md`。当前没有运行中的训练或评测。

## 2026-09-17 Mumford–Shah组停止（最新指令，优先于下方训练目标）

- 用户根据验证平台期明确要求结束Mumford–Shah新组。实际完成45/200轮、90次更新，第46轮中途停止，训练和监督已退出。禁止因旧200轮授权、预注册或历史运行日志自动恢复。
- 按预设best5选择的最佳为第25轮，独立快照已保存；滚动latest为第45轮。第35轮仅有验证报告，没有独立权重，不能冒充其他epoch。
- 本组未评估test270。验证best5=0.571493略高于历史0.571306，原点估计门数值通过但无明确增益证据；不能标为200轮完成，也不能称一般方法被否定。最终收据见`archive/research_records/2026-09-17_sammed3d_mumford_shah_single972/STOP_RESULT.json`。
- 旧均值匹配组仍停止161/200。当前没有在运行的训练/评测，禁止新增对照、参数搜索或更多测试；未来运行需用户新的明确指令。

## 2026-09-17新方法训练授权（当前优先）

- 用户在旧灰度组停止后，明确要求“替换进去然后开始训练”，并选择Mumford–Shah区域灰度能量；又明确选择恢复每轮2次参数更新。
- 当前只允许一组新SAM-Med3D训练，从同一官方Turbo初始化，固定972/108/270与患者隔离。双A100每卡微批4，全局微批8；通过累积分为504/468两次样本加权更新，每轮972各一次。目标200完整轮；旧灰度50/139/161权重不作为起点。
- 新能量采用作者区域残差+边界平滑的明确三维适配。完整设置以`archive/research_records/2026-09-17_sammed3d_mumford_shah_single972/IDEA.md`及FREEZE.json为准；不能称师兄原版或与旧组单因素对照。
- 方法选择发生在既往test揭示后，须标记test-informed探索；新组只用固定108验证逐轮best5选优，没有新test/outer/confirmation许可，不扫loss权重、seed、阈值，不额外增加组。
- 下方“旧组停止161/200”继续约束旧均值匹配实验，不能误用于阻止本次已明确授权的新组，也不能反过来恢复旧组。

## 2026-09-17用户主动停止（优先于下方200轮目标）

- 用户已明确停止当前SAM-Med3D灰度单组训练；实际完成161/200轮，第162轮中途停止，训练及监督进程均已结束。禁止因下方旧200轮目标、预注册或旧“运行中”自动恢复。
- 保留第50轮最佳、139轮恢复备份和161轮最新检查点。第140轮只有验证报告，没有独立权重，不得用其他轮次冒充。
- 第50轮已在复用内部270扫描/119患者测试，五点击最佳Scan0.566424，低于旧0.594226。用户随后明确授权单独测第161轮最新模型；只做此固定候选诊断，不恢复训练、不搜索其他epoch/阈值/参数。
- 本次提前停止与第161轮补充评估发生在第50轮test揭示后，须标记test-informed停止/post-test诊断；不能称预注册早停或独立盲测。未来训练恢复需要用户新的明确指令。
- 真实停止收据见`archive/research_records/2026-09-17_sammed3d_resume139/STOP_RESULT.json`；第161轮测试材料放`archive/research_records/2026-09-17_sammed3d_graymean_e161_test270/`。

## 固定数据与唯一训练主线（2026-09-16用户纠正，优先于历史记录）

- 以后训练集、验证集、测试集固定为原始 **972扫描／444患者、108扫描／50患者、270扫描／119患者**；case指扫描，不等同于患者。三份清单位于 `data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/`，依次为 `train_t1ce_wt.jsonl`、`validation_t1ce_wt.jsonl`、`sealed_test/test_cases_t1ce.jsonl`。
- SHA-256依次固定为 `6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65`、`f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb`、`a5ddc0ed2c0b269431758b14061da3c6163dd0c48133b60a2ef8efd7f23aa8f9`。每次新训练核对清单哈希、数量及患者隔离；不得偷偷减少病例、另划开发集、排除旧确认病例，或因旧协议沿用684／180支线。只有用户明确改变划分才可修改这些约束。
- 当前唯一训练主线为 **SAM-Med3D加师兄灰度校准项，单组训练200个epoch**。论文对应区域灰度均值约束；具体实现与未披露细节写入当次预注册。不得自行增加对照组、配对实验、消融组、seed复制或参数搜索。不得把两张A100理解成两组实验；并行方式以当次冻结协议为准，不擅自改变batch或训练轨迹。
- 同日用户追加授权：同一灰度实验使用两张A100并增大batch。当前硬件试跑选择每卡4例、全局8例；这不是增加实验组。每轮仍恰好972例，不丢尾批/不重复填充，总计200轮；增大batch改变更新次数，须记录与论文batch1的差异，不能称全部参数完全对齐。后续batch变更仍需明确记录，不从验证/测试分数选择。
- 6轮最小验证不等于200轮训练完成；报告必须写真实完成轮数。工程中断后，仅按当前单组冻结协议恢复；不得恢复已撤销的旧配对运行或SegResNet训练。
- 684／180及其派生864支线已由用户撤销；其历史指标、访问记录和测试排行榜仍按真实历史保留，不能改写成972训练，也不能当当前训练入口。清理边界见 `archive/reviews/2026-09-16_remove_684_branch/RECEIPT.md`。
- 108验证集用于本次冻结协议允许的验证与检查点选择；270测试仍为复用内部测试，禁止用其调参。旧训练池内“开发/确认”身份不再构成本轮新划分，但既往访问与重用事实不消失；不得称全链路全新盲测。新训练仍须先完成七问预注册、固定评估和访问收据。

## Explain research for a beginner

Assume the user is new to medical image segmentation. Never introduce a method only by acronym, paper, loss, or module name.

Before implementing or launching every new method, first explain in plain Chinese:

1. **现在的问题是什么？** Use current project numbers.
2. **一句话直觉是什么？** Use an everyday analogy when helpful.
3. **为什么现在选它？** Connect the idea to an observed failure/error pattern and primary source; explain why prior negatives do not already disprove it.
4. **具体怎么做？** Explain input→model/teacher→loss/fusion→output in words before formulas.
5. **只改变什么，哪些不变？** Name the single factor and controls.
6. **怎样算成功/失败？** Freeze promotion and stop gates.
7. **代价是什么？** Training time, GPU, disk, inference forwards, deployment complexity.

Expand EMA, KD, TTA, ROI, KL, CE, SAM, etc. on first use. A formula may follow the plain explanation, never replace it.

Write the same explanation before formal execution in the preregistration or `IDEA.md`, using `EXPERIMENT_IDEA_TEMPLATE.md`.

## Canonical method catalog

`AFLOW_BRATS_METHOD_CATALOG.md` is the beginner-readable idea history. Whenever a method is proposed, starts, completes, stops, or changes status, update its entry in the same task.

Each completed card must include status, one-sentence idea, starting problem, procedure, changed/fixed factors, expected mechanism, observed metrics, whether intuition was supported, cost, next decision, and evidence links. Preserve negative ideas and distinguish “this implementation failed” from “the general idea is impossible.”

## Root documentation layout

Do not create new top-level handoff continuations, idea notes, plans, or ad-hoc result Markdown files.

The only allowed root Markdown files are:

- `README.md`
- `AGENTS.md`
- `AFLOW_BRATS_HANDOFF_CURRENT.md`
- `AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md` (historical compatibility)
- `AFLOW_BRATS_LEADERBOARD.md`
- `AFLOW_BRATS_METHOD_CATALOG.md`
- `AFLOW_BRATS_RESULTS.md` (active chronological research log)
- `AFLOW_BRATS_WEEKLY_REPORT.md` (mentor-facing research report)
- `EXPERIMENT_IDEA_TEMPLATE.md`

New experiment preregistrations/results belong under `archive/research_records/<date>_<method>/`. Staged handoffs, literature notes, or obsolete plans belong under a dated `archive/` directory. Update `AFLOW_BRATS_HANDOFF_CURRENT.md` instead of creating another root continuation.

After moving/adding root documents, run:

```bash
PYTHONDONTWRITEBYTECODE=1 ../sammed3d-venv/bin/python scripts/verify_aflow_brats_doc_layout.py
```

If it fails, documentation cleanup is incomplete.

## Research workflow

1. Read `AFLOW_BRATS_HANDOFF_CURRENT.md`, leaderboard, method catalog, and relevant archived preregistration.
2. Explain the idea with the seven beginner questions before implementation/training.
3. Freeze the plain-language idea/preregistration before formal access.
4. Keep train, validation, outer, confirmation, and test roles explicit. Never call reused test pristine blind or external.
5. Freeze data/checkpoint hashes, grid, threshold, TTA/postprocessing/SAM, output, metrics, bootstrap, selection, and stop gates.
6. Add entry points instead of modifying hash-bound historical scripts. Run focused synthetic tests and proportional GPU smoke.
7. Do not use test to choose loss, weight, epoch, threshold, resolution, checkpoint, ensemble, TTA, or postprocessing unless the user explicitly changes the objective and evidence is labeled test-tuned.
8. Record negatives and implementation failures, including whether checkpoint/validation/test access happened before failure.
9. Check disk before multi-GiB caches. Do not silently delete material data. With authorization, remove raw-rebuildable/hash-archived derived caches while preserving code/checkpoints/reports.
10. After results, explain what happened, whether intuition was supported, why, and what is ruled in/out.

## Mandatory leaderboard update after test

Canonical files:

- `AFLOW_BRATS_LEADERBOARD.md`
- `AFLOW_BRATS_LEADERBOARD.json`

Whenever any method finishes a named test cohort, update both before completion. This includes positive, tied, negative, failed-gate, single, ensemble, oracle, and interactive methods.

Record method/identifier, cohort counts, Scan/Subject Dice, Point hit, checkpoint/ensemble formula, resolution/threshold, forward cost, evidence status, report path/SHA, and reused/test-tuned/external status.

Rules:

1. Never mix validation/outer/confirmation/test in one ranking.
2. Never mix automatic direct locators with GT-centered/oracle/interactive methods in the primary ranking.
3. Sort comparable JSON lists by descending Scan Dice with contiguous ranks; Markdown must match.
4. Label grid/rasterization caveats.
5. Report confidence intervals without turning point-estimate leads into significance.
6. Preserve prior rows and correct provenance explicitly.
7. Update `updated_utc` and validate.

Required command:

```bash
PYTHONDONTWRITEBYTECODE=1 ../sammed3d-venv/bin/python scripts/verify_aflow_brats_leaderboard.py
```

If it fails, the research task is not complete.

## Mandatory result archival

After formal validation/test:

1. Save raw report and pre-access/selection receipt.
2. Recompute checkpoint/report/receipt/cache-index SHA-256.
3. Write `archive/research_records/<date>_<method>/RESULT.md` plus machine JSON.
4. Update `AFLOW_BRATS_HANDOFF_CURRENT.md` with outcome, metrics, access flags, failure boundary, and next decision.
5. Update `AFLOW_BRATS_METHOD_CATALOG.md` with plain result/interpretation.
6. If test ran, update both leaderboard files.
7. Run the method-catalog, leaderboard, and root-layout validators as applicable.

## Final response checklist

State in plain Chinese: idea and motivation; changed/fixed factors; best overall and single-model comparable results; cohort/metric/CI scope; training/inference/storage tradeoff; whether intuition was supported; test/outer/confirmation access; clickable current handoff, catalog, leaderboard, report, archive, and checkpoint.

Do not call a method “best” without naming cohort and overall-versus-single scope.


## 导师可读文档维护（2026-09-14用户要求）

根文档分工固定，不再将同一张长卡片复制到多个文件：

- `AFLOW_BRATS_WEEKLY_REPORT.md`是导师汇报：研究起点、核心优化、最终Dice、完成度与局限；重要结论改变时更新，通常控制在150行内。
- `AFLOW_BRATS_RESULTS.md`是从2026-08-13起的连续日期流水账。每次工作追加/更新对应日期，用“问题→做法→结果→决定”说明；明确仅文档或工程工作是否没有新精度结果。不得再让日期入口停在旧阶段。
- `AFLOW_BRATS_METHOD_CATALOG.md`保留通俗方法卡及最终状态：保留六个必要字段，短句说明直觉、唯一改动、固定项、结果、成本和下一步，链接标签用“核对材料”等可读名称。不直接显示长目录、哈希、PID、启动命令或多份收据。逐项细节可进入资料层，但负结果必须留有索引。
- `AFLOW_BRATS_HANDOFF_CURRENT.md`只写当前状态、数据角色、真实待办及禁止恢复的已关闭分支，通常控制在100行内；历史“运行中”不得累积在正文。
- `AFLOW_BRATS_LEADERBOARD.md/.json`保持精确分队列排名和证据约束，不为精简汇报删除旧排名。

新方法的七问、预注册、过程日志、完整参数、路径、哈希和raw结果仍放对应实验归档；正式结果需同步日期流水账和方法卡。禁止继续调用旧`publish_research_card.py`把完整技术卡自动前插到handoff和方法目录；它保留作历史重现，不再作为新文档发布入口。

历史完整根文档与来源索引见`archive/reviews/2026-09-14_mentor_facing_cleanup/EVIDENCE.md`。当前汇报不应把历史“当时纪录”写成“当前最好”；内部、开发、交互、外部及不同栅格必须明确区分。文档清理不改变数据访问历史或赋予新测试许可。


## 根目录技术材料约束（2026-09-14）

根目录JSON仅允许`AFLOW_BRATS_LEADERBOARD.json`，其他实验候选、清理收据、哈希清单必须放入相应archive目录；不得新建根目录临时JSON或SHA文件。历史根交接仅保留不超过20行的兼容导航页，全文放archive。

旧候选路径迁移见`archive/legacy_root_artifacts_20260914/README.md`。冻结脚本原文不改；旧默认根路径已失效，历史复现需显式提供归档候选和预注册参数，不得自动恢复关闭的实验。

文档整理或新增技术材料后，除原三个校验外，运行`PYTHONDONTWRITEBYTECODE=1 ../sammed3d-venv/bin/python scripts/verify_aflow_root_artifacts.py`，防止根目录重新堆积历史JSON和长交接。
