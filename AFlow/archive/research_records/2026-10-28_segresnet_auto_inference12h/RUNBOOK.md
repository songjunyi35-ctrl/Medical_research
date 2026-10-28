# 自动研究接续说明（滚动更新）

本goal仍active，用户要求至少10多个小时，冻结研究窗口2026-10-28 20:41:16→2026-10-29 08:41:16；不能因首批有/无收益提前结题。固定单T1ce/w32第171轮，不训练权重；允许直接按270反馈全局选择，所有本批test-tuned，无GT单例路由或outer/confirmation。

- 原worker1152682，IPv6 fdbd:dc61:c:649::16 port9942 tiger，21:08从活动清单消失且SSH关闭。最后GPU job0003_axes120到160/270、分析81003配置到12/270；已完成render3族，但临时输出目前不可达，不假设丢失或仍在。
- 用户提供新worker1153287双A10080GB，IPv6 fdbd:dc61:1a:495::28 port10907。自建排队1153242/1153273都已取消；H100仅配额失败无分配。新节点270缓存三项SHA匹配，旧五镜像配方270掩膜/点/分数精确复现。RESOURCE_CURRENT token=r1_worker1153287。
- 原worker1152470双A10040正在独立RACE训练；必须不干预。其root持久RACE状态可只读跟踪。只有其200轮/归档/进程退出且GPU确认为空闲后才考虑复用空闲卡，不提前抢显存/算力。
- 12h根目录：archive/research_records/2026-10-28_segresnet_auto_inference12h；数值脚本scripts/segresnet_auto_inference12h_20261028/。AUTHORIZATION记录持久<=2G、余量>=20G，新增scratch<=300G、余量>=70G，仅本轮已归档无读者中间数组允许滚动清理。
- 初版GPU队列queue/*.json，gpu_worker.py与render_v1.py源码SHA已冻结；分析队列analysis_queue/*.json、analysis_worker.py、core_fusions_v1.py也冻结。不要改这些已绑定旧源码！新worker恢复使用worker_v2.py及新queue_v2/analysis_queue_v2，RESOURCE_CURRENT.json租约分隔资源。worker_v2仅改控制/元数据，不修改数值render_v1；旧队列需移到归档目录，以防旧worker复活继续接新任务。
- control_v1.py为迁移后的完整270同机旧五镜像方案对照；先full8/tail6合成门，再重新生成旧配方270结果，记录与历史0.780995的差异。不把硬件数值变化当新收益。
- 核心8镜像270F32输出位于worker/tmp/aflow_auto_inference12h_20261028/groups/0000_core_mirror8，但当前旧worker不可访问。新资源若没有则按新jobID重建，不跳过DONE却假设payload存在。
- 核心融合v1有2077×39=81003配置，所有子集/视图内温度/均值幂均值等，先逐case严格复现旧掩膜。旧v1直方图bincount在背景集中时慢；histograms_v1.py有分尾直方图精确加速，9组CPU一致性过，GPU性能门还未完成，旧GPU测试因SSH关闭未执行。下一版core_fusions需新文件和注册，不改旧文件。
- calibration_v1.py实现每入围配方4125个温度/偏置/beta/体积倍率规则；CPU纯合成门已过（确定性、直接mask计数、GT置换不影响决策）。ROUND002已预登记，尚未在真实概率直方图上跑。
- CPU fallback已完：179配置，选定min_component1000，使270 Scan0.781651665、Subject0.769047724、Point255。对0.780995差0.000656467，95%区间[-0.000520097,0.002434625]，跨0。candidates/0001_cpu_binary全270掩膜/RAW/RECIPE/RESULT均独立核验。INCUMBENT.json是中间test-tuned搜索最高，不是整个研究结束。旧部署配方不覆盖。
- CPU输入标签/index只读复制到主节点/tmp/aflow_auto_inference12h_20261028/cpu_native，SHA与旧test严格一致。CPU源码cpu_binary_v1.py+ROUND003+FREEZE保留；cpu_binary_fast_v1.py在270组合成几何测试完全相同，box closing probe约3.63×，只准备给未来任务，不重跑已完179配置。
- 新完整候选完成后需把它作为test-tuned行更新原生双榜，保留所有旧行，所有负配置在各轮完整RANKING/矩阵及SHA。最终模型/推理/掩膜独立重算、文档校验、归档、push到Medical_research；不上传影像/权重/预测数组。旧最近推送a4de957，RACE另有独立归档watcher，后续push须先fetch最新main。

## 22:27双卡真实接续

新producer PID316使用物理GPU0，analysis PID317使用GPU1，CPU校准PID318，monitor PID636；日志位于新worker/tmp/aflow_auto_inference12h_20261028/。worker_v2源码已在新队列冻结，勿再编辑。生产队列queue_v2、分析analysis_queue_v2、CPU校准cpu_queue_v2；旧worker即使恢复也不能接新队列。新core8已完整生成，13组轴置换/填充相位继续。core_fusions_v3执行81003配置；v2在执行前撤回，原任务JSON保留withdrawn_before_execution。Triton整数hist15组合成核验通过并有真实首例精确计数门；首次编译时间不当吞吐。校准每入围配方4125组。group_fusions_v1对每变换组40融合×39阈值，再入围4配方做校准。

CPU扩展244种连通域规则已完，仍为min1000最高，未新增收益（包括6/18/26连接、更大阈值、保护最大域）；完整负结果analyses/1100_component_sweep。旧179配置胜者已经登记原生双榜（32行），全部仍test-tuned。当前不是结题。RankSEG0.0.8只下载解压到主机/tmp/aflow_rankseg_vendor_20261028，不改变共享PyTorch/Triton环境；后续可对RMA/BA等官方解码作单独门与实验。

## 22:53当前接续关键点

worker_v2的预算遍历在并发原子JSON重命名时发生FileNotFound，CPU校准第一次停止，已计工程恢复2/6。自有316/317/318进程组受控停止，原状态保留amendments/001_guard_concurrent_rename；新worker_v3+guard_v2只修控制不改数值代码，新PID producer3215/GPU0、analysis3216/GPU1、cpu3217；PID仍写WORKER_V2_*_PID文件。先前数值源码/旧失败均保留，不能改已绑定sourceSHA的文件。guard修复2000次并发replace/遍历门过。所有未完任务追加新控制源码SHA，数值冻结不变；旧FAILED改名FAILED_STAGE_GUARD_V2后从已保存批次/病例/CPU输出继续。抢占可能重算未提交批次，计算成本需加旧状态范围，不能声称一次访问。

81003配置完整阈值搜索完成，最好confidence_weight0.25、阈值0.275，270Scan0.781091323/Subject0.768660223/Point259，低于当前连通域0.781651665。固定校准115500配置正在CPU，约6300/7560直方图case；局部MRF720×39正在GPU1约111/270。新增任务已冻结：官方RankSEG BA/RMA（隔离/tmp包0.0.8，源码SHA核对）、adaptive_calibration_v1（按预测体积/不确定性同一函数、1080规则/配方）、6组ROI预测框局部放大。ROI来源原预测mask非GT，270/230例真缩放，全部图像/坐标/显存门过，区域外保原预测；group_fusions_v3对ROI避免重复加基础5次成本。

promote_v1.py已经实现独立标量切点vs批量网格10探针对齐，等实际校准结果完成即可在新worker GPU1轻量独立重建完整270掩膜并与搜索矩阵精确对照：python -m scripts.segresnet_auto_inference12h_20261028.promote_v1 --analysis r1_1000_calibration --candidate 0002_core_calibration。该审计新增神经前向0，CUDA_VISIBLE_DEVICES=1、memoryfraction0.2；源任务SHA核验。成功才更新INCUMBENT；后续要登记双榜。若新配置负结果，无需制造新incumbent。

consensus_v1.py未冻结临床任务（GPU合成门正在核验）：硬STAPLE、fractional-vote EM（明确不是PSTAPLE复现）、按预测一致性加权/筛选视图，118配方×39阈值；同模型八视图强相关，不能假定独立专家，Point固定mean8概率argmax。后续若promote此kind需要先扩展promote_v1或另写v2的probability_for与point_rule；当前promote_v1只支持core/group/localMRF/officialRankSEG/固定或自适应校准。

## 23:00中间候选与下一步

固定校准115500配置完成最高0.7830104277641685/Point260，七镜像原logit/T4后概率均值，再postT1/bias-1/beta1.15/volume_scale1。promote_v1全270掩膜独立重建与矩阵精确一致，candidates/0002_core_calibration；已入双榜33行。其paired helper旧bootstrap prose“two planned comparisons”不适用当前大搜索，已在amendments/002_bootstrap_metadata保存原件后仅改说明为95%描述性、97.5%敏感性无多重校正，所有数值不改且双榜resultSHA更新。后续请用promote_v2（与v1数值相同、已包括consensus支持，bootstrap说明正确）；v1已绑定候选源SHA不能再改。

其上component_sweep_v2的244规则最高c1_min750保护False，使0.7837921336293221/Subject0.7706401502739648/Point260，promote_components_v1已独立270掩膜精确核对，candidates/0003_core_calibration_cc，新INCUMBENT，已入双榜34行。对研究前五镜像delta0.0027969354，不是新增独立确认。adaptive_calibration6480配置最高0.782931551未超过固定校准，负结果归档，若后续考虑交互，可审计其掩膜再相同244后处理（当前尚未做）。

新STAPLE/预测一致性118×39=4602已在ROUND010冻结，GPU1后续跑；其前8再校准4125。GPU0继续填充相位、3组MC16、6组ROI。后续新增大输出前必须检查scratch预算，初始22变换族+直方图/缓存预计约285GiB，接近300；待某变换族及其校准完成且非当前候选来源、无读者后，记录该族DONE文件SHA/完整分数仍保留再滚动清理本轮/tmp该族输出。绝不删除core8 atoms（所有当前候选与后续MRF/RankSEG等依赖），不删旧缓存/权重/预测。候选最新原始掩膜都在持久R/candidates，仍完整。

## 00:20之后新增关键接续（研究仍active，必须到08:41至少12h）

当前最好已审计0006_best_holes：Scan0.7838496662400387/Subject0.7706811142339259/Point260，7视图；链条0002_core_calibration(7镜像/T4,bias-1,beta1.15)→0003_core_calibration_cc(6连接>=750)→0006(6连接填<=343封闭孔)。所有270mask与搜索矩阵精确。已登记原生双榜37行，0004_fine_calibration裸0.783195、0005叠26连通750为0.783786881均保留，未刷新最高。原阈值flip0新资源从cache导出控制0.7732280998，与旧2A10040基线mask差268体素，不是新模型或收益；避免假定旧保存原mask逐位相同，新的union/distance后处理用control_from_core_flip0。203新后处理最高填孔343，已审计0006。

新并行：GPU0初始22组全DONE（最后r1_0035因inline预算遍历遇清理竞态在患者前失败，计恢复3/6；roi_render_v2仅换stable_size后接续成功，原source/failure保留amendments/003）。GPU0现在analysis_aux_worker_v1 PID6426在analysis_queue_aux_v1跑b1_0000_nonlinear_fusions，37544×39=1464216配置，cached部分公式门过；core8 atoms保留。对应CPU b1_1000_nonlinear_calibration用calibration_parallel_extended_v1、6375网格、16CPU，sourcehash已冻结。大TSV每20万行一片，score矩阵~1.5GiB仅worker/tmp。promote_v5支持aux队列/新calibration和phase/group动态prepare，并支持--recipe-index/--parameter-index全局固定beam审计；旧promote_v1/v3已绑定实际审计源码不可改。

GPU1额外生产gpu_aux_v1 PID5084/queue_aux_v1，noise64 sigma.05已全部270完成，只有mean_probability1map，约3.4GiB，后续20配方/校准队列r1_0140/r1_1140；正常GPU1 analysis_worker_v3 PID3216先高T扩展79560(已完)→consensus118×39(已完)→22变换组40×39，CPU正常PID3217在高T6375校准178500配置。GPU0普通producer PID3215仍空闲监听，但无新任务不会抢NL评分显存。

已冻结a2_0000_polyphase512（gpu_aux队列，等jobs/scratch_release_060/DONE）：512=8³独立phase，各轴+16padding，flip0固定、无插值/无丢脑，1/64/128/256/512共享前缀，全局NN总512×270，不把前缀相加。gate512坐标、38.65GiB完整batch、状态NPZ保存恢复与完整F32顺序累加逐位一致均过，PHASE512_SYNTHETIC_GATE.json。各阶段65融合×39，前8×6375/16CPU，a2_01xx/a2_10xx。phase_fusions_v1同时与旧5/T1或新7/T4基础概率混合，成本N或N+5/7。所有全局规则直接test-tuned，没有GT单例路由。新长实验预计几小时，目标窗口内完成，不能因已有提升提前结题。

滚动清理：rolling_cleanup_v1 PID8077，只释放所有消费者完整评分/校准后落后incumbent>0.002的**新**groups npy；完整SHA再次核验、无fd/mmap读者、所有已审计候选源和core atoms保护。只unlink文件，保留空目录骨架，避免Path.rglob scandir父目录消失竞态；不要再并发rmtree！所有score/cut/hist/RAW/mask/模型/原生数据不删。已删旧core派生直方图3.692GiB（其3个消费者已全DONE，FILES SHA留存；重新校准需先从core8重建hist），另cleanup_unused_core_maps_v1准备/执行删除无消费者三张core聚合图10.217GiB（不是atoms，可从8原logits零NN前向重建）。source记录cleanup_records。marker监督PID8706等总释放>=60GiB写jobs/scratch_release_060放行phase父任务。MAIN_SCRATCH_BUDGET记录主机CPU标签1.406GiB；NL/phase在临界申请前累计workerTMP+nativeCACHE+此主机大小，300GiB含余量。其他扩展先等清理，切勿盲增大map队列。当前2G持久上限仍有大量余量，>20G空闲。

工具一次SSH inline合成探针自动审批**超时**，改落盘模块重试后已完成，不是安全拒绝/未解阻塞，记TOOL_INCIDENT_001；无需用户输入。全部后续SSH附ConnectTimeout20。新worker IPv6/port/knownhosts同前；RACE1152470继续不动。publishing仍需到Medical_research独立/tmp checkout，root AFlow .git不要add/push。可以先推中间已审计提升与running状态，但最终仍须收尾归档/推送。

## 02:20实时接续（本任务仍active，08:41前不提前结题）

当前INCUMBENT已变0008_consensus_cc：Scan0.7841391463522558/Subject0.7711561284416697/Point258，8次单权重前向。它是agreement_t0.2_logit_w16.0＋postT4/bias-1/beta0.7/scale1＋6连接>=512；不是STAPLE胜者。7次替代0006_best_holes仍0.783849666/Point260，指标取舍保留。新独立deploy_core_v1 API不接收GT/病例ID，PORTABLE_RECIPE_INTERIM.json无realized-cut查表；fresh完整270重新8神经前向已通过，全部mask差0/point相同/逐例Dice差0，约249.8秒并行环境，见live_verification/0008_consensus_cc/DONE。源码/配置可在原研究截止后独立调用，不依赖cohort manifest。当前API只支持核心视图，不支持group_mix；后续如果phase/group成为最高要扩对应预测器。

中间推送完成并远程核实main a6087a7cd3e442d7c7ef7cf177e249c616f84bc2，Medical_research独立/tmp checkout；约111MB/7734新文本文件，manifest17589项/8新行2160逐例计数和SHA/本批新链接通过。发布历史快照原有63断链/旧catalog校验title不一致，已证明HEAD基线相同，无新增断链；不要为此乱同步其他历史或上传模型。PUBLICATION_INTERIM_RECEIPT记录。最终还需再推。注意相位4个pending评分已移GPU0aux队列；下一次stager必须移除发布checkout旧active_queue路径并从manifest删对应条目（migrations/001有旧件），不能复制后留下重复队列。

GPU0大组合37544×39已完整DONE b1_0000_nonlinear_fusions，入围190，再1211250校准已于02:07全DONE，裸最高0.7831264938/Point260（weighted_logit T4/W8/e-1/subset251）。当前0012已用promote_v7独立270mask通过，待连通域组合。PROMOTE最新版v7支持3种新fast-calibration handler、缓存guard，不要再用旧v6去读已迁移handler。

CPU性能迁移成功：旧父进程报告1750时实际已有17331个已提交case；每task完整盘点R耗0.704秒拖慢。只CPU进程组3217受控停，GPU没停，旧件amendments/004保留。新CPU worker_v4 PID9770，case数学fast_volume_cuts_v1：去零支撑/保持同float64前缀、单峰二分＋原浮点局部argmax、原体积倍率映射；45真实样本329625切点和同worker旧输出63750切点逐位一致，数学20.69→2.55秒。guard_cached_v1每次deadline、0.5s磁盘、5s全盘点，近512MiB上限每次扫；CSV每20万行分片强制盘点，同2GiB/余量不变。为保守把计划性能重启也计恢复，现在4/6。新CPU24个未完任务source添加worker4/fast/helper SHA；已完成输出全部复用。三种新handler calibration_fast_standard_v1 /fast_extended/fast_consistency；原所有数值文件不改。b1未完剩余34k数学case从重启到全DONE约6分钟，而非旧估十小时。

GPU0原producer PID3215：NL结束后两组小尺度r1_0040/0041已跑，再r1_0050_polyphase512_flip7正在模型推理，phase512_v2只参数化flip7及父队列。GPU1 aux PID5084原flip0 a2_0000继续。两个方向各512×270，不是多权重。1/64/128/256/512共享前缀；flip7 gate512往返/原子状态位相等/38.87GiB过。新120GiB存储门已满足，总清理>132GiB，旧数据/模型/原core8 logits和全部评分保留。只unlink新中间文件留目录骨架，别rmtree造成另一次stat竞态。GPU0 analysis_aux PID6426现在处理未来phase/resize评分，正常GPU1 analysis PID3216在r1_0200_case_consistency_extension（262080配置，全新Point规则=p融合argmax）。后续CPU11475校准用fast_consistency。不要修改这些已绑定SHA模块。

四个纯缓存裸候选已审计：0009_adaptive0.78293155、0010_extended0.78309334、0011_official_rankseg0.78262843、0012_nonlinear0.78312649；main CPU正并行component_sweep_v3，环境AFLOW_POST_CPU_WORKERS=2每组，共8CPU，源矩阵1104_adaptive_components /1105_extended_components /1106_rankseg_components /1107_nonlinear_components。session28408/7899/12015/10278待取结果；若刷新或接近最高再promote_components_v1审计。需要登记这些已完成裸候选时用register_candidate_v2（保持latest研究元数据指向最高，v1误把最后注册较低值当latest但行排序无误）。

新增真实3D DenseCRF方向在准备，不是旧local26邻域近似：MONAI1.5.2没有_C扩展，pydensecrf2 1.1只build到main/tmp/aflow_densecrf_vendor_20261029并tar到worker TMP/densecrf_vendor，共享env未升级、无额外模型权重。wheelSHA e63f...有DENSE_CRF_VENDOR_PROVENANCE。dense_crf_v1.py原生3D通用DenseCRF(N,2)、XYZ/I float32特征；小图无边项恒等、重复逐位、整160208168强度白噪声10轮约10.44秒/RSS1.39GiB Gate过。prepare_dense_crf_inputs_v1正在GPU0从已存8原logits生成当前校准q图1map/例和mean8 point，0新神经前向/GT输入；ready后可CPU6进程做12双边参数×5/10轮+无边/仅高斯控制共27配置、全native不插值。预计最大24图×270×10.44s/6约1.6小时，实际nativecrop更小；预算足够才冻结。须先写真实270/GT仅评分的CPU handler、直方图/完整矩阵/后校准。当前还没跑CRF患者结果。

下一种值得检验的组合（尚未实现）：从b1校准按全270前16 pixel配方，乘自身预测一致性rater因子exp(lambda*(s_i-maxs))（lambda含负/0/正、tau0.1/0.2/0.3）；lambda0必须逐值复现原NL。全局beam不是GT逐例路由；若lambda!=0要先跑8views求一致性，成本8，lambda0可用pixel子集成本。该joint分支还未注册、不要声称完成。

## 03:10之后：新增分支/隐私/后续审计接口

两卡512继续，flip0已有1/64/128，flip7已有1/64/128；前缀融合裸最高约0.7829，未见超过低成本8视图，完整512仍按预登记跑。跨方向e1_*已注册analysis_queue_aux_v1，333/330配方×39＋原base7/base8raw/base8soft的alpha0控制（alpha0成本7/8，不能把控制收益算给相位），4个前缀总51597配置；cal_fast_consistency11475。源cross_phase_fusions_v1会读两方向预测和core8，0新增CNN。B1大cal全部DONE后，51,300张派生hist已核SHA释放25.055GiB（scores/cuts/masks/core logits都保留），cleanup_completed_histograms_v2；hist重建需要core8和冻结配方。

新增brain支撑域f1_0000（brain_domain_prepare_v1）在GPU0纯缓存数学准备：4个基础概率（7T4/原consensus8/case24/joint32）×no-support/输入image!=0填孔及0/1/2膨胀=16，随后183600配置。关键：切点只看支撑内无标签counts；score分母保留整个原crop GT总量，域外阳性全部算FN，独立gate证明没有偷缩评估域。新calibration_brain_domain_v1是fast_consistency变体，score_cuts从GT_TOTALS独立读全GT，只在cut之后评分。若成为赢家需专门auditor/portable分支，普通promote_v7不支持brain kind；目前还没结果。

shape-based signed EDT：准备8预测阈值0.2/0.5/0.8。worker/tmp的二值掩膜复制主机时auto-review明确拒绝，原因跨主机医疗衍生载荷授权不足；没有执行或重试该传输，不许通过共享路径绕过。已安全替代为原worker4CPU/PID10359完成，数据留原处，TOOL_INCIDENT_002记录。SDF定义field(masks,reference,recipe,cache)不接GT；42×15=630配置，主参考SBA2007 PMID17283774；GT只在source返回后评分。SDF早期160/270，预计很快DONE，源码shape_fusion_sweep_v1当前LABEL_DIR=CACHE。若很低则保留负结果不必重算，若近/刷新需专门完整mask审计。

DenseCRF原生3D、worker6CPU PID10150，27×39＋8后cal，早期137/270约34分钟/个案~80秒，预计03:40–04:00完成。DENSE_CRF_STATUS/READY/ROUND018，输入q在TMP/dense_crf_inputs，仅1概率图/例，当前8模型源，0新增CNN。所有模型/标签/输入不动。真实infer(probability,image,params)不接GT。普通promote_v7不能审计dense kind，需专门CPU6重复最高单图再核270矩阵，预算充足。

0013_adaptive_cc已完整审计0.7840012399/Subject0.771026028/Point260，7次；0014_nonlinear_cc0.783827031/Point260，7次。两者裸父0009/0012和其他0010/0011都已入原生双榜45行，登记用register_candidate_v2，研究head指最高（别用v1把最后较低行当head）。7视图0013再203形态worker4CPU/PID10384，分析1201_adaptive_binary_postprocess，搜索最高fill_hole343=0.784072660/Point260尚待专门完整mask审计。

新case扩展cal已DONE：r1_1200_case_consistency_calibration裸0.783559327/Point260（τ0.2/λ24/k8/logitT4，postT1/bias-1/beta1.15）。joint cal d1_1000裸0.783534898/Point259，但最高是pixel exponent0的统一控制，未见voxel因子额外贡献，不要吹成hybrid胜利。两者已用promote_v7完成270独立mask审计到0015_case_consistency、0016_joint_calibration，待244组件矩阵1108_case_consistency_components/1109_joint_components；原worker两个2CPU后台启动（STDOUT见相同名.log）。看DONE选后再promote_components_v1（可原worker上调用，MODULE用LABEL_DIR主机路径的版本需改v2或新wrapper，不能悄改已冻结源）。

Core portable v2已新增joint路由，但v2现在被brain准备的sourceSHA绑定，勿修改；之后扩brain/dense/group请写v3。v1已被fresh270核验绑定，不动。promote最新版v7识别fast3 handler且用cached guard，支持pure core/case/joint/local/group纯数学；不支持brain/dense/SDF专门种类。所有完整候选必须核270mask与搜索再register。

WEB_REVIEW.md存20+原始论文/官方资料、落实/负结果和明确排除的TENT/额外生成器。尚有约5.5小时，最低12小时窗口不提前结束；原source_root本批所有新模型权重训练0。当前完整配置>4m但精确数需再扫DONE，不能把partial当完成。

03:15补充：组件1108_case_consistency最高0.783976211(26连接512)/Point260，1109_joint最高0.783903731(6连接512)/Point259，未刷新0.784139；完整负矩阵保留无需再当新最高吹。SDF已DONE最高0.781269641/Point258(τ.2,一致性32,level0)，630全完。7视图0013再填孔343最高0.784072660/Point260，1201矩阵完成，尚未用promote_postprocess_v2在worker审计到0017等新ID。
新增promote_brain_domain_v1专审f1最优，完整GT域，预测mask支撑内而分母全GT，若刷新可直接worker CUDA0执行 --candidate合适新ID；不用普通promote_v7。当前brain数据准备约41/270仍跑，完成后CPU自动calibration_brain_domain_v1。

## 03:40续接提示

坚持active goal直到08:41至少12h。WORKING_STATE.json是最简checkpoint。不要再增无关分支，专注现有完整实验、独立审计、最终可部署接口与发布。
当前最好0008仍0.784139146/Point258，独立API fresh270 mask0差通过；7视图0018_adaptive_holes已经完整审计并注册，0.784072660/Point260，ALT7配置已导出。原生双榜49行。新plot脚本只用已审计数字，figures/audited_accuracy_cost.svg文本可发布，png留本地。
Dense27×39及后cal全270结束，bestunary-only0.783491753/Point258，没有pairwise收益。promote_dense_crf_v1已在原worker审计到0019，已注册50行；其244组件1111_dense_control_components在原worker运行（PID15724）。无需把无边项收益算CRF。
BrainDomain16格全270＋183600校准完成，bestbase2(case24/T4)/margin0/postT1/bias+.5/beta.5，裸0.783639558/Point260。已启动promote_brain_domain_v1 --candidate0020_brain_domain（SSHsession43528待取）重建全部270，仍全原GT分母；原mask0支撑外831 GT voxels/11scans全部计FN，不偷改域。待auditRESULT后原workercomponent_sweep_v4，再按是否刷新选PP/portable。普通promote_v7不支持brain kind，专审脚本已备。deploy_core_v3准备了brain支持，v1/v2别改（已绑定live/prepare的SHA）；未来真最高要用新V3/APIfresh270再核。
Safe transfer提醒：任何二值预测载荷worker/tmp→主机传输被auto-review拒绝，未执行/未重试；shape仍在原worker已完成630负结果，不要用共享路径绕过同一拒绝。后续所有新病人mask处理在原worker，数据留授权原环境；发布仅源码/文本统计。
CPU速度已优化，worker_v4 PID9770、fast3 handlers和guardcached。根4/6恢复。最大全局CSV按20万行分片；源数据/模型/core8 logits/所有score/cut/mask完整留存；rolling只删新可重建无读者图/hist。已释放>200GiB，新tmp远低300。
最后push只能/tmp/medical_research_publish_20261014，远程maina6087a7；新stager/tmp/stage_auto12h_interim_20261029.py已加移除4个已迁移a2旧activequeue+manifest/pathspec删除，没再推新源。最后fetch最新main，合并本研究marker和native行，保留RACE其他内容；全history63断链/titlechecker mismatch是HEAD基线，自己新增links/SHA/RAW270/manifest必须过。当前代码接口helpers纯函数不读GT/ID，普通核心8fresh270已通过。

## 03:55最新续接

新最高**0021_brain_domain_cc**：Scan0.7842553883239075、Subject0.7714334207652164、Point260、8次；1112的26连接>=512，基于0020 case24/T4+image脑区margin0+postT1/bias+.5/beta.5。promote_components_v2在原worker全270mask精确核验通过，已登记，原生双榜53行。0022_dense_unary_cc也核验登记，0.784138000/Point258，明确无CRF边项控制。

新ROUND023下1202_brain_binary_postprocess（session4813）与1203_dense_control_binary_postprocess（session59413）各203规则原worker4CPU，尚在跑；若赢用promote_postprocess_v2审计新0023等ID。全部新患者mask处理留原worker，禁止再试被拒绝的跨主机传输。

ROUND024 f2_0000_brain_component_beam（3904=16族best×244）已全完，最高与0021完全同配方/分数，不用重复审计。brain_component_beam_v1先从core8用GPU数学生成4张uint16codes/例（~6.8GiB，CNN0），首例16identity全部逐位复现原cal分数，全部准备约51秒，6CPU扫描104秒；codes留workerTMP/analyses/f2.../codes。deploy_core_v3被这个sourceSHA绑定，勿改。

ROUND025 f3_0000_brain_decoder_component_beam刚启动（SSHsession33712）：brain_component_beam_v2逐族从原11475参数按全270均值选8不同全270切点向量SHA去重，128×244=31232，复用f2codes，CNN0。每例identity也必须逐位与原矩阵相等。DONE后如刷新，用**promote_brain_beam_v2 --candidate下一ID**，它从原logits重新聚合/CPU独立singlecut/组件/全GT核270，别把f2/v1auditor混用。新best是base+component结构。若负则保留完整矩阵即可。

PORTABLE_RECIPE_BRAIN_INTERIM.json已导出0021。verify_portable_live_v3准备，导入deploy_core_v3，synthetic由zeros改ones以满足输入脑区非空门；尚未运行，须等phase结束显存释放再batch8/尾6重推当前最终最高，别用仍旧0008fresh通过冒称0021已fresh核验。v1/v2old源码都别改。

research_snapshot_v1新脚本计算完整配置元组、completed NN lowerbound及20秒采样GPU时间加权利用率，写COMPUTE_SNAPSHOT.json。只统计记录，old失联/未完/失败重算未入lowerbound，无空档外推；phase父项与共享前缀不双计。当前5.75m配置（f2完成后再刷新）、completed182340患者前向lowerbound，观察新worker均值GPU0约78%、GPU1约70%，不可把某次双100%说成全程100%。最低12小时仍08:41才完成，工作继续。

03:57补充：1202脑区203形态DONE，fill_small_hole1000最高Scan0.7843116955072792、Subject0.7714053038392915、Point260、8次；已在原worker启动promote_postprocess_v2 --analysis1202_brain_binary_postprocess --candidate0023_brain_holes，SSHsession80762，**尚待全部270auditRESULT**，不能提前注册或声称freshAPI已过。0023若完成，register_candidate_v2后deploy_core_v3导出当前配方；fresh仍等phase腾卡。1203dense控制session59413还在收尾。f3联合31232 session33712，早期30/270；其源auditor是promote_brain_beam_v2。goal持续active，不在08:41前结束。
