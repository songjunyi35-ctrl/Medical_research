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
