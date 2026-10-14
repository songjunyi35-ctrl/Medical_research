# 10小时任务工程接续

窗口2026-09-21 13:04:52–23:04:52 UTC；新训练21:04:52后禁止，全部训练22:04:52前停。用户已批准执行，不重复求许可。所有结果test-tuned。目标>=0.80且比最强五参考至少+0.03，不保证达到。

## 活跃入口
- 双卡worker1104883、单卡1105022，SSH同IPv6 fdbd:dccd:cde2:1001:a4bd:2bcd:1455:6d15，端口11323/10235。当前tiger默认SSH身份可以BatchMode直连，无需读取密钥。
- 两worker运行scripts/supervise_t1ce_10h_v2.py dual/single。v2是工程修订001，只增加等待另一个独立fixed test流程完成，不改冻结训练代码。
- master运行scripts/complete_t1ce_10h.py，启动PID1466005；负责完成后固定top3面板、跨节点缓存直接tar传输、部署检查、文档和五检查，持续到截止。
- master资源监控scripts/watch_t1ce_10h_resources.py，PID1466490，每分钟三GPU利用/显存/温度。
- 实时真相PIPELINE_STATUS.json、HEARTBEAT_dual/single.json、RESOURCE_STATUS.json和workspace各A/B/C/STATUS.json；不因旧文档文字否认访问。

## 数据与实现
- 公共配置scripts/t1ce_10h_common.py；prep、evaluate、train、supervise均独立新入口。FREEZE.json冻结数据阶段依赖；EXECUTION_FREEZE.json冻结训练评估和监督；COMPLETION_CODE_FREEZE.json冻结收尾。
- /tmp/aflow_pretrained_native_res_20260921原生train/validation复用；/tmp/aflow_test270_10h_20260921/test_native新建270缓存须与2026-09-20两payload SHA完全一致。
- workspace/BraTS/segresnet_t1ce_wt/test270_10h_20260921/label_relay仅压缩原生验证/测试标签和几何，供B/C统一原生评分；不进入分割训练。
- B影像聚类权重已准备：min0.559985/max1.300063，ESS908.63，固定全972均值1，未删病例。
- A组合166热启动原始128³分块训练，8块高斯覆盖推理；B179热启动目标影像权重；C179热启动Tversky仅当参考最强FNR-FDR>=0.05。训练上限200/100/100，墙钟4/2/2小时。
- 每5轮+epoch0/1验证，10/25/50预算淘汰，每组270最多4不同checkpoint。训练每轮972各一次，122更新/global8。新榜key test270_tuned_native_rank_by_scan，不动传统榜。

## 已有检查与并发事项
- CPU unittest6项全通过；8块identity reconstruction max9.54e-7。每组GPU smoke在正式train前自动执行。
- 同目录另一会话启动了固定166/172评测，已13:25:36完成且running=false；原生1660.766271575，128³1720.759967225，均未超过各自强单项。本轮等待交接，没有停止或覆盖该任务。保留CONCURRENT_EVALUATION_HANDOFF.json。
- 本轮五模型前向用于完整float32 logits缓存及FP/FN诊断，既有逐例分数不提供这些数据；对旧指标作一致性核对，不能称新独立测试。

## 收尾与注意
- 新冻结代码如需修复，加新版本入口和工程修订，保留旧源码/收据，别直接改哈希绑定历史脚本。
- finalize_t1ce_10h.py select/panel/publish；deploy_t1ce_10h.py check和仅--image/--output的NIfTI入口；verify_t1ce_10h.py额外核对新分榜。
- 四旧验证器必须全过，另有新分榜第五检查；最终EVIDENCE_SHA256、ACCESS_SUMMARY、COMPLETION_AUDIT均需存在。
- 持久新增<=8GiB/空闲>=20GiB，临时本轮<=100GiB/空闲>=100GiB。不删除任何历史文件。监督结束确认无训练/评测进程后才能将goal标complete。

## 14:09 UTC接续补记
- A合成NCCL清理曾挂起，无患者训练前失败；工程修订002添加train_t1ce_10h_v2.py仅在输出提交+barrier后20秒清理退出。双卡现在supervise_t1ce_10h_v3.py，原源码保留。A已过60轮，最优验证epoch40=0.75108449；正式训练正常。
- B完成25轮/3050更新后预算停止，best仍epoch0；C初始参考FNR0.2083/FDR0.23395，条件不符未启动。单卡只保留监控。B/C独立RESULT已生成。
- A test门25：0.77011430（相对最强ref+0.00154120，CI[-.00615391,.01176736]，Point253）；门50：0.76937371，Point250。尚未100/200门。
- 当前参考173用本轮同数值模式得0.7685731058（旧0.7685778971），小幅数值差和1例Point变化记录NUMERIC_COMPARABILITY；缓存/模型SHA确实一致，未改历史分数。
- 从首门错误描述看，旧Dice<.5的15例均值+0.134535，高分>=.8的149例均值-.007759，说明互补但目前总体收益微弱；只描述性，不改训练。
- 已向用户异步询问是否把推理上限从8×放宽至11×以允许A8块+原native2.667×；尚未收到答复，必须保持8×，不要把时间经过当授权。
- 新selector select_t1ce_10h_finalists.py明确每训练分支只取1个目标最佳快照，再取前3不同模型状态，避免3个A epochs冒充3条路线。仍<=19配置，当前cap8。若用户明确选11，才创建INFERENCE_COST_AMENDMENT.json；还需使新分榜额外validator支持该授权，不能默认执行11。
- master已交接complete_t1ce_10h_v3.py PID1474897；它继承v2的每候选归档包装，再接上述selector。每次交接旧PID/源码SHA均归档，训练没有被打断。resource watcher仍1466490。
- 最终步骤自动在A结束后运行；仍必须留到23:04:52 UTC完成十小时监控才将goal complete。若全部优化提前结束，不能新增训练组或为凑时长空转GPU。

## 收尾待执行补记
- A门100用新增epoch90模型：test Scan0.7728296451，强参考差+0.0042565393，95%CI[-0.0039995,+0.0155955]；已3个A快照，最后门200。到155轮仍best90。B参数/逐例报告与原179 exact相同，见B/INITIALIZATION_IDENTITY_CHECK。
- master仍v3 PID1474897；用户11×问题仍未答，cap8。
- 已准备scripts/diagnose_t1ce_10h_trainfit.py，TRAIN_FIT_DIAGNOSTIC_PLAN.json冻结SHA。仅在FINAL_SELECTION与deployment/manifest存在后，于双卡worker空闲GPU1运行（CUDA_VISIBLE_DEVICES=1）；固定193、173、最终流程，在原972 train和108 validation做无增强拟合诊断，0更新/0新test访问，不回流选择，最多90min，属于已批准错误分析/验证预算。此项尚未启动，完成后链接进最终报告并加哈希。
- 当前所有已生成预测（5refs+2A+B）已做finite/shape/SHA审计，LOGIT_AUDIT_dual/single；A100及最终新增输出需在收尾再核查。
- CHECKPOINT_IDENTITIES.json区分原始/新增epoch与门审epoch，收尾需更新最后A快照身份并在报告明确。
- ACCESS_BOUNDARY_CLARIFICATION补记：canonical缓存字节校验早于native构建阶段TEST_PIXEL_PREACCESS，但授权/冻结先存在；DATA_AUDIT原已标pixel_hash_reads=true。不可把native构建收据解释成整个任务首次字节读取；没有倒填日期或隐藏标签选择。

## 15:50 UTC后：计算完成，必须继续到23:04:52监控截止
- A完成200轮/24400更新，best新增90轮；200门复用100门同一快照，没有重复270推理，实际A3个快照。B25轮best0与179参数/逐例预测exact，C未训练但使用了270启动诊断。
- FINAL_PANEL_FREEZE：cap8，12项，A90/native173/combination166三不同方法。用户11×问题尚未回复，未创建INFERENCE_COST_AMENDMENT，不能擅自执行11×。
- 总体panel10=旧native173+旧combination166等权logits，p>0.4，Scan0.775891493、Subject0.761694199、Point261/270。对同轮数值模式强参考0.768573106差+0.007318387，描述性95%CI[.003115548,.012786232]；对固定193同格差+.014756423、CI[.008780445,.021238419]。未达0.80/+0.03，全部test-tuned。
- 最佳单模型panel02=A新增90轮，8块，Scan0.772829645、Subject0.758800132、Point253/270。总体胜者没有用到新训练A，必须明确收益来源。
- 固定.5阈值融合已达.775823547，调.4仅+.000067946；来源OUTPUT_CONTRASTS.json/output_panel/RESULT.md，不是新配置或新推理。
- 打包字典字段覆盖出错，但12项raw和FINAL_SELECTION在失败前已保存。工程修订003/export_t1ce_10h_v2修复，仅生成总体和最佳单模型部署包，没有重跑面板。COMPLETION_RECOVERED passed；旧COMPLETION_FAILURE保留，勿因旧文件重训或重测。
- DEPLOYMENT_CHECK passed：固定4例严格重载，bf16 batch1 vs评分batch2非逐位一致，最大mask disagreement1.896e-5。FINAL_PANEL_CHECKS passed：3个single .5配置与缓存报告逐例Dice/point exact；四旧校验+verify_t1ce_10h均过。
- benchmark合成暖启动：总体batch1 .23566s/scan约5.35×128基线；A单模型 .35137s约7.99×。不含N4/加载/IO。训练进程墙钟折算3.667 GPU小时，不是10h满载。
- 固定clean-fit诊断已完成，无新270：训练972/444 fixed193 .809540/native173 .810327/最终流程 .817523；验证108/50 .742672/.746099/.754431。不能据此单独证明过拟合病因或T1ce上限。RAW_CLEAN_*、TRAIN_FIT_DIAGNOSTIC*.json/md均已保存。
- 最终监控入口scripts/monitor_t1ce_10h_final.py，PID1490748；resource watcher仍1466490。旧master v3/resume已交接停掉，保留各receipt。最终监控包装器在23:04:52调用原收尾并补入拟合诊断、真实成本、C访问范围和checksum访问范围，自动重跑5检查与哈希。
- 当前PIPELINE_STATUS应是results_ready_monitoring，不是complete；deadline前不要update_goal complete。若收到自动goal续轮，继续这条任务监控，不开始新训练。
- OUT/deployment有总体manifest.json、best_single_manifest.json、两旧native模型和A90三个权重。持久新增约.275GiB；dual本轮scratch34.276GiB/single3.406GiB，项目仍约32.6GiB可用。模型/报告/代码保留，无历史删除。
- 新workflow方法卡已补入目录；A/B/C卡的过程和C诊断访问已讲清楚，root交接顶部指向本任务。LIVE_CURVES已更新至200轮，图右侧是门审epoch，并明确快照15/40/90、B为0。
- 最后需确认COMPLETION_AUDIT真实elapsed>=36000、各监督/资源监控退出、没有本轮训练或评测PID；检查DOC_CHECKS五项0、verify_t1ce_10h_panel再核验必要时再查；将AGENTS本轮标题/状态改完成，并同步root顶层指针闭环。OUTPUT_CONTRASTS.json已手工加EVIDENCE_SHA；deadline包装器重写时会漏该额外项，可最终补回（raw已含source SHA且输出方法md被hash，不能改raw）。
- 正式最终答复必须写未达质变目标、原生270/119及test-tuned/无新outer-confirmation、实际训练和推理/磁盘成本，给报告/交接/目录/榜单/归档/模型链接。只有真正10小时结束才标goal complete；不保证指标达到0.80。

## 16:24 UTC 独立就绪审计与verified wait
- 新scripts/audit_t1ce_10h_requirements.py已执行通过。COMPLETION_READINESS_AUDIT.json确认：三清单SHA/患者隔离；源码冻结；A latest所有AdamW参数step24400/B3050；TRAIN.jsonl逐轮972/122；验证最佳选优；A3/B1/C0目标快照预算；21份270报告的每例TP/FP/FN→Dice、患者均值、5670个Point标签判定及绝对/配对患者簇区间独立重算误差<=1e-12；12项cap8面板及目标未达判定；部署哈希/数值范围、磁盘门。唯一pending为10h窗口。
- 此独立核验只读已保存报告和压缩native标签，没有新模型前向/配置/选择；预访问记录INDEPENDENT_METRIC_AUDIT_PREACCESS.json。到最终归档时把该receipt及readiness audit加入总evidence/access汇总。
- 16:04:42–16:24:43做了20分钟真实PID有界观察：master1490748/resource1466490每分钟存活；worker监督37657/8669每5分钟ps确认。记录FOREGROUND_MONITOR_OBSERVATIONS.jsonl。该foreground工具session5808正常结束，不是后台监控结束。
- 持久4个监控进程仍在运行；不要因有界观察session结束重启任何任务。后续goal回合须先核对真实PID，再做verified wait，保持目标直到23:04:52，不提前complete，不将等待当阻塞。

## 16:59 UTC verified wait补记
- 本goal续轮继续核对真实进程并完成16:29:51–16:59:51的30分钟有界观察；master1490748/resource1466490每分钟存活，worker37657/8669每5分钟ps核实。没有任何训练/评测恢复。
- 可复用新scripts/observe_t1ce_10h_live.py --seconds 1800（需允许对这两个worker只读SSH），它只写FOREGROUND_MONITOR_OBSERVATIONS.jsonl且不重启任何研究进程；每次工具等待<=60秒。前景观察session正常结束不是后台监控结束。
- 全目标仍待23:04:52 UTC。后续goal回合归类verified wait，先实查PID；不要因达到静态结果而提前complete，也不要因时间等待而blocked。

## 19:08 UTC单卡可见性事件（须保留，不重启）
- 单卡1105022最后HB19:06:25；19:08首次SSH255 kex closed，复查仍失败；两次mlx worker list仅1104883，1105022已不在活动库存。远程PID8669当前不可直接观察，不能因SSH255宣称PID确定退出，更不能自动重启已关闭B/C。
- WORKER_VISIBILITY_INCIDENT_001.json、MONITORING_AMENDMENT_004.json记录；SUPERVISOR_single状态已改为worker handle missing/remote PID unknown，未假称继续存活。主监控1490748、资源1466490、双卡37657仍存活，继续原23:04:52截止。
- 所有持久模型/报告已验SHA完整；B缓存270文件的既存SHA与dual ref_pretrained179完全一致。SINGLE_CACHE_REDUNDANCY_VERIFIED.json将由双卡现场SHA核验生成，不需要任何模型重评。单卡临时盘当前不可访问，不能宣称仍在线或伪造full10h单卡心跳。
- 后续foreground用scripts/observe_t1ce_10h_live_v2.py --seconds1800；它将SSH255标alive:null/transport unavailable，而不是false退出证据。当前旧foreground window session68873可自然结束，旧观察保持原样并用事件解释。
- 最终报告与审计必须明确这段可见性中断；十小时为整个科研监督窗口，不声称每个worker都全程在线。平台worker handle缺失且研究早已terminal，可按权威库存记录其不可用；不请求新GPU、不恢复训练。截止时再查询库存确认，主/双卡退出需实际PID核验。

## 19:17 UTC 事件已纳入自动终报
- 复查平台清单已保存PLATFORM_INVENTORY_AFTER_SINGLE_LOSS.txt，仍只有1104883；单卡当前不可观测，原因未知。SINGLE_CACHE_REDUNDANCY_VERIFIED.json现场核对dual ref179的270文件SHA与single B既存SHA完全相同，因此当前选定结果不依赖失联临时盘。
- 主监控已交接scripts/monitor_t1ce_10h_final_v2.py PID1526488（旧1490748已按声明停止）；resource1466490、dual37657保持。新包装器会在最终报告明确单卡中断，storage.single只代表最后观测值，不能声称每节点连续在线10h。
- 新foreground应使用observe_t1ce_10h_live_v2.py --seconds1800；SSH255必须是alive:null，不是PID退出证明。原foreground可能因计划内master交接短暂窗口结束，仅启动新观察，不重启研究任务。
- 所有模型/配置/指标未变，四旧检查+新分榜检查继续全过；整体监督截止仍23:04:52。事件和原始平台清单已加入最终evidence自动封存。

## 21:06后双卡也不可见（不能重启研究）
- dual最后HB21:06:12，21:06:43起SSH255，复查仍失败；两次mlx worker list均只有表头，活动worker清单为空。WORKER_VISIBILITY_INCIDENT_002.json、PLATFORM_INVENTORY_AFTER_DUAL_LOSS.txt、MONITORING_AMENDMENT_005.json已落位。
- A在15:17完成、所有面板/部署检查/拟合诊断在15:32前完成。节点不可访问后再核验3部署权重、21报告、3清单SHA全对，见PERSISTENT_ARTIFACTS_AFTER_DUAL_LOSS.json。临时缓存现在均不可访问，未重建、未重评、不新开GPU。
- SUPERVISOR_dual状态改为worker handle missing/remote PID unknown；不要仅凭SSH255断言PID退出。平台权威活动handle缺失需如实记录，整体10h由master继续监控已终态工作、资源端点与磁盘，不能声称GPU worker全程在线。
- 主监控已交接scripts/monitor_t1ce_10h_final_v3.py PID1549240；旧1526488停止。resource1466490仍活。所有后续检查必须动态读COMPLETION_LAUNCH，不用旧PID硬编码。终报会纳入两事件、两节点存储仅最后观测值和整体监督范围。
- 四旧检查+新分榜检查仍0。截止仍23:04:52；final应核查master/resource退出、平台活动清单及记录远程不可直接观察，不重启GPU任务。当前foreground session20557可自然结束，下一goal turn仍用observe_t1ce_10h_live_v2.py。
