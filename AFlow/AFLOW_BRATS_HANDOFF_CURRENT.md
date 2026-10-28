# 当前科研状态与接续边界

<!-- AUTO_INFERENCE12H_20261028 -->
**12小时自动推理研究仍运行至29日08:41。** 用户明确直接按270反馈全局搜索，固定同w32第171轮、T1ce及原972/108/270，无训练/outer/confirmation，独立RACE不动。worker1153287双A10080，数据SHA和旧五镜像270mask/point/Dice精确一致。当前已审计最高Scan0.784139146、Subject0.771156128、Point258/270，8视图一致性加权/校准/连通域；7视图替代0.783850/Point260；较原普通推理+0.010911106，描述性95%[+0.006975465,+0.016120434]，全部test-tuned且未校正多重搜索。所有270掩膜独立核对，双榜39行保留阶段历史。

约76万全270配置完成，1464216组合在GPU0评分；512整数平移相位将在新scratch释放60GiB后用GPU1运行，1/64/128/256/512共享前缀、16CPU校准。恢复3/6；仅删本轮已归档无读者派生文件，输入/权重/原始core8 logits/全部评分掩膜保留。研究不提前结题。[快照](archive/research_records/2026-10-28_segresnet_auto_inference12h/INTERIM_RESULT.md) · [实时状态](archive/research_records/2026-10-28_segresnet_auto_inference12h/RESEARCH_STATUS.json) · [下一步接续](archive/research_records/2026-10-28_segresnet_auto_inference12h/RUNBOOK.md)。
<!-- END_AUTO_INFERENCE12H_20261028 -->

<!-- INFERENCE_SELECTED_TEST270_20261028 -->
**2026-10-28固定五镜像＋体积解码270测试完成。** 270/119 Scan0.780995、Subject0.768306、Point255/270；对同权重原始推理差+0.007767，配对95%区间[0.004495999901909861, 0.011791830233831117]，配对区间支持本复用队列的探索性正差。 模型/配方从108选择后固定，本次无test调参，训练0、无新outer/confirmation。复用内部探索，非独立泛化；本批结束不扩搜/重测，不干预独立RACE训练。[报告](archive/research_records/2026-10-28_segresnet_inference_selected_test270/RESULT.md)。
<!-- END_INFERENCE_SELECTED_TEST270_20261028 -->

<!-- INFERENCE_SEARCH_20261028 -->
**2026-10-28固定w32推理搜索完成。** 用户选择32通道171轮唯一权重，worker1152682单A10080GB；32037配置全部原108/50验证，无新训练/270/outer/confirmation。108验证原始0.753271→搜索最高0.761779（差+0.008508），Subject0.750409，Point96/108；患者分折0.761204。 复用验证选择后的探索证据，非独立确认；原RACE双卡训练不受影响。搜索终止，不自动扩搜或测试。[结果](archive/research_records/2026-10-28_segresnet_inference_search/RESULT.md)。
<!-- END_INFERENCE_SEARCH_20261028 -->

<!-- WIDTH32_TEST270_20261028 -->
**2026-10-28固定w32第171轮270测试：** 固定171轮270评测完成：Scan0.773228、Subject0.760027、Point256/270；对多尺度166差+0.001842，95%CI[-0.003712439459196132, 0.007115013465642496]。配对区间跨0，没有确认稳定增益或等效。[结果](archive/research_records/2026-10-28_segresnet_w32_e171_test270/RESULT.md)。 新用户授权仅该模型；原训练验证门未过，单T1ce/同原生栅格/阈值0.5，不重新选epoch或融合。无新outer/confirmation。[方案](archive/research_records/2026-10-28_segresnet_w32_e171_test270/IDEA.md) · [状态](archive/research_records/2026-10-28_segresnet_w32_e171_test270/PIPELINE_STATUS.json)。
<!-- END_WIDTH32_TEST270_20261028 -->

<!-- WIDTH_RESUME_20261027 -->
**2026-10-27最新：用户撤销早停，恢复累计200轮。** 两组各200轮完成；w16最佳166轮108 Scan0.747497；w32最佳171轮108 Scan0.753276；32−16差+0.005779，收益门False。[结果](archive/research_records/2026-10-27_segresnet_width_resume/RESULT.md)。 旧85/50停止事实保留，新阶段接续86/51轮，微批8/4、有效8/122更新/LR不变。[新阶段](archive/research_records/2026-10-27_segresnet_width_resume/IDEA.md) · [实时状态](archive/research_records/2026-10-27_segresnet_width_resume/PIPELINE_STATUS.json)。无新270/outer/confirmation；旧停止目录不是新阶段运行状态。
<!-- END_WIDTH_RESUME_20261027 -->

<!-- WIDTH_THROUGHPUT_AUDIT_20261027 -->
历史提速阶段：从完整75/45轮恢复到76/46轮，微批8/4、有效8/122更新不变，并非从头重训。随后用户授权前50轮筛选触发，实际85/50轮停止，训练进程已退出、两卡已释放。提速收据不代表当前仍在训练。[最终停止结果](archive/research_records/2026-10-27_segresnet_width/RESULT.md)。
<!-- END_WIDTH_THROUGHPUT_AUDIT_20261027 -->

<!-- WIDTH_20261027 -->
**2026-10-27容量对照：** 用户授权前50轮筛选后停止：w16前50最好0.710403、实际结束85轮；w32前50最好0.700533、实际结束50轮。32未超过同期16，撤销剩余200轮预算，不是完整训练终态；不证明32通道普遍无效。[结果](archive/research_records/2026-10-27_segresnet_width/RESULT.md)。 新授权仅16/32通道两组、同随机初始化规则，不恢复旧组。原972/108、各200轮；无新270/outer/confirmation。[协议](archive/research_records/2026-10-27_segresnet_width/IDEA.md) · [实时状态](archive/research_records/2026-10-27_segresnet_width/PIPELINE_STATUS.json)。
<!-- END_WIDTH_20261027 -->

<!-- SEGRESNET_REVIEW_20261027 -->
**2026-10-27文献与失败复核完成：** 三组蒸馏未过门，已核对SegResNet原文、官方BraTS/Auto3DSeg配置及现有训练日志。当前16通道470万参数是有效实现，但不是原文32通道+VAE完整配方；CPU实例计数32通道约1880万。两蒸馏组人工标签训练损失也较高，原因未被因果定位。下一步优先设计容量/训练配方分离验证，不恢复旧训练，不自动加损失或测试。[核对材料](archive/reviews/2026-10-27_segresnet_literature_audit/RESULT.json)。
<!-- END_SEGRESNET_REVIEW_20261027 -->

<!-- DISTILL_20261026 -->
2026-10-26蒸馏三组：三组各200轮完成；control最佳189轮108 Scan0.745119；plain最佳183轮108 Scan0.743886；reliable最佳158轮108 Scan0.738286。明确收益门False。[结果](archive/research_records/2026-10-26_segresnet_teacher_distill/RESULT.md)。 [协议与边界](archive/research_records/2026-10-26_segresnet_teacher_distill/IDEA.md) · [实时状态](archive/research_records/2026-10-26_segresnet_teacher_distill/PIPELINE_STATUS.json)。无新270/outer/confirmation，不恢复旧分支。
<!-- END_DISTILL_20261026 -->

2026-10-22从头训练B组complete：完成550/550轮，34100次更新尝试、34091次实际更新，选525轮；108/50验证Scan0.771742492、Subject0.768001627、Point104/108。对原e525差-0.012929153，患者配对95%区间[-0.019600229,-0.006408284]，明确收益门=False。 无新270/outer/confirmation。[结果](archive/research_records/2026-10-22_nnunet_local_full550/RESULT.md)



2026-10-21旧缓存清理完成：按用户授权删除12个旧输入载荷，持久盘释放14.21875GiB、空闲约37.44GiB；另删除主节点/tmp旧公开334缓存3.26171875GiB。构建脚本、索引/历史SHA、原始源数据、全部模型及报告保留；检查两组训练进程/子进程共14个，确认未用所删缓存，训练继续。清理本身无数组解码或新模型/评分执行。[清理与重建入口](archive/reviews/2026-10-21_unused_cache_cleanup/RESULT.md)。

2026-10-21两个nnU-Net算法组complete。A体积预算竞争：完成150/150轮，尝试37500、实际更新37488；选新增0轮，108 Scan0.784664851、Subject0.780815550、Point105/108。对旧e525 Scan差-0.000006794，患者配对97.5%区间[-0.000023814,+0.000001814]，明确收益门=False。 B局部视图一致性：完成150/150轮，尝试37500、实际更新37488；选新增120轮，108 Scan0.785393948、Subject0.780034644、Point107/108。对旧e525 Scan差+0.000722304，患者配对97.5%区间[-0.002742044,+0.003888593]，明确收益门=False。 本批无新270/outer/confirmation；普通微调对照已取消，算法与额外训练的贡献未被单独识别。 [结果](archive/research_records/2026-10-21_nnunet_error_balance/RESULT.md)





<!-- HF_DICE081_20261020 -->
**2026-10-20完整0.81模型归档完成。** [私有Hugging Face仓库](https://huggingface.co/songjunyi/Medical_seg)现已保存nnU-Net e525（单模型0.811283）、旧原生173／组合166及0.812965融合配方，原始与推理权重、配置和审计齐全；77文件远端核验及六份权重回下载SHA核验通过，旧334预训练档案保留。没有新训练或患者评测；当前0.81结果仍为test-tuned。[发布记录](archive/reviews/2026-10-20_huggingface_dice081/README.md)。
<!-- END_HF_DICE081_20261020 -->

**2026-10-20预训练起点已备份到Hugging Face。** 公开334例、150轮、6300更新的SegResNet原始final权重与等价safetensors已上传至私有仓库[songjunyi/Medical_seg](https://huggingface.co/songjunyi/Medical_seg/tree/main/segresnet-t1ce-openneuro334-pretrain-e150)，21文件远端核验及两份权重回下载SHA核验通过。它是后续SegResNet微调起点，不是nnU-Net模型；没有新训练或患者评测。[上传与核验记录](archive/reviews/2026-10-20_huggingface_pretrain334/README.md)。

<!-- NNUNET_RESENC_20261013 -->
**2026-10-14残差编码器组已完成训练、验证、270及六项融合。** 550轮完成，108选525轮Scan0.784672，验证门通过。270单模型Scan0.811283，最佳三模型融合Scan0.812965、Subject0.797512、Point266/270；相对旧榜首+0.037074，95%CI[+0.030140,+0.044262]。达到本批0.8均值目标，全部test-tuned，非独立泛化。 训练节点1132660，评测接续1133854。V3结果与独立复核已归档；旧COORDINATOR_FAILURE描述历史归档失败。本组结束，不自动重训、扩组、换epoch或重测，无新outer/confirmation。[最终报告](archive/research_records/2026-10-13_nnunet_resenc_reference/RESULT.md) · [真实终态](archive/research_records/2026-10-13_nnunet_resenc_reference/PIPELINE_STATUS.json)。
<!-- END_NNUNET_RESENC_20261013 -->

同日的[候选筛选与纯合成探测](archive/research_records/2026-10-13_dice080_proposal/IDEA_strong_recipe.md)随后转入正式运行，并于10月14日完成；旧“训练0轮”和“尚无新270分数”只描述准备阶段。旧0.775891为本组前历史榜首，当前结果见上方终态。首次合成重载接口问题已在患者模型访问前通过V2入口修复，后续评测归档通过V3完成；保留旧失败记录，不重启旧入口。

<!-- VALID_REGION_20261010 -->
**2026-10-10有效区域监督：完成200轮。** worker1127648双A10040GB，单组原972/108、单T1ce原生多尺度SegResNet；当前完成200/200轮，唯一变化为几何有效区域损失，global8/每轮122更新。选175轮108 Scan0.746077、Subject0.734457、Point98/108；对旧多尺度166差-0.007015、95%CI[-0.023860,+0.003203]。未达到事前明确验证收益门；只评价当前固定配方，不证明所有有效区域策略无效。无本候选新270/outer/confirmation访问，test-informed开发。[真实状态](archive/research_records/2026-10-10_segresnet_valid_region_supervision/PIPELINE_STATUS.json) · [方案与报告](archive/research_records/2026-10-10_segresnet_valid_region_supervision/RESULT.md)。
<!-- END_VALID_REGION_20261010 -->

<!-- NEXT_IDEAS_20261010 -->
**2026-10-10新资源方案筛选完成：** 用户提供2张A10040GB并要求网上搜索。优先考虑有效区域监督，其次公开334辅助回放，较大预算备选为nnU-Net ResEnc L完整管线参考。原972训练几何显示画布补零占比中位数35.92%，只是候选依据，尚无Dice增益。三项均为草案、训练0轮/更新0/患者前向0；本轮无新270像素/模型或outer/confirmation访问，worker1127648已通过两卡CUDA/bf16和存储预检，候选尚未启动。旧已关闭实验不恢复。[筛选依据与七问](archive/research_records/2026-10-10_two_a100_next_ideas/REVIEW.md)。
<!-- END_NEXT_IDEAS_20261010 -->

<!-- MULTISCALE_E166_TEST270_20260929 -->
**2026-09-29多尺度166固定270评测完成：** 原270扫描/119患者，native1mm裁剪栅格排除填充，Scan 0.771386、Subject 0.757817、Point 248/270。对同格旧组合166差+0.005115、患者配对95%CI[+0.000590,+0.009713]；复用队列的95%配对区间为正，但97.5%区间仍跨0；这是小幅收益的探索性证据，尚不能认定稳健或独立泛化提升。原验证门未过，test-tuned开发/复用内部探索；此次唯一166模型已访问270，无当前test调参或新outer/confirmation。评测与归档结束，不自动重测或新训。[报告](archive/research_records/2026-09-29_segresnet_multiscale_e166_test270/RESULT.md)。
<!-- END_MULTISCALE_E166_TEST270_20260929 -->

<!-- MULTISCALE_RESUME_20260929 -->
**2026-09-29多尺度辅助监督续训终态：** 完成200轮、24400更新。108扫描/50患者原生1mm裁剪栅格：选第166轮，Scan 0.753092、Subject 0.739669、Point 94/108；对旧公开预训练＋原生组合166同格Scan差+0.001913，50患者配对95%CI[-0.004999,+0.010082]。 冻结提升门未通过；训练阶段无新270/outer/confirmation访问；后续用户授权166固定270测试已完成，见最新评测条目、非独立泛化。旧数据处理审视和病灶均衡选块为其他已关闭批次。[完整报告](archive/research_records/2026-09-29_segresnet_native_multiscale_resume50/RESULT.md)。
<!-- END_MULTISCALE_RESUME_20260929 -->

<!-- MULTISCALE_20260924 -->
**2026-09-24多尺度辅助监督终态：** 按50轮规则停止、6100更新。108扫描/50患者原生1mm裁剪栅格：选第49轮，Scan 0.718759、Subject 0.708114、Point 101/108；对旧公开预训练＋原生组合166同格Scan差-0.032420，50患者配对95%CI[-0.040790,-0.024388]。 冻结提升门未通过；本组无新270/outer/confirmation访问、非独立泛化。旧数据处理审视和病灶均衡选块为其他已关闭批次。[完整报告](archive/research_records/2026-09-24_segresnet_native_multiscale_supervision/RESULT.md)。
<!-- END_MULTISCALE_20260924 -->

**2026-09-24最新审视：** 用户授权直接查看270测试影像/标签特征以判断原972训练数据处理。既有128³缓存三组全部核查；影像分布可分性AUC0.554，14项中位数差均<训练IQR0.16倍；测试小体积比例未超训练，标签微碎片和画布边缘比例也相近。结合旧目标域加权B组25轮未带来验证增益，本批没有足够依据修改训练数据或再开GPU训练。270图像/标签已用于test-tuned诊断，但无新模型前向/评分、无outer/confirmation。结论、证据边界及真实访问见[完整审视](archive/reviews/2026-09-24_test270_image_profile/RESULT.md)；之前病灶均衡选块诊断仍是已关闭批次。

**2026-09-24最新终态：** worker1109780双A100实际计算通过；原972/444的病灶均衡选块诊断完成。预设881个小区域覆盖只提高0.842个百分点，低于5个百分点准入门，按方案不启动训练；训练0轮/0更新、模型前向0，无新验证/test/outer/confirmation。小区域为1–5体素标签连通区域，不直接等同临床小病灶；绝对覆盖未测，不声称旧覆盖充分。本实现关闭，不自动改门或扩组。[最终报告](archive/research_records/2026-09-24_segresnet_lesion_balanced_patch_proposal/RESULT.md) · [状态](archive/research_records/2026-09-24_segresnet_lesion_balanced_patch_proposal/PIPELINE_STATUS.json)。9月21日10小时批次已完成，下方其“当前执行批次”属于历史措辞；旧任务不恢复。

<!-- ACTIVE_10H_POINTER -->
**2026-09-21历史10小时批次已完成并关闭。** 原训练、门审、输出面板与可见性事件保留在[历史报告](archive/research_records/2026-09-21_t1ce_test270_10h/RESULT.md)；不恢复旧运行，不能据此声称当前有训练任务。
<!-- END_ACTIVE_10H_POINTER -->

更新时间：2026-09-21。**原972纯SegResNet193仍是唯一比较基线。原始分辨率组（方向1）200轮训练与用户授权的一次test270均已完成：同栅格配对差+0.007444、CI跨0=True，分辨率方向关闭。9月21日复核模型哈希及270例报告完整一致，无需重复评测；随后新授权的两候选执行状态见下。** [完成状态核对](archive/reviews/2026-09-21_native_res_test270_completion_audit/RESULT.md)。

**本次固定候选测试终态：** 本次两个固定候选各一次270扫描/119患者测试均已完成。组合166在native grid的Scan 0.766272、Subject 0.750958、Point 236/270；对原native173差-0.002306、95%CI[-0.009078,+0.005791]。患者等权172在128³的Scan 0.759967、Subject 0.745647、Point 250/270；对预训练179差-0.001682、95%CI[-0.008389,+0.005860]。不同栅格不直接相减。验证门未过的事实保留；本次为用户授权、复用内部队列探索，无本批test调优或新outer/confirmation。[组合测试](archive/research_records/2026-09-21_segresnet_pretrain_native_e166_test270/RESULT.md) · [患者等权测试](archive/research_records/2026-09-21_segresnet_patient_equal_e172_test270/RESULT.md)。

**训练与验证终态：** 两组均完成200轮/24400更新及108/50验证，组合选166、患者等权选172；均未过明确提升门。随后用户另行授权的270评测已完成，训练阶段无test标记仅描述当时阶段。此次测试结束，不自动恢复本批训练或重复评测；其他批次见其独立协议与状态。

**前一批次终态：** 两组全部训练与一次270评测完成，当前无本轮运行任务。语义组Scan0.757718，未提分；公开334预训练150轮+原972微调200轮，验证选179，test Scan0.761649/Subject0.745859/Point246，是128³自动总体及1× Dice最高点估计；对原193差+0.002470、95%CI[−0.004343,+0.009627]跨0，未达到预设验证提升门，不能称决定性优化。原193继续作为用户固定对照，不自动替换部署或追加组。两候选test均已访问，无当前test调优或新outer/confirmation。[预训练结果](archive/research_records/2026-09-21_segresnet_external_pretrain/RESULT.md) · [语义结果](archive/research_records/2026-09-21_segresnet_semantic_regions/RESULT.md)。

**前一批次归档状态：** 以[旧两候选总状态](archive/research_records/2026-09-21_segresnet_two_experiments/LIVE_STATUS.json)为准；旧顺序监督的PIPELINE_STATUS/源数据失败标记仅描述旧队列，不代表新单卡预训练停止。

## 当前优化起点（2026-09-18用户指定）

用户已确认后续训练与部署保持单T1ce，不加入其他MRI序列。以后统一以原972纯SegResNet第193轮单次推理为基线：验证Scan0.740032、测试Scan0.759179，角色分开比较。旧756/864/蒸馏结果不再充当当前对照。仅108验证的固定八镜像诊断已完成：Scan0.741476相对基线仅+0.001444，配对CI跨0，Point97→95，未晋级；不重训、不重跑270。[固定基线与本次结果](archive/research_records/2026-09-18_plain972_e193_flip8_validation/RESULT.md)。

## 已冻结基线结果

验证108扫描/50患者选中第193轮：Scan0.740032、Subject0.731538。随后固定模型一次评估复用内部270扫描/119患者：**Scan0.759179、Subject0.744985、Point242/270**。Scan患者整簇95%区间[0.732638,0.781993]，Subject[0.718192,0.768915]。[测试完整报告](archive/research_records/2026-09-17_segresnet_plain972_e193_test270/RESULT.md)。

此193模型是用户固定比较基线；128³自动总体及1× Dice最高点估计现为预训练179的0.761649/0.745859，但对193区间跨0，未达预设明确提升门。下面保留193的历史比较：比旧8镜像0.757270高0.001909，但配对95%区间[−0.004567,+0.009086]跨0，不能称显著胜出。Point低于旧8镜像265/270与旧蒸馏单次262/270，不称全面最佳。[精确榜单](AFLOW_BRATS_LEADERBOARD.md)。

## 固定训练与评估口径

原972扫描/444患者训练、108/50验证、270/119测试及三清单SHA/患者隔离不变。纯470万参数SegResNet随机初始化、完整200轮，每轮972各一次/122更新，共24400更新；best/latest严格重载一致。

单T1ce，WT标签1/2/3排除4；N4、图像自身脑区裁剪到128³。无GT定位/点击、蒸馏、灰度项、中心头、EMA、融合、TTA或后处理；固定raw logit>0、bfloat16 batch2评估，部署1次前向。SAM使用GT中心裁剪和模拟点击，权限/栅格不同，不作纯架构消融。

训练双A100约1小时43分、每卡峰值10.1GiB。测试GPU推理8.72秒（缓存已备，不含N4构建）；独立193模型约18MiB，无新影像缓存。[固定193轮模型](archive/research_records/2026-09-17_segresnet_plain972_e193_test270/checkpoint_epoch193.pth)。

## 访问与历史边界

本组test已经用户授权并完成，不能再把训练阶段FINAL_RESULT.json中的new_test_pixels_access:false解释为当前整个实验未测试。访问补记、独立test收据与本页为准。无新增outer/confirmation/外部评估。

test270是已复用内部队列；候选在本次测试前选定且固定，但此前研究方向受历史测试揭示影响，不称全链路盲测。不得据结果调loss、epoch、阈值、分辨率、融合/镜像或后处理，不自动测试其他检查点。

旧纯SegResNet0.725820/0.741204用了756训练；972蒸馏0.751760、8镜像0.757270是历史优化方案，保留原分数与角色。原fit864优化器0.752003是旧1×Scan纪录，不恢复旧支线。

两次SAM灰度组仍停止：均值匹配161/200，50/161轮test分别0.566424/0.515107；Mumford–Shah45/200，只有108验证best5=0.571493。历史负结果和访问保留；SAM权重已在9月18日按用户授权删除，不能因旧记录自动恢复。

## 2026-09-18最新决定

用户明确不需要八镜像test；准备工作在启动前取消，没有新增test像素访问或推理。[取消收据](archive/research_records/2026-09-18_plain972_e193_flip8_test270/CANCELLED.json)。当前仅讨论SegResNet改进：现有预处理已较完整，优先辨明输入信息/空间细节与训练误差；保留空间细节的分块训练仍是未启动候选，不保证大幅收益；用户确认单T1ce后，多序列方向退出当前计划。[分析与官方依据](archive/reviews/2026-09-18_segresnet_improvement_options/RESULT.md)。

## 后续待办

固定基线已建立；八镜像test取消、无N4停止、原始分辨率173完成的边界保留。本次组合166与患者等权172训练、固定测试和归档均完成，没有待自动恢复的本批任务；本次未动用test调优许可。其他新授权批次见独立状态，不能据本批终态停止其他任务。

组合训练及单卡患者等权实验现已完成训练、验证和用户追加授权的固定270测试；旧提案与阶段日志保留为历史。

[训练最终报告](archive/research_records/2026-09-17_segresnet_plain972_baseline/FINAL_RESULT.json) · [当前测试JSON](archive/research_records/2026-09-17_segresnet_plain972_e193_test270/RESULT.json) · [方法目录](AFLOW_BRATS_METHOD_CATALOG.md) · [流水账](AFLOW_BRATS_RESULTS.md) · [导师汇报](AFLOW_BRATS_WEEKLY_REPORT.md)

## 项目清理

已删除SAM权重与关闭分支派生缓存约37.956GiB，保留全部SegResNet权重、原始数据、数据清单及报告。公共代码不按旧目录名删除。[清理收据](archive/reviews/2026-09-18_segresnet_only_cleanup/RESULT.md)。

<!-- NON4_972_STATUS -->
无额外N4单T1ce纯SegResNet：50轮按用户规则停止，禁止自动恢复，完成50/200轮；每轮原972各一次、122次更新。 最佳验证第45轮Scan 0.703354。 原108验证选优，270测试不访问；唯一研究变化为取消额外N4，不预设当前0.759179为创新收益。 [进度](archive/research_records/2026-09-18_segresnet_plain972_no_n4/RESULT.md) · [实时JSON](archive/research_records/2026-09-18_segresnet_plain972_no_n4/RESULT.json)
<!-- END_NON4_972_STATUS -->

## 第50轮用户预算修订

无N4组已在50轮完成后停止：同轮0.700262对N4的0.694616，差+0.005645、95%CI[−0.006018,+0.018626]；41–50均差−0.000907，未达明确差异门。保留前50最佳45轮0.703354及50轮快照，禁止自动恢复，不是等效证明，无新test。[判断规则](archive/research_records/2026-09-18_segresnet_plain972_no_n4/EPOCH50_REVIEW/AMENDMENT.md)。

<!-- NATIVE_RES_972_STATUS -->
原始分辨率整脑纯SegResNet（不缩放到128³）：200轮完成，验证选第173轮；用户授权的一次test270已完成。原始栅格Scan 0.768578、Subject 0.754144、Point 262/270；同栅格基线193为Scan 0.761134，配对差+0.007444、95%CI[-0.000608,+0.014305]，配对区间跨0，不确定。验证阶段配对差+0.003422、CI[-0.0009,+0.0077]。128³栅格的0.759179不可与本行直接相减。复用内部非盲测；不据此调参或重测。 [测试结果](archive/research_records/2026-09-20_segresnet_native_res_e173_test270/RESULT.md) · [训练记录](archive/research_records/2026-09-18_segresnet_plain972_native_res/RESULT.md)
<!-- END_NATIVE_RES_972_STATUS -->

<!-- TWO_IDEAS_PIPELINE -->
两候选均已完成：语义分解未提分；公开预训练仅小幅点估计提升，未达到预设明确增益门。本轮停止，不自动重训、换epoch或重复测试。[最终总状态](archive/research_records/2026-09-21_segresnet_two_experiments/LIVE_STATUS.json)。
<!-- END_TWO_IDEAS_PIPELINE -->


公开数据缓存已全部334例核验通过；复制到1105022临时盘的三文件SHA与1104883一致。项目另保留约3.26GiB缓存中转副本，未上传HDFS。早期字段名/外部编号兼容失败及工程修订保留；不重开原失败队列。[存储与修复](archive/research_records/2026-09-21_segresnet_two_experiments/EXTERNAL_STORAGE_AND_REPAIR.md)。

1104813/1105002/1105007/1105011的CUDA失败记录保留；1105022已通过UVM、驱动和真实计算检查，不再等待故障节点。[驱动对照](archive/reviews/2026-09-21_cuda_uvm_differential_diagnosis/RESULT.md)。

<!-- TWO_IDEAS_SEMANTIC_REGIONS -->
前景语义分解SegResNet：200轮完成，选第158轮；108验证Scan 0.739689，未达到预设验证提升门。270/119同128³测试Scan 0.757718、Subject 0.743434、Point 238/270；对193差-0.001461、95%CI[-0.006823,+0.003567]，差值区间跨0。单模型1次推理；无当前test调优或新outer/confirmation。[结果](archive/research_records/2026-09-21_segresnet_semantic_regions/RESULT.md)。
<!-- END_TWO_IDEAS_SEMANTIC_REGIONS -->

<!-- TWO_IDEAS_FINETUNE_PRETRAINED -->
公开334例预训练后微调SegResNet：200轮完成，选第179轮；108验证Scan 0.746851，未达到预设验证提升门。270/119同128³测试Scan 0.761649、Subject 0.745859、Point 246/270；对193差+0.002470、95%CI[-0.004343,+0.009627]，差值区间跨0。单模型1次推理；无当前test调优或新outer/confirmation。[结果](archive/research_records/2026-09-21_segresnet_external_pretrain/RESULT.md)。
<!-- END_TWO_IDEAS_FINETUNE_PRETRAINED -->

<!-- PRETRAIN_NATIVE_COMBINATION -->
训练/108验证阶段已完成200轮，选第166轮，验证门未过；后续用户授权test270也已完成。旧训练JSON中的无test标记仅指训练阶段。[本候选测试与访问补记](archive/research_records/2026-09-21_segresnet_pretrain_native_e166_test270/RESULT.md)。
<!-- END_PRETRAIN_NATIVE_COMBINATION -->

<!-- PATIENT_EQUAL_LOSS -->
训练/108验证阶段已完成200轮，选第172轮，验证门未过；后续用户授权test270也已完成。旧训练JSON中的无test标记仅指训练阶段。[本候选测试与访问补记](archive/research_records/2026-09-21_segresnet_patient_equal_e172_test270/RESULT.md)。
<!-- END_PATIENT_EQUAL_LOSS -->


<!-- TEST270_10H -->
10小时单T1ce定向优化：complete。270/119原始栅格总体Scan 0.775891、Subject 0.761694、Point 261/270；最佳单模型Scan 0.772830。对最强同栅格参考差+0.007318、95%CI[0.0031155479307933965, 0.012786231698480971]，未达到0.80且至少+0.03目标。全部test-tuned，非独立泛化；270不进分割梯度，无outer/confirmation。[报告](archive/research_records/2026-09-21_t1ce_test270_10h/RESULT.md)。
<!-- END_TEST270_10H -->

<!-- FOLLOWUP_TEST_PATIENT -->
预训练＋患者等权损失172：test270/119完成，128³栅格，Scan 0.759967、Subject 0.745647、Point 250/270；对pretrained179差-0.001682、95%CI[-0.008389,+0.005860]。验证门未过，用户授权探索性测试；无当前test调优或新outer/confirmation。[完整报告](archive/research_records/2026-09-21_segresnet_patient_equal_e172_test270/RESULT.md)。
<!-- END_FOLLOWUP_TEST_PATIENT -->

<!-- FOLLOWUP_TEST_NATIVE -->
公开预训练＋原分辨率组合166：test270/119完成，原始1mm裁剪栅格（native grid），排除填充，Scan 0.766272、Subject 0.750958、Point 236/270；对native173差-0.002306、95%CI[-0.009078,+0.005791]。验证门未过，用户授权探索性测试；无当前test调优或新outer/confirmation。[完整报告](archive/research_records/2026-09-21_segresnet_pretrain_native_e166_test270/RESULT.md)。
<!-- END_FOLLOWUP_TEST_NATIVE -->


<!-- TEST270_10H_WORKER_VISIBILITY -->
19:08 UTC起单卡1105022端点不可访问，平台活动列表只剩1104883；B训练早已完成，模型/报告持久化完整，B缓存哈希与双卡参考完全一致。未恢复任何训练，主节点/双卡监督继续到原截止；单卡此后的进程状态不冒充已确认存活。详见[节点事件](archive/research_records/2026-09-21_t1ce_test270_10h/WORKER_VISIBILITY_INCIDENT_001.json)。
<!-- END_TEST270_10H_WORKER_VISIBILITY -->


<!-- TEST270_10H_DUAL_VISIBILITY -->
21:06 UTC起双卡1104883也不可访问，平台活动worker清单为空。全部科研计算已结束；3部署权重/21报告/3固定清单在事后重新验SHA完整。余下窗口由主节点监控持久化终态、资源端点和磁盘；不重建缓存、不恢复训练，不声称GPU worker全程在线。[双卡事件](archive/research_records/2026-09-21_t1ce_test270_10h/WORKER_VISIBILITY_INCIDENT_002.json)。
<!-- END_TEST270_10H_DUAL_VISIBILITY -->

<!-- RACE_PREP_20261028 -->
## 2026-10-28 RACE-Seg双卡单算法200轮已接续

用户取消的新对照已在完整4轮/488更新停止，禁止自动恢复。算法从完整第5轮/610更新模型与AdamW接双卡DDP，已完整完成第6轮/732更新、108扫描/50患者验证和CPU严格重载，随后进入第7轮。两rank参数SHA一致；两卡共同训练一个算法，目标累计200轮/24400更新，有效batch8、每卡微批2、原200轮学习率不变。
首个双卡完整epoch含验证3.31分钟，单卡前一轮7.17分钟；首轮实测约2.17倍。32通道、原生160×208×160及原972/108载荷SHA不变；预取增强逐值核对通过。首轮早期分数不代表最终提分。原24小时截止不变，无新270/outer/confirmation。后续与固定历史w32参考作描述性比较，不能分离种子/执行差异与算法贡献。
[双卡启动核验](archive/research_records/2026-10-28_race_seg/DDP_STARTUP_VERIFIED.json) · [实时状态](archive/research_records/2026-10-28_race_seg/PIPELINE_STATUS.json) · [接续说明](archive/research_records/2026-10-28_race_seg/RUNBOOK.md)。旧173轮启动和中间修订保留为历史，最终通过审计的汇总将另行存档。
<!-- END_RACE_PREP_20261028 -->
