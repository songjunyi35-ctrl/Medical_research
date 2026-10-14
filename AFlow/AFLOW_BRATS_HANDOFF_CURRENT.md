# 当前科研状态与接续边界

<!-- NNUNET_RESENC_20261013 -->
**2026-10-14残差编码器组已完成训练、验证、270及六项融合。** 550轮完成，108选525轮Scan0.784672，验证门通过。270单模型Scan0.811283，最佳三模型融合Scan0.812965、Subject0.797512、Point266/270；相对旧榜首+0.037074，95%CI[+0.030140,+0.044262]。达到本批0.8均值目标，全部test-tuned，非独立泛化。 训练节点1132660，评测接续1133854。V3结果与独立复核已归档；旧COORDINATOR_FAILURE描述历史归档失败。本组结束，不自动重训、扩组、换epoch或重测，无新outer/confirmation。[最终报告](archive/research_records/2026-10-13_nnunet_resenc_reference/RESULT.md) · [真实终态](archive/research_records/2026-10-13_nnunet_resenc_reference/PIPELINE_STATUS.json)。
<!-- END_NNUNET_RESENC_20261013 -->

同日的[候选筛选与纯合成探测](archive/research_records/2026-10-13_dice080_proposal/IDEA_strong_recipe.md)已转入上方正式运行；旧“训练0轮”仅描述准备阶段。旧0.775891仍是已有270榜首，本组尚无新270分数。首次合成重载接口问题已在患者模型访问前通过V2入口修复，旧COORDINATOR_FAILURE只描述失败的V1合成阶段；当前状态看新监督心跳，不重启旧入口。

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
