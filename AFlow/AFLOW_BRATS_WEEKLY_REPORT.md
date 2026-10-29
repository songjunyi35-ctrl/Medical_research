# AFlow BraTS研究汇报

<!-- RACE164_TEST270_20261029 -->
## 2026-10-29：RACE164叠加既有最佳推理已完成

原270/119，RACE原始Scan 0.783374，固定8镜像＋解码＋后处理Scan 0.789263、Subject 0.775438、Point 256/270；同权重差+0.005889，描述性95%区间[0.0036050984607035896, 0.008361956075480034]。训练0、2160病例前向，两个输出共用8视图。继承test-tuned历史，无当前新搜索或outer/confirmation。本轮结束不自动重测或续训。[结果](archive/research_records/2026-10-29_race_e164_inference_test270/RESULT.md)。


相对旧w32最佳推理，Scan点估计提高0.004881，但Point由260降至256；本次并非各指标全胜，也未超过历史nnU-Net单模型0.811283/融合0.812965。
<!-- END_RACE164_TEST270_20261029 -->

<!-- AUTO_INFERENCE12H_20261028 -->
## 2026-10-29：12小时无训练推理研究完成

本轮至少12小时自动推理研究已完成。固定w32第171轮/T1ce/原270/119，网络权重训练0。最高Scan **0.784382113**、Subject **0.771378811**、Point **260/270**，8次前向；对原始差+0.011154073。完成7,896,659有限全局元组，全部test-tuned，选择后区间未校正搜索，无独立泛化确认。270缓存掩膜审计及API神经复核范围见[最终报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/RESULT.md)。不自动扩搜或重启旧实验，独立RACE不受影响。
<!-- END_AUTO_INFERENCE12H_20261028 -->

<!-- INFERENCE_SELECTED_TEST270_20261028 -->
## 2026-10-28：不训练模型的推理改进固定测试

固定同一w32第171轮、验证选定五镜像＋温度2体积解码。270/119 Scan0.780995、Subject0.768306、Point255/270；对同权重原始推理差+0.007767，配对95%区间[0.004495999901909861, 0.011791830233831117]，配对区间支持本复用队列的探索性正差。 原270/119、单T1ce、原生SHA不变，训练0，每例5前向；复用内部探索无本次test调参，非独立泛化，无新outer/confirmation，评估结束不自动重测。[报告](archive/research_records/2026-10-28_segresnet_inference_selected_test270/RESULT.md)。
<!-- END_INFERENCE_SELECTED_TEST270_20261028 -->

<!-- INFERENCE_SEARCH_20261028 -->
## 2026-10-28：不训练模型的推理搜索

固定同一32通道171轮SegResNet，单A10080GB完成32037项配置。108验证原始Scan0.753271→镜像概率融合＋体积解码0.761779（+0.008508，患者配对描述性95%区间[+0.001880,+0.015552]），Subject0.750409，Point96/108；按患者分折选配置为0.761204。增加的是5次单模型前向与解码，训练0；8例输出已独立逐体素复现。属于大量搜索后的复用验证证据，分折也非完全嵌套，不称独立泛化；无新270测试，旧测试0.773228不能加验证增量。40视图/直接滑窗未胜出，负结果完整保留。[报告](archive/research_records/2026-10-28_segresnet_inference_search/RESULT.md)。
<!-- END_INFERENCE_SEARCH_20261028 -->

<!-- WIDTH32_TEST270_20261028 -->
## 2026-10-28：宽模型固定测试

固定171轮270评测完成：Scan0.773228、Subject0.760027、Point256/270；对多尺度166差+0.001842，95%CI[-0.003712439459196132, 0.007115013465642496]。配对区间跨0，没有确认稳定增益或等效。[结果](archive/research_records/2026-10-28_segresnet_w32_e171_test270/RESULT.md)。 仅108验证选定的32通道171轮模型，270扫描/119患者内部复用探索；固定阈值/预处理，无本次测试调参，不称独立泛化。原验证门未过。[材料](archive/research_records/2026-10-28_segresnet_w32_e171_test270/IDEA.md)。
<!-- END_WIDTH32_TEST270_20261028 -->

<!-- WIDTH_RESUME_20261027 -->
## 2026-10-27：容量对照继续至完整预算

两组各200轮完成；w16最佳166轮108 Scan0.747497；w32最佳171轮108 Scan0.753276；32−16差+0.005779，收益门False。[结果](archive/research_records/2026-10-27_segresnet_width_resume/RESULT.md)。 用户在前50筛选停止后授权接续85/50轮checkpoint，继续累计200轮；不重置优化器/LR。自适应预算修订如实披露，旧负筛选结果保留。无新测试。[方案](archive/research_records/2026-10-27_segresnet_width_resume/IDEA.md)。
<!-- END_WIDTH_RESUME_20261027 -->

<!-- WIDTH_20261027 -->
## 2026-10-27：先验证网络容量

用户授权前50轮筛选后停止：w16前50最好0.710403、实际结束85轮；w32前50最好0.700533、实际结束50轮。32未超过同期16，撤销剩余200轮预算，不是完整训练终态；不证明32通道普遍无效。[结果](archive/research_records/2026-10-27_segresnet_width/RESULT.md)。 两组均从头训练，避免16通道旧预训练与32通道随机起点混比。加宽是基线容量研究，不包装为新算法；原972/108，未访问新test。[方案](archive/research_records/2026-10-27_segresnet_width/IDEA.md)。
<!-- END_WIDTH_20261027 -->

<!-- SEGRESNET_REVIEW_20261027 -->
## 2026-10-27：失败复核与文献依据

蒸馏未取得收益后，对照SegResNet原论文与MONAI官方配置：当前使用合法16通道轻量实现，而原文采用32通道及训练期重建分支。容量差是待检验假设，不能作为已证实失败原因；普通/加权蒸馏的人工标签训练目标均未改善。下一步优先建立可归因的容量与配方对照，增强基线不冒充原创算法。本轮仅文献、日志与CPU参数核对，无新训练/测试。[材料](archive/reviews/2026-10-27_segresnet_literature_audit/RESULT.json)。
<!-- END_SEGRESNET_REVIEW_20261027 -->

<!-- DISTILL_20261026 -->
## 2026-10-26：强教师指导小模型

三组各200轮完成；control最佳189轮108 Scan0.745119；plain最佳183轮108 Scan0.743886；reliable最佳158轮108 Scan0.738286。明确收益门False。[结果](archive/research_records/2026-10-26_segresnet_teacher_distill/RESULT.md)。 比较同起点普通学生、普通蒸馏与可靠性加权蒸馏，避免把教师0.81直接称为学生成绩。当前仅原972训练与108验证；既往test-informed身份不变。[方案](archive/research_records/2026-10-26_segresnet_teacher_distill/IDEA.md)。
<!-- END_DISTILL_20261026 -->

2026-10-22从头训练B组complete：完成550/550轮，34100次更新尝试、34091次实际更新，选525轮；108/50验证Scan0.771742492、Subject0.768001627、Point104/108。对原e525差-0.012929153，患者配对95%区间[-0.019600229,-0.006408284]，明确收益门=False。 无新270/outer/confirmation。[结果](archive/research_records/2026-10-22_nnunet_local_full550/RESULT.md)






2026-10-21两个nnU-Net算法组complete。A体积预算竞争：完成150/150轮，尝试37500、实际更新37488；选新增0轮，108 Scan0.784664851、Subject0.780815550、Point105/108。对旧e525 Scan差-0.000006794，患者配对97.5%区间[-0.000023814,+0.000001814]，明确收益门=False。 B局部视图一致性：完成150/150轮，尝试37500、实际更新37488；选新增120轮，108 Scan0.785393948、Subject0.780034644、Point107/108。对旧e525 Scan差+0.000722304，患者配对97.5%区间[-0.002742044,+0.003888593]，明确收益门=False。 本批无新270/outer/confirmation；普通微调对照已取消，算法与额外训练的贡献未被单独识别。 [结果](archive/research_records/2026-10-21_nnunet_error_balance/RESULT.md)

2026-10-21 18:26阶段更新：A/B原进程持续训练，V5汇总器17:08 I/O异常后由V6附着接管，未重载模型、不增加恢复次数。18:25 A122轮、最好验证0.783914；B61轮、60轮验证0.784031，均未超过起点0.784665，暂无算法增益证据。预算150轮与原截止不变，无新270/outer/confirmation。[监控接管](archive/research_records/2026-10-21_nnunet_error_balance/AMENDMENT_005_MONITOR_ATTACH.md)。

2026-10-21旧缓存清理完成：按用户授权删除12个旧输入载荷，持久盘释放14.21875GiB、空闲约37.44GiB；另删除主节点/tmp旧公开334缓存3.26171875GiB。构建脚本、索引/历史SHA、原始源数据、全部模型及报告保留；检查两组训练进程/子进程共14个，确认未用所删缓存，训练继续。清理本身无数组解码或新模型/评分执行。[清理与重建入口](archive/reviews/2026-10-21_unused_cache_cleanup/RESULT.md)。

2026-10-21 15:20磁盘保护后接续：A44轮、B19轮后曾停止；公开334未用派生缓存完整校验备份后转移释放3.2617GiB，空闲约23.2GiB。V5从原完整checkpoint接续A45/B20轮，各150轮与原配方不变；每组为第2次且最后一次工程恢复。当前训练后验证最高A0.783914/B0.781598，尚未超过起点0.784665，无新270/outer/confirmation。[磁盘恢复记录](archive/research_records/2026-10-21_nnunet_error_balance/AMENDMENT_004_DISK_RESERVE_RECOVERY.md)。

2026-10-21两个nnU-Net算法组incomplete。A体积预算竞争：完成44/150轮，尝试11000、实际更新10997；选新增0轮，108 Scan0.784664851、Subject0.780815550、Point105/108。对旧e525 Scan差-0.000006794，患者配对97.5%区间[-0.000023814,+0.000001814]，明确收益门=False。 B局部视图一致性：完成19/150轮，尝试4750、实际更新4750；选新增0轮，108 Scan0.784664851、Subject0.780815550、Point105/108。对旧e525 Scan差-0.000006794，患者配对97.5%区间[-0.000023814,+0.000001814]，明确收益门=False。 本批无新270/outer/confirmation；普通微调对照已取消，算法与额外训练的贡献未被单独识别。 [结果](archive/research_records/2026-10-21_nnunet_error_balance/RESULT.md)

2026-10-21 12:33工程接续：原972/108 SHA及几何门通过，两算法正式冻结150轮。修复CPU权重导出重复共享参数的问题，逐张量完全一致，原失败阶段已归档；A从完整8轮/2000更新接续第9轮，B开始第1轮，预算/配方不变，无新270/outer/confirmation。[接续记录](archive/research_records/2026-10-21_nnunet_error_balance/AMENDMENT_003_STORAGE_RECOVERY.md)。

2026-10-21两个nnU-Net算法组incomplete。A体积预算竞争：完成8/150轮，尝试2000、实际更新2000；选新增0轮，108 Scan0.784664851、Subject0.780815550、Point105/108。对旧e525 Scan差-0.000006794，患者配对97.5%区间[-0.000023814,+0.000001814]，明确收益门=False。 B局部视图一致性：没有产生可选择模型。 本批无新270/outer/confirmation；普通微调对照已取消，算法与额外训练的贡献未被单独识别。 [结果](archive/research_records/2026-10-21_nnunet_error_balance/RESULT.md)

2026-10-21用户修订：取消普通微调control（患者训练/模型访问0），改为A困难背景竞争与B局部视图一致性两算法、各一张A10080GB。两组分别比较已有e525，额外训练贡献未被单独隔离；V3准备中，无新增Dice或270访问。[最新方案](archive/research_records/2026-10-21_nnunet_error_balance/AMENDMENT_002_TWO_ALGORITHMS.md)。

2026-10-21新增nnU-Net算法候选：用户提供worker1143438双A10080GB、约24小时。旧e525最小WT体积四分位验证Dice0.684294，拟检验体积预算困难背景竞争项；同e525初始化的普通微调作为同预算对照。CPU5项与两组GPU合成门通过，约29.06GiB/卡、算法训练开销约5%；正在重建972/108缓存，尚无新患者模型训练/验证，无新270/outer/confirmation。算法效果及创新性待验证。[方案](archive/research_records/2026-10-21_nnunet_error_balance/IDEA.md)。

<!-- NNUNET_RESENC_20261013 -->
**2026-10-14新增结果：** 550轮完成，108选525轮Scan0.784672，验证门通过。270单模型Scan0.811283，最佳三模型融合Scan0.812965、Subject0.797512、Point266/270；相对旧榜首+0.037074，95%CI[+0.030140,+0.044262]。达到本批0.8均值目标，全部test-tuned，非独立泛化。 双卡训练约15.47小时，单模型较大且采用滑窗，最佳融合需要3个模型；不能将收益只归于架构。[报告](archive/research_records/2026-10-13_nnunet_resenc_reference/RESULT.md)。
<!-- END_NNUNET_RESENC_20261013 -->

<!-- VALID_REGION_20261010 -->
**2026-10-10有效区域监督：完成200轮。** worker1127648双A10040GB，单组原972/108、单T1ce原生多尺度SegResNet；当前完成200/200轮，唯一变化为几何有效区域损失，global8/每轮122更新。选175轮108 Scan0.746077、Subject0.734457、Point98/108；对旧多尺度166差-0.007015、95%CI[-0.023860,+0.003203]。未达到事前明确验证收益门；只评价当前固定配方，不证明所有有效区域策略无效。无本候选新270/outer/confirmation访问，test-informed开发。[真实状态](archive/research_records/2026-10-10_segresnet_valid_region_supervision/PIPELINE_STATUS.json) · [方案与报告](archive/research_records/2026-10-10_segresnet_valid_region_supervision/RESULT.md)。
<!-- END_VALID_REGION_20261010 -->

<!-- NEXT_IDEAS_20261010 -->
**2026-10-10后续候选：** 两张A10040GB条件下，拟优先检验训练损失排除人工补零，再考虑公开334在微调期间低权重参与；现代nnU-Net残差编码器列为较大预算参考。原972补零比例中位数35.92%，不等于已证明影响Dice。此轮仅文献和方案，无新训练/精度/270访问。[方案审视](archive/research_records/2026-10-10_two_a100_next_ideas/REVIEW.md)。
<!-- END_NEXT_IDEAS_20261010 -->

<!-- MULTISCALE_E166_TEST270_20260929 -->
**2026-09-29多尺度166固定270评测完成：** 原270扫描/119患者，native1mm裁剪栅格排除填充，Scan 0.771386、Subject 0.757817、Point 248/270。对同格旧组合166差+0.005115、患者配对95%CI[+0.000590,+0.009713]；复用队列的95%配对区间为正，但97.5%区间仍跨0；这是小幅收益的探索性证据，尚不能认定稳健或独立泛化提升。原验证门未过，test-tuned开发/复用内部探索；此次唯一166模型已访问270，无当前test调参或新outer/confirmation。评测与归档结束，不自动重测或新训。[报告](archive/research_records/2026-09-29_segresnet_multiscale_e166_test270/RESULT.md)。
<!-- END_MULTISCALE_E166_TEST270_20260929 -->

<!-- MULTISCALE_RESUME_20260929 -->
**2026-09-29多尺度辅助监督续训终态：** 完成200轮；108扫描/50患者原生1mm裁剪栅格：选第166轮，Scan 0.753092、Subject 0.739669、Point 94/108；对旧公开预训练＋原生组合166同格Scan差+0.001913，50患者配对95%CI[-0.004999,+0.010082]。 冻结提升门未通过。训练/108验证归档，训练阶段无新270模型评分；后续166固定270评测已完成，无新外部确认，见最新条目；最高旧test-tuned融合仍0.775891。[报告](archive/research_records/2026-09-29_segresnet_native_multiscale_resume50/RESULT.md)。
<!-- END_MULTISCALE_RESUME_20260929 -->

<!-- MULTISCALE_20260924 -->
**2026-09-24多尺度辅助监督终态：** 按50轮规则停止；108扫描/50患者原生1mm裁剪栅格：选第49轮，Scan 0.718759、Subject 0.708114、Point 101/108；对旧公开预训练＋原生组合166同格Scan差-0.032420，50患者配对95%CI[-0.040790,-0.024388]。 冻结提升门未通过。训练/108验证归档，无新270模型评分或外部确认；最高旧test-tuned融合仍0.775891。[报告](archive/research_records/2026-09-24_segresnet_native_multiscale_supervision/RESULT.md)。
<!-- END_MULTISCALE_20260924 -->

**2026-09-24目标域数据审视：** 用户授权直接比较270测试和原972训练数据；128³缓存的14项固定影像特征区分AUC0.554，肿瘤体积与标签碎片/边缘负担没有显示值得直接调训练集的明显差异。此前目标域病例加权验证未获益；本批不新开训练。270图像与标签已用于test-tuned诊断，无新模型前向、精度或外部确认。[审视报告](archive/reviews/2026-09-24_test270_image_profile/RESULT.md)。

**2026-09-24终态：** worker1109780双A100正常；原972训练标签的病灶均衡选块诊断完成。预设小区域覆盖只提高0.842个百分点，低于5个百分点门，因此不启动训练。这里是窗口覆盖而非Dice；小区域实际为1–5体素标签区域，不直接等同临床小病灶。没有新模型或精度结果、无新验证/test；旧最高270定向结果仍0.775891，非独立泛化。[最终报告](archive/research_records/2026-09-24_segresnet_lesion_balanced_patch_proposal/RESULT.md)。

更新：2026-09-21。本次组合与患者等权两组均完成200轮、108验证和用户追加授权的一次270测试；结果见下，其他授权批次单独记录。

**新增固定评测结果：** 本次两个固定候选各一次270扫描/119患者测试均已完成。组合166在native grid的Scan 0.766272、Subject 0.750958、Point 236/270；对原native173差-0.002306、95%CI[-0.009078,+0.005791]。患者等权172在128³的Scan 0.759967、Subject 0.745647、Point 250/270；对预训练179差-0.001682、95%CI[-0.008389,+0.005860]。不同栅格不直接相减。验证门未过的事实保留；本次为用户授权、复用内部队列探索，无本批test调优或新outer/confirmation。[组合测试](archive/research_records/2026-09-21_segresnet_pretrain_native_e166_test270/RESULT.md) · [患者等权测试](archive/research_records/2026-09-21_segresnet_patient_equal_e172_test270/RESULT.md)。

**2026-09-20原始栅格结果：** 原始分辨率（不缩放到128³）纯SegResNet第173轮test270/119在原始1mm栅格上Scan 0.768578、Subject 0.754144；原193基线重采样到同栅格为0.761134，配对差+0.007444、95%CI[-0.0006,+0.0143]，配对区间跨0，不确定。不能与128³栅格的0.759179直接相减。分辨率因素已量化并关闭。[完整结果](archive/research_records/2026-09-20_segresnet_native_res_e173_test270/RESULT.md)。
## 当前结论

固定原972扫描/444患者训练、108/50验证、270/119内部测试，纯SegResNet随机初始化、完整200轮，由验证选择第193轮。该模型仅单次全自动前向，test Scan Dice **0.759179**、Subject Dice **0.744985**；它继续作为用户固定比较基线。当前128³自动总体及单次前向Dice最高点估计为公开334例预训练后微调179：**Scan0.761649、Subject0.745859、Point246/270**。对193 Scan仅+0.002470、95%CI[−0.004343,+0.009627]跨0，验证门未过，不称决定性或稳定改进；原始栅格结果单独解释。

**原始分辨率工作的贡献定位：** 针对固定尺寸缩放可能损失空间细节的问题，已完成保持1mm裁剪、补零整脑输入的方案设计、200轮训练和同栅格对照评估。这是完整的假设驱动方法探索，可作为输入处理改进与消融证据；内部测试平均提升约0.74个百分点，但配对区间跨0，尚未确认稳定收益。训练双卡约3.9小时，输入体素量超过128³的2.5倍；量化这些成本与收益本身具有研究价值。“方向关闭”表示本轮完成并停止继续投入，不表示没有idea；实验管线已落地，默认部署仍为193基线。原创算法新颖性需另行论证。[贡献定位与依据](archive/reviews/2026-09-21_native_res_test270_completion_audit/RESEARCH_POSITIONING.md)。

Scan患者整簇95%区间[0.732638,0.781993]，Subject[0.718192,0.768915]。相对旧8镜像0.757270，Scan仅高0.001909，配对95%区间[−0.004567,+0.009086]跨0，不能称显著胜出。Point242/270低于旧8镜像265/270，不能称三指标全面最佳。[完整结果](archive/research_records/2026-09-17_segresnet_plain972_e193_test270/RESULT.md)。

## 研究起点及路线

最初复现的是SAM-Med3D三维医学提示分割模型，在同内部test270的五点击最佳Scan为0.594226，依赖真实标签中心裁剪和模拟点击。随后改用约470万参数SegResNet，直接输出完整肿瘤掩膜，适合无人工提示的使用方式；两种方案定位权限和输入栅格不同，不能作为同配方架构消融。

输入始终为单张增强T1，分割整体肿瘤标签1/2/3，排除切除腔4。当前自动管线使用N4强度校正、图像自身脑区裁剪到128³，固定logit阈值0，不用GT定位或提示。

## 主要里程碑（同test270/119，历史行保留）

| 阶段 | 方法与范围 | Scan Dice | 推理成本/解释 |
|---|---|---:|---|
| 8月17日 | 756训练的原始纯SegResNet | 0.725820 | 1次，全自动起点 |
| 8月18日 | 756纯模型，更充分训练与后期降低学习率 | 0.741204 | 1次；训练时程组合有益 |
| 8月24日 | 972病例池、从头训练的知识蒸馏模型 | 0.751760 | 1次；已包含额外优化 |
| 8月31日 | fit864优化器支线 | 0.752003 | 1次；曾为单次Scan纪录，非full972 final |
| 9月4日 | 972蒸馏模型＋8镜像预测平均 | 0.757270 | 8次；旧总体Dice纪录，当前Point仍最高 |
| **9月17日** | **原972完整覆盖、纯SegResNet第193轮** | **0.759179** | **1次；当时128³ Dice纪录，现固定对照** |
| **9月21日** | **公开334例预训练→原972微调179** | **0.761649** | **1次；128³新点估计纪录，对193区间跨0，未过提升门** |

全自动起点至当前128³点估计纪录差约0.035829，即3.583个百分点。是整条研究路线的历史进步，不能全部归于某个模块；新旧训练病例池、抽样和选优不同。

## 固定纯193基线的训练与成本

完整200轮，每轮972个不同扫描、122次更新，共24400更新；每轮固定108验证，选193轮验证Scan0.740032、Subject0.731538。最新200轮验证Scan0.737539；best/latest严格重载逐病例结果一致。

两张A100，每卡4例、全局8；Dice+0.1交叉熵，前100轮lr2e-4，后100轮余弦至1e-5。没有蒸馏、灰度项、中心头、EMA、融合、镜像或后处理。为满足用户当前要求，旧加权有放回采样改为原972各一次，不丢尾批。

训练墙钟约1小时43分、双卡约3.44分配GPU小时，峰值约10.1GiB/卡。193轮270例GPU推理约8.72秒，包含缓存hash与加载约17.92秒；这不含N4预处理构建时间。单模型权重约18MiB，所有训练快照约449MiB，原缓存复用。

## 灰度研究的负结果与复现边界

- SAM灰度均值匹配组按用户指令停止161/200轮。test270第50轮0.566424，第161轮0.515107，均低于旧交互基线0.594226。
- 三维Mumford–Shah组停止45/200轮；仅108验证best5=0.571493，与旧同验证0.571306基本持平，未测test。
- 多agent审计确认这些是论文公式的本地适配，不是完整作者复现。旧SAM基线与灰度组实际更新次数、loss系数等不一致，不能把下降全部归因灰度项。
- 负结果、停止轮数、权重和访问历史全部保留；不恢复旧灰度组或关闭的684/180支线。[审计](archive/reviews/2026-09-17_graymean_reproduction_audit/RESULT.md)。

## 证据强度与局限

所有270结果都是多次使用的内部测试，不是新盲测、外部确认或官方挑战隐藏榜。本候选在本次测试前由验证选好并固定，用户授权后只测一次；但研究方向此前已受到历史测试揭示影响。

相对历史纯168、Full972蒸馏和优化器支线，预先锁定的配对差值区间偏正；这些是描述性多重比较，未做多重性校正，且不能分离数据/采样/训练时程影响。相对旧8镜像差值CI跨0，不能将点估计最高包装成显著优越。

既有外部队列的历史结果继续保留在排行榜，不能替代对新193模型的外部验证；本次没有新outer/confirmation/外部访问。Kaggle镜像与官方corrected-v2源字节一致性尚未完成的来源限制仍保留。

## 下一步

本次两个固定候选的评测和证据已归档，不据本批结果自动改参数、换epoch或重复测试；其他另行授权的研究批次按独立协议执行。固定193仍为比较基线。

[当前交接](AFLOW_BRATS_HANDOFF_CURRENT.md) · [连续记录](AFLOW_BRATS_RESULTS.md) · [方法目录](AFLOW_BRATS_METHOD_CATALOG.md) · [精确排行榜](AFLOW_BRATS_LEADERBOARD.md)。此前完整汇报已保存在[本次更新前归档](archive/research_records/2026-09-17_segresnet_plain972_e193_test270/WEEKLY_BEFORE_E193_CLEANUP.md)，历史证据未删除。

2026-09-18优化迁移：按用户指定统一以新972纯193作基线。旧八镜像方法在当前108验证Scan仅+0.001444、差值CI跨0，Point97降至95，未通过效用门；不因旧方法曾有效就自动采用8倍推理，没有重跑270测试。当前1×基线与test0.759179保持不变。[迁移结果](archive/research_records/2026-09-18_plain972_e193_flip8_validation/RESULT.md)。

2026-09-18补记：固定八镜像test在执行前由用户取消，无新增测试分数。后续只讨论纯SegResNet的信息与训练瓶颈：现有强度/裁剪/标准化已具备，优先诊断低分验证病例；高分辨率分块或多MRI序列尚未运行，不承诺涨幅。[依据与边界](archive/reviews/2026-09-18_segresnet_improvement_options/RESULT.md)。

用户随后确认训练和部署均保持单T1ce，多序列候选退出当前方向；当前仍无新训练或测试。

2026-09-18最新：用户授权一组无额外N4的单T1ce纯SegResNet基础对照，保持972/108/270和原200轮配方，仅取消N4。原0.759179尚不能解释为N4创新增益。项目按授权删除SAM权重与不用派生缓存约38GiB，保留全部科研结果；不自动测试。[新组](archive/research_records/2026-09-18_segresnet_plain972_no_n4/RESULT.md)。

用户随后将无N4组改为第50轮观察：没有明确同轮差异就停止，不无条件跑满200轮；该规则已执行，50轮结论见下。

无N4基础对照按用户省时规则停止于50/200轮；同轮无N4−N4验证差+0.005645、CI[-0.006018021430237807, 0.018625530739219822]，未达明确差异门，不是等效证明，无新test。[终态](archive/research_records/2026-09-18_segresnet_plain972_no_n4/RESULT.md)。

原始分辨率整脑纯SegResNet（不缩放到128³）200轮完成，最佳验证原始栅格Scan=0.746102，同栅格配对差值（新−原193）Scan=+0.003422、95%CI=[-0.0008976655242559269, 0.007705410726704575]，无新增test。[最终结果](archive/research_records/2026-09-18_segresnet_plain972_native_res/FINAL_RESULT.json)。

2026-09-21下一轮：用户允许最多两个SegResNet实验，选定“公开334例单T1ce监督预训练后微调”和“前景语义分解后合成WT”，各拟用两张A100。分别检验学习经验与监督表达，当前只是协议草案，没有承诺大幅提升。worker1104813缺nvidia_uvm、CUDA返回999，正式训练未启动；实现和执行冻结待完成。用户新增test定向优化许可尚未使用，若使用须单列适配成绩。本次没有新精度或test/outer/confirmation像素访问。[七问方案与状态](archive/reviews/2026-09-21_segresnet_next_two_ideas/RESULT.md)。

2026-09-21节点更换后续：用户提供worker1104883（2×A100），CUDA张量检查通过；两候选改为顺序运行、各自全局batch8不变。新代码已实现、6项CPU检查通过、36文件冻结，监督入口已启动，正式训练仍须通过合成与数据门。[统一执行协议](archive/research_records/2026-09-21_segresnet_two_experiments/IDEA.md)。

<!-- TWO_IDEAS_PIPELINE -->
两候选均已完成：语义分解未提分；公开预训练仅小幅点估计提升，未达到预设明确增益门。本轮停止，不自动重训、换epoch或重复测试。[最终总状态](archive/research_records/2026-09-21_segresnet_two_experiments/LIVE_STATUS.json)。
<!-- END_TWO_IDEAS_PIPELINE -->

2026-09-21正式执行补记：修正训练清单字段名接口后，972例类别重建和WT逐字节核对全部通过，两组双卡合成检查通过。语义组已进入200轮正式训练；公开334对数据668文件下载与SHA校验完成，CPU缓存准备后按队列预训练/微调。没有新test/outer/confirmation访问，尚不能判断提分。[启动核验](archive/research_records/2026-09-21_segresnet_two_experiments/STARTUP_VERIFICATION.json)。

2026-09-21单卡并行补记：用户提供1105022单A100且CUDA正常。334例预训练缓存跨节点SHA完全一致；10项CPU检查及单卡batch8/尾批6合成检查通过，峰值约20.16GiB。第二组公开预训练已在新单卡启动，与1104883语义组并行；原等候器已停止防重。150轮预训练后仍以原双卡配方完成972微调200轮。全局batch8/更新次数保持，但预训练随机数与归约轨迹变更已事前记录，无当前test调优或新test/outer/confirmation访问。[并行启动核验](archive/research_records/2026-09-21_segresnet_two_experiments/PARALLEL_START_VERIFIED.json)。

<!-- TWO_IDEAS_SEMANTIC_REGIONS -->
前景语义分解SegResNet：200轮完成，选第158轮；108验证Scan 0.739689，未达到预设验证提升门。270/119同128³测试Scan 0.757718、Subject 0.743434、Point 238/270；对193差-0.001461、95%CI[-0.006823,+0.003567]，差值区间跨0。单模型1次推理；无当前test调优或新outer/confirmation。[结果](archive/research_records/2026-09-21_segresnet_semantic_regions/RESULT.md)。
<!-- END_TWO_IDEAS_SEMANTIC_REGIONS -->

<!-- TWO_IDEAS_FINETUNE_PRETRAINED -->
公开334例预训练后微调SegResNet：200轮完成，选第179轮；108验证Scan 0.746851，未达到预设验证提升门。270/119同128³测试Scan 0.761649、Subject 0.745859、Point 246/270；对193差+0.002470、95%CI[-0.004343,+0.009627]，差值区间跨0。单模型1次推理；无当前test调优或新outer/confirmation。[结果](archive/research_records/2026-09-21_segresnet_external_pretrain/RESULT.md)。
<!-- END_TWO_IDEAS_FINETUNE_PRETRAINED -->

两组总结：语义分解0.757718未超过193；公开预训练0.761649仅小幅正向且区间跨0，没有达到本轮希望的明确提升。公开预训练单卡39分钟、微调双卡约1小时42分钟，部署仍1次前向/约18MiB；额外数据和额外计算不可由本组单独拆开归因。当前均结束，无自动后续训练、调参或测试。

<!-- PRETRAIN_NATIVE_COMBINATION -->
训练/108验证阶段已完成200轮，选第166轮，验证门未过；后续用户授权test270也已完成。旧训练JSON中的无test标记仅指训练阶段。[本候选测试与访问补记](archive/research_records/2026-09-21_segresnet_pretrain_native_e166_test270/RESULT.md)。
<!-- END_PRETRAIN_NATIVE_COMBINATION -->

<!-- PATIENT_EQUAL_LOSS -->
训练/108验证阶段已完成200轮，选第172轮，验证门未过；后续用户授权test270也已完成。旧训练JSON中的无test标记仅指训练阶段。[本候选测试与访问补记](archive/research_records/2026-09-21_segresnet_patient_equal_e172_test270/RESULT.md)。
<!-- END_PATIENT_EQUAL_LOSS -->

本批主要研究因素：组合在原native配方中改用公开预训练初始化；患者等权在预训练128³配方中改每例损失系数。均保持单T1ce、SegResNet和原972/108/270；训练分别约双卡3小时49分、单卡2小时39分，部署仍单次前向。

**本批测试结论：** 本次两个固定候选各一次270扫描/119患者测试均已完成。组合166在native grid的Scan 0.766272、Subject 0.750958、Point 236/270；对原native173差-0.002306、95%CI[-0.009078,+0.005791]。患者等权172在128³的Scan 0.759967、Subject 0.745647、Point 250/270；对预训练179差-0.001682、95%CI[-0.008389,+0.005860]。不同栅格不直接相减。验证门未过的事实保留；本次为用户授权、复用内部队列探索，无本批test调优或新outer/confirmation。[组合测试](archive/research_records/2026-09-21_segresnet_pretrain_native_e166_test270/RESULT.md) · [患者等权测试](archive/research_records/2026-09-21_segresnet_patient_equal_e172_test270/RESULT.md)。


<!-- TEST270_10H -->
10小时单T1ce定向优化：complete。270/119原始栅格总体Scan 0.775891、Subject 0.761694、Point 261/270；最佳单模型Scan 0.772830。对最强同栅格参考差+0.007318、95%CI[0.0031155479307933965, 0.012786231698480971]，未达到0.80且至少+0.03目标。全部test-tuned，非独立泛化；270不进分割梯度，无outer/confirmation。[报告](archive/research_records/2026-09-21_t1ce_test270_10h/RESULT.md)。
<!-- END_TEST270_10H -->

<!-- FOLLOWUP_TEST_PATIENT -->
预训练＋患者等权损失172：test270/119完成，128³栅格，Scan 0.759967、Subject 0.745647、Point 250/270；对pretrained179差-0.001682、95%CI[-0.008389,+0.005860]。验证门未过，用户授权探索性测试；无当前test调优或新outer/confirmation。[完整报告](archive/research_records/2026-09-21_segresnet_patient_equal_e172_test270/RESULT.md)。
<!-- END_FOLLOWUP_TEST_PATIENT -->

<!-- FOLLOWUP_TEST_NATIVE -->
公开预训练＋原分辨率组合166：test270/119完成，原始1mm裁剪栅格（native grid），排除填充，Scan 0.766272、Subject 0.750958、Point 236/270；对native173差-0.002306、95%CI[-0.009078,+0.005791]。验证门未过，用户授权探索性测试；无当前test调优或新outer/confirmation。[完整报告](archive/research_records/2026-09-21_segresnet_pretrain_native_e166_test270/RESULT.md)。
<!-- END_FOLLOWUP_TEST_NATIVE -->

<!-- RACE_PREP_20261028 -->
## 2026-10-28 RACE-Seg双卡算法终态

RACE-Seg双卡单算法complete，累计200轮，最佳164轮108 Scan 0.763266。历史参考差+0.009990，描述性区间[0.0009791717071146928, 0.021314055377539866]；历史提升门True。同期对照已由用户取消，不能单独确认算法因果收益。无新270/outer/confirmation，本轮结束不自动恢复。[结果](archive/research_records/2026-10-28_race_seg/RESULT.md)。
<!-- END_RACE_PREP_20261028 -->
