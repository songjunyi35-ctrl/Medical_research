# AFlow BraTS 方法百科（小白版）

更新时间：2026-09-10 UTC

## 患者模型固定八镜像 🧪 工程通过，固定双seed开发读出中

**问题/直觉：** 患者模型1× dev Scan0.753475/0.755906但Point167/168，异质等权融合没稳定超过强成员。只让同一强模型从八个镜像方向看并还原平均，避免混弱模型。

**流程/控制：** 固定两seed subject_balanced epoch168、fit684/313、原hard128/阈值0/argmax/batch8/BF16；唯一变化1×→完整8镜像同顺序raw-logit均值，0训练。历史项目Flip8提供依据，但不保证新权重，不继承“独立确认”说法。

**门/成本/下一步：** 各seed两Dice>0、Point差≥−1、两簇CI下界≥−.003、平均Scan≥+.003，且Point≥169/162；失败不扫子集。开发2880forwards/预计几分钟，部署同4.70M/8×。若全门过，事前seed11单test面板2160forwards两个输出全报，仍最多一次。见[七问](archive/research_records/2026-09-10_subject_fixed_flip8/IDEA.md)、[预注册](archive/research_records/2026-09-10_subject_fixed_flip8/PREREG.md)。

**10:47工程/读出：** fit8×两模型×8共128forward smoke通过，3.15s/peak5.050GiB；已启动唯一双seed dev2880forward。先检查identity逐例与原source raw exact，非权重修改；尚无本应用收益结论。见[smoke](archive/research_records/2026-09-10_subject_fixed_flip8/SMOKE_RECEIPT.json)。

## 数据配方异质模型等权融合 ❌ 未稳定超过强成员，固定组合关闭

**最终10:42：** 融合seed11/12 Scan0.754116/0.750449，Subject0.740313/0.736772，Point174/167。对native两seedDice/Point都正且DiceCI正，但对sampling强成员Scan+.000642/−.005457、Subject−.000320/−.005952；第二seed两CI负，未超强成员门。不能把借来的采样收益称融合成功，固定组合及test分支关闭，不调weight。0训练、720forwards/17.06s、4.849GiB，部署2×；无test/保护队列。见[结果](archive/research_records/2026-09-10_data_recipe_ensemble/RESULT.md)、[验证](archive/research_records/2026-09-10_data_recipe_ensemble/VERIFICATION_RECEIPT.json)。以下阶段记录保留历史。

**问题/直觉：** 患者均衡dev Dice较高（0.753475/0.755906，Point167/168），占比标签Point较高（175/167，但Dice0.746659/0.739852）；可能互补，但没有融合证据。两原配方保持失败关闭。

**做法/单因素：** 同seed两固定168模型的raw logits各半平均，原0阈值/argmax/硬GT/输入保持，不拼指标、无标签路由、无重训。仅一个权重组合，两seed都检验；这次是开发揭示后新提案、不是原始盲计划。nnU-Net官方验证配置组合提供范式，旧同配方多seed负结果不是本异质组合的直接验证。

**门/成本/下一步：** 每seed融合两Dice严格超两成员最大值、Point≥最佳成员−1，对native Subject≥+.003/Scan>0/两seed平均Scan≥+.003；对两成员Dice簇CI下界≥−.003。失败不扫权重/成员。0训练、dev720forwards，部署2×或16×，绝不称1×。先合成/fit8 smoke再冻结一次dev；若全门过仍最多一个test panel。见[七问](archive/research_records/2026-09-10_data_recipe_ensemble/IDEA.md)、[预注册](archive/research_records/2026-09-10_data_recipe_ensemble/PREREG.md)。

**10:38启动：** 固定等权公式2项合成测试通过；fit8×4模型32forwards smoke通过，peak4.849GiB/1.66s，无新训练。现唯一720forwards dev评测启动，逐例重现两个成员历史raw后才接受融合结果，不换成员/权重。见[工程收据](archive/research_records/2026-09-10_data_recipe_ensemble/SMOKE_RECEIPT.json)。

## 训练标签精确体积占比 ❌ 定位改善但分割未提高，原配方/单模test关闭

**最终10:31：** seed11/12 Scan0.746659/0.739852、Subject0.732690/0.726270、Point175/167。对native Scan−.000104/−.004325，Subject−.000523/−.003048，Point+5/+4；平均Scan−.002214。第一seed两CI跨0，第二Scan上界−.00002594极近0、Subject跨0，不夸大统计稳定性。主要增益门失败，直接test分支关闭、不扫核/阈值/seed。几何守恒/定位正向得到支持，完整分割提升未支持；非完整SoftSeg结论。4.2627GPUh、20.664GiB、dev360forwards/10.99s、cache2.672GiB，0test/outer/confirmation/external。见[结果](archive/research_records/2026-09-10_fractional_occupancy/RESULT.md)、[解释](archive/research_records/2026-09-10_fractional_occupancy/INTERPRETATION.md)、[验证](archive/research_records/2026-09-10_fractional_occupancy/VERIFICATION_RECEIPT.json)。后文训练阶段记录保留历史。

**08:14备用接续：** 主弱形变负结果已独立归档，本方法fit8/batch8 GPU smoke通过（会话80451已exit0）：初值/图像/原RNG匹配、soft target实际改变梯度更新、aux梯度0、export0、显存门通过。还未正式训练/读本候选dev。原图像/损失/抽样/两seed/168轮/硬GT评分不变，不加入elastic；完整时间门与哈希收据过后才启动。

**08:15正式启动/08:18进度：** 时间预算与所有哈希准入通过，两候选seed2026090711/12并行启动（监督器会话39049，PID19199/19200），已各2/168轮。预计约2小时多，截止仍11:28:19 UTC。未读本候选dev/test，不能称提升；保留固定seed11的条件性test规则，不挑seed。见[GPU检查](archive/research_records/2026-09-10_fractional_occupancy/SMOKE_RECEIPT.json)、[训练前收据](archive/research_records/2026-09-10_fractional_occupancy/TRAIN_PREACCESS.json)、[实现冻结](archive/research_records/2026-09-10_fractional_occupancy/IMPLEMENTATION_FREEZE.json)。

**10:24正式开发读出：** 两seed168轮完成，各7673.09/7672.57秒、峰值20.664GiB、export0。全336轮抽样/原RNG/记录第一batch图像byte exact、只有target改变，独立重放通过。现在一次dev180/78×两模型360forwards，评分仍原hard128，尚无新性能结论。见[全程重放](archive/research_records/2026-09-10_fractional_occupancy/TRAINING_REPLAY_FINAL.json)。

**问题/直觉：** fit标签128³往返Dice0.965432表明边缘离散误差；每格如果30%是肿瘤，就保留0.3作为训练target，而非提前0/1。当前native dev Scan0.746762/0.744177。SoftSeg支持部分体积方向，但其20人2D肿瘤核心/改激活和回归loss不能当本单因素保证。

**做法/单因素：** 原image-only crop内WT1/2/3按三轴精确格重叠积分得到128³占比，float64计算/float16储存；只换训练标签，图像、同两seed/native抽样/初态/原DiceCE/168轮/阈值0/1×不变，无教师/新增elastic/后处理。开发仍旧硬GT与栅格，不用新target抬评分。

**门/状态/机制：** 先核684原label SHA、硬cache byte parity、有限[0,1]/质量守恒；只有弱形变无合法晋级资格且时间足够才训练两候选。每seed原Subject≥+.003、Scan>0、Point≥−1、两簇CI下界≥−.003、平均Scan≥+.003。失败关闭当前target+原loss组合；无新模型/指标，未支持直觉。若双seed过门，事前固定seed11（非选最好seed）才能另冻结一次同test270，标fit684而非full864；整个session最多一个最终test panel。

**成本/下一步/证据：** CPU先建2.672GiB临时label cache、不重建影像；两候选双卡预计2–2.3h、部署4.70M/1×（预定镜像时8×），主要风险是软监督未必匹配原loss/硬阈值。先合成和fit工程，不抢当前GPU。见[七问](archive/research_records/2026-09-10_fractional_occupancy/IDEA.md)、[预注册](archive/research_records/2026-09-10_fractional_occupancy/PREREG.md)。

**04:59执行：** 11项合成测试通过（独立重叠矩阵、非整数缩放、放大/缩小、常数/identity与float16量化）；仅fit684 raw标签的CPU缓存构建已启动，会话12958。不修改旧图像/标签/cache，无新增GPU训练或本候选开发性能。Stage A弱形变仍有优先权。

**05:44工程完成：** cache684/684原hard byte exact、原生label SHA匹配，逐例新target字节/统计重读exact。float64最大相对质量误差3.36e−16；float16最大4.02e−5/平均1.93e−6（旧hard平均0.008549），只说明裁剪内数据表示守恒。12000530个fractional格；>0.5硬化无整例空WT，与旧hard掩膜均值Dice0.973320，非模型分数/上限。公共crop外4体素仍记录、不排例。2356.10秒CPU、RSS1.094GiB、2.672GiB/tmp缓存；13项算子/损失测试+2项评估合同测试过，未训练/读新dev。见[工程结果](archive/research_records/2026-09-10_fractional_occupancy/CACHE_RESULT.md)、[cache收据](archive/research_records/2026-09-10_fractional_occupancy/CACHE_RECEIPT.json)、[验证](archive/research_records/2026-09-10_fractional_occupancy/CACHE_VERIFICATION.json)。

## 保护定位峰值的小孤岛清理 ❌ 开发无收益，固定配方关闭

**问题/直觉：** 两native旧control开发预测内假阳性比例均值22.44%/22.27%，是否为孤立块未知。清理小岛但保护最高logit位置，保留所有足够大的连通块，避免武断只留一个病灶。

**流程/单因素：** 冻结旧fit684两seed模型及dev180，原logit>0后26邻接分块，删大小<max(8,最大块1%)且不含argmax的块；被删logit设−1。模型、阈值/Point、栅格、batch8/BF16不变；无训练/教师/额外forward/交互。规则唯一，不搜索比例。

**门/状态/机制：** 双seed各Subject≥+.003、Scan>0、两簇CI下界≥−.003、平均Scan≥+.003，逐例Point必须exact。希望减少空间孤立假阳性，但可能误删真小灶或主要错误连着大块，尚无结果。独立研究先重读已揭示control dev，不改正在运行elastic的四臂统一读出/参数。若elastic失败且本法过门，才考虑另注册全864固定模型；不把清理临时混进elastic。

**成本/证据/决定：** 新训练0，开发360forwards/预计数分钟，新增GPU≤12GiB先smoke，CPU组件计数增加部署延迟、模型仍1×，报告<2MiB无大cache。nnU-Net官方后处理为参考，当前不是largest-only复现。见[七问](archive/research_records/2026-09-10_peak_protected_cleanup/IDEA.md)、[预注册](archive/research_records/2026-09-10_peak_protected_cleanup/PREREG.md)。

**04:15工程/访问状态：** 9项算子/评分门测试通过；fit前8例batch8 GPU smoke峰值4.753GiB、Point exact、仅删不加、0.82s，低于12GiB限制。一次360forwards开发入口已启动；本研究会新增reused dev访问，但不加载正在训练的elastic checkpoint，不改其参数/四臂后统一读出。见[smoke](archive/research_records/2026-09-10_peak_protected_cleanup/SMOKE_RECEIPT.json)。

**最终结果04:17 UTC：** 清理后seed11/12 Scan0.746668827/0.744056511、Subject0.733121612/0.729148266、Point170/163。Scan差−.000093489/−.000120907，CI[-.000270551,+.000067809]/[-.000318713,+.000062150]；Subject差−.000091221/−.000169869，CI也跨0，不称显著恶化。两seed提升门失败，平均Scan−.000107198，不扫比例/连接性救门，条件性full864支线关闭。

**直觉/机制/成本/决定：** 峰值保护360/360点exact得到支持，分割提升直觉未支持。删TP3805/5071、FP4401/4282，后者仅占原FP0.3447%/0.3762%，当前小岛规则不是主要提分来源；不泛化为所有后处理不可能。0新训练、360forwards/31.63s、额外峰值4.815GiB，无新checkpoint/cache、部署1×另加CPU处理。仅reused dev180/78，无test/outer/confirmation/external；弱形变原实验不变。见[结果](archive/research_records/2026-09-10_peak_protected_cleanup/RESULT.md)、[解释](archive/research_records/2026-09-10_peak_protected_cleanup/INTERPRETATION.md)、[raw](archive/research_records/2026-09-10_peak_protected_cleanup/RAW_DEVELOPMENT.json)、[验证](archive/research_records/2026-09-10_peak_protected_cleanup/VERIFICATION_RECEIPT.json)。以上启动时记录保留历史。

## 弱形变全864固定重训 ⛔ 主开发门失败，未准入/未执行

**最终准入结论08:12：** Stage A两配方失败，按条件关闭本分支。full864训练0、test0、无新checkpoint，原dev180没有成为训练；不是full864模型负结果。后文执行前预估保留历史，当前不再等待或启动。见[未执行结果](archive/research_records/2026-09-10_weak_elastic_full864/RESULT.md)、[机器记录](archive/research_records/2026-09-10_weak_elastic_full864/RESULT.json)。

**问题/直觉：** 旧test270整体Scan0.757270、1×Scan0.752003；开发涨分不能代替超榜。只有两seed弱形变过门后，才用全部授权864/391人从头学一个固定方案并一次比较。

**流程/控制/门：** 原fit684+dev180变为训练、旧confirmation继续排除；新seed2026091031同初始化、抽样、168轮、主损失，两臂只追加形变与否。无教师；辅助系数0/零占位需核主梯度等价。Stage A全门与时间/工程全过才启动，固定control/candidate×identity/八镜像一次test270，全四行归档。点估计超榜和患者簇CI支持分开，不按test救门。

**机制/结果/成本/下一步：** 希望形状增强收益在更多训练病例上保留；尚无full模型/性能，不支持也不否定。预估双卡2–2.5h、无新大cache、checkpoint<300MiB，部署1×或预定8×。先等Stage A结果，失败则不执行。见[七问](archive/research_records/2026-09-10_weak_elastic_full864/IDEA.md)、[预注册](archive/research_records/2026-09-10_weak_elastic_full864/PREREG.md)。

## 弱弹性形变双抽样层 ❌ 四模型开发完成，两固定配方关闭

**问题/直觉：** native两个固定seed开发Scan0.746762/0.744177，现增强缺连续形状变化；把图和答案一起轻捏，希望改善轮廓适应。患者抽样旧配方Dice正但Point门失败，结论不翻案。

**流程/控制：** 原128³增强后20%样本追加边界归零、最大2输入体素的平滑位移，图像三线性/标签最近邻；无教师、同模型/初值/168轮/损失/阈值1×。两个固定抽样层分别比较有无形变，四新增训练复用四个无形变对照，不偷偷混因素。明确修订9月8默认native-only提案为事前2×2设计，不调抽样权重。

**门/机制/状态：** 每seed相对native Subject≥+.003、Scan>0、Point≥−1、两Dice簇CI下界≥−.003、两seed平均Scan≥+.003；subject_elastic另要求相对自身对照两Dice>0且Point不减。全门才晋级；无结果、无支持结论。预计四训练双卡4–5h、无新大缓存、部署1×；先核40GB GPU、形变无折叠/非空标签和精确导出。失败不扫强度救门。见[七问](archive/research_records/2026-09-10_weak_elastic/IDEA.md)、[预注册](archive/research_records/2026-09-10_weak_elastic/PREREG.md)、[8小时session](archive/research_records/2026-09-10_weak_elastic/SESSION.md)。旧提案状态以下保留历史，本节为当前执行状态。

**03:46 UTC工程/执行：** 5项合成测试通过；fit前8例GPU smoke的初值与旧control exact，原增强和模型RNG匹配、导出error0、峰值20.664GiB，48次输入9次形变且安全门通过。不是开发涨分证据。数据/cache/checkpoint/旧raw与实现已冻结，第一seed两抽样层启动，第二seed随后自动启动；四训练全部结束才统一dev读出。见[smoke](archive/research_records/2026-09-10_weak_elastic/SMOKE_RECEIPT.json)、[实现冻结](archive/research_records/2026-09-10_weak_elastic/IMPLEMENTATION_FREEZE.json)、[训练前收据](archive/research_records/2026-09-10_weak_elastic/TRAIN_PREACCESS.json)。

**fit-only标签压力诊断（完成）：** 全684/313逐例固定形变，整WT全保留；2277个128³组件中5个ID消失，共6体素/最大2体素/涉及5例。绝对体积差中位1.65%、p95 5.32%、最大12.72%（228→199），工程合同通过但非完全无损。掩膜变形前后均值Dice0.913754不是模型分数；100%逐例压力样本不是正式20%实际发生率。113.45秒/2CPU线程、RSS2.392GiB（高于<2GiB估计）、0GPU/model/dev、无新cache，参数未改。见[七问](archive/research_records/2026-09-10_weak_elastic/FIT_AUGMENTATION_AUDIT_IDEA.md)、[结果](archive/research_records/2026-09-10_weak_elastic/FIT_AUGMENTATION_AUDIT.md)、[raw](archive/research_records/2026-09-10_weak_elastic/FIT_AUGMENTATION_AUDIT.json)。前16轮抽样/原增强RNG/新形变流独立重建exact，发生率19.81%，仅机制核验，见[训练重放](archive/research_records/2026-09-10_weak_elastic/TRAINING_REPLAY_035944.json)。

**优先级文献补记：** 新核nnU-Net当前默认elastic概率为0，不能把“官方默认采用elastic”当选择理由。本选择来自BraTS专门增强消融和本项目缺口；旋转/缩放、低分辨率模拟、图像/标签混合只作备选筛查，不在已冻结训练中追加。LoMix旧训练辅助监督不等于TensorMixup图像混合。见[本轮筛查与排序](archive/reviews/2026-09-10_data_processing_priority/RESEARCH.md)。

**第一seed完成05:57/核查06:04：** native/subject两层168轮各7767.62/7763.09秒、峰值20.664GiB、export0，完整抽样和原RNG与各自旧control匹配，新形变20328步独立重放exact，实际20.0038%样本形变、无新操作擦空WT。仅训练工程结果；两checkpoint保留，未读其dev。第二seed自动运行，原四臂门不变。见[完整第一seed/第二seed起步重放](archive/research_records/2026-09-10_weak_elastic/TRAINING_REPLAY_055913.json)。

**08:07正式读出开始：** 四条168轮全部完成；seed12 native/subject耗时7754.75/7769.98s，各20.664GiB、export0。全672轮抽样/原增强RNG与四条20328步形变流重放exact，seed12发生率19.8735%。现在统一dev180/78的四候选720forwards，未有新性能结论，test/outer/confirmation/external未访问。见[全程重放](archive/research_records/2026-09-10_weak_elastic/TRAINING_REPLAY_FINAL.json)。

**最终结果08:10：** native两seed Scan0.745530/0.744132，Subject0.731086/0.730980，Point172/167；对自身control Scan−.001232/−.000046、CI均跨0，Point+2/+4，但增益门未过。subject两seed Scan0.742229/0.751126，Subject0.732752/0.739671，Point172/165；对自身subject-only Scan−.011245/−.004780，两扫描CI均负，Point+5/−3。两配方全部关闭，selected_arm=None，不扫强度或挑seed，full864/test形变支线不准入。

**解释/成本/下一步：** source工程和672轮重放都过，安全形变不等于分割泛化收益。subject seed12虽高于native却低于自身control，不能误把旧采样收益归给elastic；只native层Point改善得到两seed支持。四训练8.6265 GPUh，峰值20.664GiB，dev720forwards/21.84s，部署1×，无test/outer/confirmation/external。仅同180/78 reused development、非独立两cohort，旧80GB control复用边界保留。按结果前准备转占比target，不改当前负配方。见[结果](archive/research_records/2026-09-10_weak_elastic/RESULT.md)、[解释](archive/research_records/2026-09-10_weak_elastic/INTERPRETATION.md)、[raw](archive/research_records/2026-09-10_weak_elastic/RAW_DEVELOPMENT.json)、[验证](archive/research_records/2026-09-10_weak_elastic/VERIFICATION_RECEIPT.json)。

## 数据处理：fit空间往返保真诊断 ✅ 全684例完成；微小块损失已量化

**问题/直觉：** fit684裁剪中位136×174×142被缩成128³，尺度差异中位1.2879；先把答案缩小再还原，量化边界和细小连通块损失，不把模型0.75当上限。

**做法与固定项：** 只读fit原生标签/影像header和旧cache，固定image-only crop与nearest-exact128重建；再按最近邻、三线性>0.5分别还原原生坐标，报告Dice/体积/表面误差及26邻接组件ID存活。两算子全部报告，不选高分；图像像素/模型/开发/测试不访问。

**结果：** V2全684/313完成、cache684/684 byte exact、首例跨进程重复与独立计数聚合核验通过。最近邻/三线性指示图还原Scan Dice0.965432/0.964736、Subject0.963995/0.963593；这是标签保真，不是模型分数/上限，无bootstrap。2446个26邻接块中184个缩小后无自身ID取样，涉及124例，合计469体素（WT总量0.001185%），最大63体素；126体素以上1136块本次全有表示。1例crop外另有4体素，保留不排例。

**直觉/成本/决定：** 主体掩膜较好保留，微小块/拓扑确有损失；不能推断当前0.75已经到上限，也不能把7.5%连通块消失叫肿瘤体积或模型漏检率。约149.7秒、4 CPU worker、单worker最高RSS1.039GiB、raw统计2.83MB，0 GPU/model/dev/test，部署不变。最高优先诊断已完成；继续保留轻弹性形变为首个训练候选，其工程/正式训练尚未启动。见[结果](archive/research_records/2026-09-08_fit_spatial_roundtrip_v2/RESULT.md)、[解释](archive/research_records/2026-09-08_fit_spatial_roundtrip_v2/INTERPRETATION.md)、[验证](archive/research_records/2026-09-08_fit_spatial_roundtrip_v2/VERIFICATION.json)。

**单位修订：** V1保存2行后遇未声明mm标签停止，预取额外fit访问数不完整，原记录保留。全684 header数值几何完全一致但14例两文件均unknown；V2统一原生体素距离，不猜mm、不排例，Dice/体积/组件公式原样复用。16项算子/单位合成测试及4项归档测试通过；V2会话88466已exit0，访问日志685次fit标签尝试/完成（首例重复），唯一684。见[V1结果](archive/research_records/2026-09-08_fit_spatial_roundtrip/RESULT.md)、[V2修订](archive/research_records/2026-09-08_fit_spatial_roundtrip_v2/AMENDMENT.md)。

## 数据处理：轻量弹性形变 ◻️ 调研提案，未实施

**想法/问题：** 图和答案一起作小幅平滑形变，补足当前直接分割入口没有连续形状扰动的问题；native control开发Scan0.746762/0.744177。当前入口只有翻转/平移/全局轻度强度与噪声，其他SAM分支的elastic实现不能代替本协议验证。

**做法/固定项：** 只追加一个弱空间操作，图像和全部标签共用坐标场；网络、病例抽样、初值、主损失、168轮、基础增强和1×推理不变。工程起点为p0.2/最大位移2体素，不等同论文标准差2，正式前必须固定坐标/插值/范围且不按dev搜索。研究依据为BraTS2020原始增强消融，四模态/3D U-Net结果不能搬到本单T1ce任务。

**机制/状态/成本/决定：** 希望改善形状变化鲁棒性，尚无新性能或支持结论。需先验证无折叠、标签对齐、边界与非空WT，再正式预注册；Dice/Point和CI沿用既定联合门，不放宽。新增两候选双卡约2–3h起步估计，是否重训control另算；不建新cache，推理仍1×。七问与适用边界见[IDEA A](archive/research_records/2026-09-08_data_processing_candidates/IDEA.md)、[调研](archive/reviews/2026-09-08_segmentation_data_processing/RESEARCH.md)。

## 数据处理：保长宽比缩放与补边 ◻️ 先做fit保真诊断，未实施

**想法/问题：** 缩小脑区时不把它拉成立方体。fit684裁剪中位136×174×142，统一128³后三轴尺度比中位1.2879、最大1.5；几何现象已证实，精度损失未测。旧160³负结果不直接检验这个因素。

**做法/固定项：** 同一image-only裁剪，最长轴缩至128、其他轴等比例后补边到128³；其余归一化/网络/抽样/训练不变。先审header spacing和标签往返保真；它不是模型上限。改变栅格后，必须在两臂相同原生参考标签上评价，不能拿不同128³标签的Dice混比。

**机制/状态/成本/决定：** 希望保留相对形状，但短轴分辨率更低可能有害；无新性能、未判断直觉。需要新空间缓存和共同网格对照，数GiB至十GiB额外缓存先查磁盘；模型仍1×。正式成功/停止门不变，先工程探针再冻结，不重测旧test。见[IDEA B](archive/research_records/2026-09-08_data_processing_candidates/IDEA.md)、[fit几何审计](archive/reviews/2026-09-08_segmentation_data_processing/FIT_GEOMETRY.json)。

## 数据处理：固定物理尺度块训练/盲滑窗 ◻️ 分阶段空间路线，未实施

**想法/问题：** 不把整脑压进一个小盒子，局部看细节再拼回全脑；当前最小缓存WT228体素、整WT均非空，但小连通灶/边界保真未核，不能断言到上限。

**做法/固定项：** 先建立固定spacing/patch/无GT滑窗的新基线；再只比较同协议下均匀patch与前景patch抽样，不一次捆绑spacing、强增强和采样后称单因素。训练可用fit标签选前景，推理必须覆盖全部image-only网格，不能GT定位。

**机制/状态/成本/决定：** 希望保留局部细节；没有新结果。官方nnU-Net给出范式，不保证本任务增益。推理多块，可能数倍至十余倍forward，不能称1×；缓存可能十余GiB，需新控制和共同原生网格评分。参数/预算尚未冻结，只在fit保真诊断后决定是否进入该路线。见[IDEA C](archive/research_records/2026-09-08_data_processing_candidates/IDEA.md)、[调研](archive/reviews/2026-09-08_segmentation_data_processing/RESEARCH.md)。

## 患者等概率、患者内体积加权采样 🟡 Dice提升，但Point安全门失败；当前晋级关闭

**2026-09-10元数据补记：** 重建四历史训练672轮索引SHA全exact，native批次重复患者12.74%，但不同患者均值7.866/8（重复槽位1.67%）；subject策略为8.57%/7.912。仅分散同患者、保留全轮抽样次数是不同的数据组织候选，但当前潜在去重复量有限，未实现/未正式注册，不重启旧患者权重配方。0像素/model/dev新访问，见[核查](archive/reviews/2026-09-10_data_processing_priority/FIT_BATCH_REDUNDANCY.md)。

**一句话/问题：** 每位患者先获得相等训练机会，再按其内部扫描的原生体积权重选图。当前fit313人概率相差22倍，最高32人占27%，每轮968 draw预计覆盖271人；患者等概率算术参考覆盖299人，而主Subject Dice按患者等权。

**做法与单一因素：** 只将每扫描w_i除以所属患者权重和。复用两个预先指定native control（无教师/aux系数0），候选相同双通道初始化、损失、缓存、168轮/121步/两seed、阈值和1×导出；未修改旧训练源码。提案来自训练元数据和簇加权统计文献，早于源dev揭示；此前扫描覆盖/小病灶回放没有直接检验患者总概率相等。

**冻结门：** 每seed Subject≥+.003、Scan>0、Point≥−1，Subject与患者聚类Scan CI下界≥−.003，两seed平均Scan≥.003；否则关闭，不调混合比例或seed。证据[训练概率审计](archive/reviews/2026-09-07_fit_subject_sampling/RESULT.md)、[七问](archive/research_records/2026-09-07_subject_balanced_sampling/IDEA.md)、[预注册](archive/research_records/2026-09-07_subject_balanced_sampling/PREREG.md)。

**最终观察（12:49 UTC）：** seed11 candidate/control Scan0.753475/0.746762、Subject0.740633/0.733213、Point167/170；seed12为0.755906/0.744177、0.742725/0.729318、168/163。平均Scan差+0.009221、Subject差+0.010413。Scan CI[+.001555,+.012001]/[+.005221,+.019021]；Subject CI[+.001531,+.013552]/[+.006627,+.020811]，四区间都正。唯一失败项为seed11 Point−3；第二seed全部门过，不能只挑该seed或平均Point来放行。仅同一180扫描/78人的reused development，不是独立外部确认。

**直觉与决定：** 训练覆盖确实更均匀，开发整块肿瘤Dice收益也得到两seed支持，但最高分单点定位未稳定改善。四个预定体积组两seed Scan差均正；seed11 Point净损失3在最小组，seed12该组却+3。保留Dice-positive/Point-unstable证据与两checkpoint，当前完整目标的晋级关闭，不搜索temperature/mixture/epoch/seed，不自动测test或外部。不能把联合门失败写成“完全没有分割效果”，也不能把Dice正证据写成“整体成功”。

**成本与工程：** 新增两训练4.028 GPU-hours/双卡workflow约2.04h、峰值20.664GiB，复用cache无新影像缓存，部署4.70M/1×。新增dev360 forwards/12.12秒，control不重跑。两初值、168轮增强RNG、121步/轮匹配，导出error0；全部672个epoch-policy抽样SHA独立重建exact，720条保存点/真值计数与dev标签exact，审计0新forward。证据[结果](archive/research_records/2026-09-07_subject_balanced_sampling/RESULT.md)、[raw](archive/research_records/2026-09-07_subject_balanced_sampling/RAW_DEVELOPMENT.json)、[验证](archive/research_records/2026-09-07_subject_balanced_sampling/VERIFICATION_RECEIPT.json)、[错误描述](archive/reviews/2026-09-07_development_error_description/SUBJECT_SAMPLING.md)、[完整抽样重建](archive/reviews/2026-09-07_fit_subject_sampling/SAMPLER_REPLAY_FINAL.md)。以下带时间阶段记录保留历史，不代表仍在运行。

**训练元数据风险量化（09:38 UTC）：** 最小肿瘤四分位抽样质量44.89%→27.44%，最大四分位12.33%→22.38%；预计每轮不同患者271.02→298.87，不同扫描仅476.36→479.78。提高患者覆盖会削弱全局小病灶偏重，不能先验当作准确率收益；方案和门均未改。见[取舍核查](archive/reviews/2026-09-07_fit_subject_sampling/SAMPLER_TRADEOFF.md)。

**10:47正式状态：** 源辅助监督已完整负结果归档，三validator通过、SOURCE_CANONICAL_CLOSED冻结；本方法内容与源结果前设计一致。GPU smoke中患者总质量、患者内比值、native权重、初值匹配、增强RNG、采样索引改变、导出error0及显存门全部通过；训练前冻结后两个候选已并行启动，监督器会话12286/PID840708。尚无本候选开发结果，不能称支持。上面的CPU等待状态保留历史；当前继续固定168轮，见[smoke](archive/research_records/2026-09-07_subject_balanced_sampling/SMOKE_RECEIPT.json)、[训练前收据](archive/research_records/2026-09-07_subject_balanced_sampling/TRAIN_PREACCESS.json)。

**前21轮采样机制核查：** 两seed/两策略各21轮有效索引SHA与独立公式和连续随机序列完全一致；每轮患者覆盖约271→299–300，小体积组抽样45%→27–28%。机制成立不等于精度提升，不横比不同采样分布的训练Dice。工具3测试通过，完整168轮核查待训练结束；见[阶段报告](archive/reviews/2026-09-07_fit_subject_sampling/SAMPLER_REPLAY_SNAPSHOT.md)。

## 训练期切除腔辅助监督 ❌ 复用开发门失败，当前0.1配方关闭

**一句话：** 在训练时额外教会共享网络区分“切除腔”和“普通背景”，部署仍只输出全肿瘤。

**问题/依据：** 原训练池814/972有RC标签却在二分类中合并为背景；新授权fit684/313、dev180/78尚无基线分数。BackSplit提供语义背景监督的SegResNet一手先例，但本变体保留单通道WT输出，不能直接套论文收益。此前中心点头/多尺度监督负结果没有检验这一个因素。

**流程与固定因素：** 两臂结构、随机初始化、fit、128³影像/WT、采样、增强、主损失、AdamW、168轮相同；均无教师/KD。仅候选加入0.1倍masked RC Dice+BCE损失；对照系数0。两seed2026090711/12四训练全部完成后才访问180扫描开发集。

**门：** 各seed Subject差≥.003、Scan>0、Point≥−1，Subject与患者聚类Scan CI下界≥−.003，两seed平均Scan≥.003；全部工程门通过才称复用开发支持。失败不调weight/epoch；不自动访问test或外部。

**最终开发结果（10:43 UTC）：** seed11 candidate/control Scan0.744044/0.746762、Subject0.729224/0.733213、Point166/170；seed12为0.739218/0.744177、0.724989/0.729318、161/163。两seed平均Scan差−0.003839、Subject差−0.004159；每seed全部性能门失败。Scan配对CI分别[-.010669,+.003737]/[-.009988,+.000150]，Subject分别[-.012727,+.003427]/[-.009880,+.001407]，均跨0，不夸大为显著恶化。仅同一180 scans/78患者reused development，不是两份独立确认。

**直觉/成本/决定：** 辅助loss在两seed末10轮降至0.15495/0.15064，对照约1，证明辅助任务确实学会；但WT收益直觉未获支持且Point少4/2。四训练工程门、精确导出均过，故关闭该实现/0.1权重，不否定所有背景语义监督。正式训练8.064 GPU-hours/双卡约4.07h、峰值20.664GiB，RC cache1.34GiB，开发720 forwards/23.12秒，部署4.70M/1×不变。没有新增test/outer/confirmation/external，旧榜不变；下一独立患者采样早于本dev结果预注册，不按本结果改配方。见[结果](archive/research_records/2026-09-07_structured_background_aux/RESULT.md)、[raw](archive/research_records/2026-09-07_structured_background_aux/RAW_DEVELOPMENT.json)、[独立验证](archive/research_records/2026-09-07_structured_background_aux/VERIFICATION_RECEIPT.json)。以下阶段记录保留历史，不代表仍在运行。

**08:42阶段结果：** 第一seed168轮全部完成，末10轮augmented sampled fit Dice control/candidate=0.785106/0.785573，辅助loss1.000954/0.154948；辅助任务学到而主拟合几乎tie，dev尚未访问。两臂168轮采样/增强全exact，导出误差0，耗时7222.6/7294.2秒；第二seed已自动启动、初值一致。见[第一pair训练复核](archive/research_records/2026-09-07_structured_background_aux/PAIR1_TRAINING_REVIEW.json)。

**观察/成本/下一步：** 仅工程结果：fit684重建WT byte exact、561例有RC；RC cache约1.34GiB。初版CPU导出有2.4e−7差，拆开末层计算后CPU/GPU bit-exact；合成两臂初值/增强相同、控制aux梯度0/候选非0，峰值20.66GiB。实现/预访问冻结后于06:39 UTC启动第一seed双卡正式训练，dev尚未访问；4条168轮、双A100两批，每臂上限6h/24GiB，导出仍4.70M/1×。证据[七问](archive/research_records/2026-09-07_structured_background_aux/IDEA.md)、[预注册](archive/research_records/2026-09-07_structured_background_aux/PREREG.md)、[工程](archive/research_records/2026-09-07_structured_background_aux/ENGINEERING.md)、[smoke](archive/research_records/2026-09-07_structured_background_aux/SMOKE_RECEIPT.json)、[训练前收据](archive/research_records/2026-09-07_structured_background_aux/TRAIN_PREACCESS.json)。

这份文档回答的不是“代码在哪”，而是：每个科研idea为什么出现、到底做了什么、有没有用、为什么停下。详细数字排名看[`AFLOW_BRATS_LEADERBOARD.md`](AFLOW_BRATS_LEADERBOARD.md)，完整证据看各方法归档。

2026-09-07开发流程已获用户授权：原972池排除旧confirmation的53人/108 scans后为864 scans/391人，固定313/78患者fit/dev，实际684/180 scans，明确标为reused development。当前首个辅助监督候选与无教师对照已开始训练；其他test/outer/confirmation/外部角色继续保护。证据见[角色冻结结果](archive/research_records/2026-09-07_reused_development_roles/RESULT.md)。BackSplit提供背景细分监督的一手先例，本项目采用保留WT主输出的辅助头变体；不能把论文收益搬用，见[文献适配](archive/reviews/2026-09-07_gpt6_reentry/LITERATURE_FIT.md)。

2026-09-05证据身份补记：Flip8 A/B与镜像一致C/D的学生训练/inner确实隔离，但两位固定教师的历史756例训练集与inner分别重叠87/72/88/72 scans、39/33/39/37 subjects。因此下文“fresh”只能解释为学生侧新划分，不能解释为整个蒸馏系统从未见过患者。原正负数值不变、间接泄漏偏差未量化；新内部研究须记录教师训练/选择患者身份并解决全链路隔离。[清单审计](archive/reviews/2026-09-05_30h_chain_audit/TEACHER_INNER_EXPOSURE.json)。历史卡片中的“当前最高/下一候选”描述其当时阶段；现时排序以顶部总览和排行榜为准。

2026-09-05训练源provenance补记：官方BraTS2024 corrected-v2 `syn60086071`匿名下载仍因SelfSign requirement9606685返回403；本地Kaggle version2三包与官方zip/member字节parity未完成。历史用户授权允许development fit/tune继续，但不是官方来源核验通过，也不自动授权confirmatory/test。所有内部方法卡继承“镜像来源、官方字节parity未证明”限制；外部TCIA队列本身的来源证据独立，但模型checkpoint仍继承训练源限制。[复查结果](archive/reviews/2026-09-05_brats2024_source_provenance_recheck/RESULT.md)。

## 先理解我们在做什么

UCSD全75例直接官方核验已完成：150/150几何一致、75/75标签值exact；官方75图像float32、镜像75图像uint8，图像值0/75 exact，18,156个原非零体素变零。患者映射官方复核通过。2026-09-06按冻结协议完成同cohort官方float来源纠正；旧mirror行保留，新结果明确标reused/not independent。证据：[来源核验](archive/reviews/2026-09-05_30h_chain_audit/UCSD_SOURCE_RESULT.md)、[纠正结果](archive/research_records/2026-09-06_ucsd_official_float_source_correction/RESULT.md)。

来源更正（2026-09-05/06）：旧UCSD75评测实际使用`data-nih/tcia`整理镜像的75份uint8图像；镜像自述来自Pittsburgh Fiber Data Hub，不能称TCIA官方GitHub。患者映射已由TCIA官方临床v3表直接复核75/75一致、53患者零重叠。旧结果限定为mirror-uint8；另行预注册的official-float32纠正只换影像字节并完成一次。两版本同患者、同标签，不算两份外部证据。

输入是一份患者的T1ce MRI三维图像。目标是自动找出WT（项目定义为原标签1/2/3，标签4不算）的三维区域。

可以把模型想成给每个体素打一个分数：

- 分数大于0，就认为该体素属于肿瘤；
- 所有体素中分数最高的位置，作为自动point；
- 预测区域和真实区域重叠越多，Dice越高。

常见指标：

- **Scan Dice**：每次扫描等权平均；这是当前排行榜主指标。
- **Subject Dice**：先在同一患者内部平均，再让每个患者等权。
- **Point hit**：最高分位置是否真的落在肿瘤里。

状态符号：✅成功；❌负结果；🟡有局部收益但不是最终最佳；🔒协议不同、只能参考；🧪计划中。

## 一页总览

| 方法 | 最直白的想法 | 状态 | 核心结论 |
|---|---|---|---|
| 患者等概率采样 | 先给每个患者相等机会，再从其扫描中按体积挑图 | 🟡 Dice正/Point门失败 | reused dev两seed Scan差+0.006713/+0.011729且CI正；Point−3/+5，不能整体晋级 |
| 训练期切除腔辅助监督 | 额外教网络分清切除腔与普通背景，部署裁掉辅助头 | ❌ 当前0.1配方关闭 | 辅助任务学会，但reused dev两seed Scan差−0.002718/−0.004959，Point−4/−2 |
| N4仅用Otsu自动前景拟合 | 只改变补光场的参考区域，保留全图组织 | ❌ Stage A关闭 | E小正但CI门失败；F identity/Flip8均负，未进Full972/外部 |
| 不额外做N4，fold内重训教师 | 同条件比较有无额外亮度偏置校正，并让教师也避开评测患者 | ❌ Stage A关闭；E正/F不稳 | E Scan约+0.018，F接近tie且CI/Point门失败；跨折均值不能放行 |
| RHUH发布态no-N4输入桥接 | 恢复黑背景后不再二次自动补光 | ✅ input-only；模型链关闭 | 80/80输入门过；no-N4 Stage A失败，0标签/0GPU/0模型退出 |
| RHUH发布态Otsu-N4输入桥接 | 恢复背景后用Otsu脑组织拟合N4 | ❌ exact重建门 | 80/80单次finite，但3例最终image SHA均不复现；0标签/0模型 |
| UTSW人工标签双版本固定3例桥接 | 把两版T1ce与两版人工答案逐一叠格子 | ❌ 唯一pair门；标签谱系部分解决 | 双传输15/15、数组有效；`r`标签全网格，但FeTS/ANTs两图都对齐，0模型 |
| UTSW官方主FeTS图no-N4输入桥接 | 按发布说明选标准FeTS图，不按模型分数选图 | ✅ 固定3例input-only门 | 双副本128³ image/WT/geometry 3/3 exact；0模型，其余359人工病例未访问 |
| UTSW主FeTS图canonical-N4输入桥接 | 给未来same-seed control按原训练配方补光 | ✅ 固定3例input-only门 | 跨进程image/label/geometry3/3 exact；对no-N4标签/几何同、图像不同 |
| UTSW主FeTS图Otsu-N4输入桥接 | 为可能单独晋级的Otsu候选提前接外部输入 | ✅ input-only；模型链未授权 | 3例工程门过；Otsu Stage A失败，未下载其余359例/跑模型 |
| UTSW362 canonical/no-N4配对外部确认 | 同一跨阶段队列让两套完整流程逐人比较 | ❌ 条件门前关闭 | no-N4 Stage A失败；0preaccess/0新病例/0cache/0模型 |
| UTSW362冻结baseline跨阶段外部基线 | 不再调模型，把KD168与GCSAM184送去新医院术前队列 | ❌ 标签域门前关闭 | 双下载724/724 exact；raw标签含未定义3/5，0UTSW模型/指标 |
| ReMIND T1ce/manual tumor来源探针 | 固定首例先验图像、标签、脑mask能否原生对齐 | ❌ NRRD双传输门 | 43例metadata候选；首轮文件成功，第二轮SSH失败，未解码/未跑模型 |
| ReMIND固定首例DICOM SEG探针 | 同一官方DICOM盒中按SOP UID对齐图像与标签 | ❌ 源间隔exact门 | 下载复现全过；gap误差0.000860mm超冻结1e-5，pixel array前停止 |
| ReMIND量化感知/物理grid映射 | 按DICOM文本刻度区间和真实坐标叠不同grid | ❌ SEG spacing区间门 | source区间过；SEG内部残差最高2.42e-5层，SEG pixel前停止 |
| 标准化MRI背景零点解码 | 先把发布文件挪走的黑背景零点移回去，再走原输入链 | ❌ 完整输入链等价性门失败 | 支持/裁剪框exact，但N4后最大误差0.05768超0.001门；没有外部模型访问 |
| Vanilla SAM-Med3D | 先复现最原始的交互分割基线 | 🔒 | 建立基线，但依赖GT点击且分数低 |
| Image brain crop | 先去掉大片黑背景，再缩放 | ✅ | 旧协议test提升约`+0.054` |
| GT crop / residual refiner | 已知肿瘤附近位置后再精修 | 🔒 | 有效，但不是全自动 |
| Direct plain locator | 直接让小网络输出肿瘤，不再接旧SAM | ✅ | 形成全自动主线，epoch99 test `0.725820` |
| CAPG中心头 | 另加一个头专门学肿瘤中心 | ❌ | point远差于plain argmax，SAM也没改善 |
| 灰度/intensity residual | 利用灰度线索修正mask | ❌ | Dice和point都没有稳定收益 |
| Late-cosine epoch168 | 100轮可能没收敛，继续低LR训练 | ✅ | test Scan从`0.725820`到`0.741204` |
| True-160³ | 用更密的三维网格保留更多细节 | ❌ | 成本近2倍，test反而下降 |
| EMA | 平滑最近多个训练权重 | 🟡 | 单独Dice持平，point更好，提供互补知识 |
| Logit ensemble | 强模型为主，混入25% EMA输出 | ✅ | 历史总体test曾达`0.745054`，后被Full972/Flip8超过；推理2× |
| Parameter soup | 不跑两模型，直接平均它们的参数 | ❌ | 跨seed不在同一权重盆地，validation崩溃 |
| Voxel-KD | 让单模型模仿成功ensemble的每个体素概率 | ✅ | 旧版本单模`0.742976`；Full972 KD168成为现1×主干，固定Flip8刷新总体 |
| ReCo-KD | 分开教前景、边界和背景，避免背景淹没 | ❌ | 同epoch略好，但不超最佳base KD |
| Safe adapter / SAGE | 冻结base，只学习小的安全修正层 | 🟡/🔒 | oracle validation有小收益，不属全自动主线 |
| MCTS / routing | 减少ensemble重复forward | 🟡/🔒 | 明显加速，但没有Dice算法优势证据 |
| LoMix-3D | 训练时混合本模型粗细尺度，部署仍只用最终头 | ❌ | fixed validation低于matched control与KD；未进test |
| DCD detail distillation | 用小波只教encoder方向细节 | ❌ | 学会匹配detail，但validation低于base KD；未进test |
| KAS-on-voxel-KD | teacher离场后让student自我同化 | ❌ | paired validation低于control/KD200且Point下降；未进test |
| GCSAM Full972稳定优化 | 每步先轻推参数，再要求被推后也答得好 | ❌配方/🟡checkpoint外域 | 双复制不稳；原checkpoint在Africa95 Dice较Full高0.0686且CI正，但Point少3，不回流复活配方 |
| GCSAM固定184双复制 | 不调方法，在两套新train-only seed上重做同一比较 | ❌ Stage A | A/B内部复制仍失败；Africa只支持原checkpoint域特异覆盖，不证明优化器可复制 |
| T1ce固定1–99百分位裁剪 | 先剪掉每例脑内最极端的1%亮暗值，再统一亮度 | ❌ Stage A | A/B Scan分别下降`0.008210/0.007520`且四个Dice CI整体低于0；未进Full972/榜单 |
| Direct locator固定Flip8 TTA | 同一模型把MRI按8种镜像看完，翻回后平均 | ✅ 完成；跨队列方向支持/8×代价 | 五cohort Scan点估计均正；internal/CPTAC/WAW-pre CI正，Africa/WAW-post跨0 |
| 镜像前后答案一致训练 | 训练时让原图与镜像图翻回后的最终mask一致 | ❌ fresh C/D失败并关闭 | 一致性降35–36%，但8× Scan平均`-0.001918`、identity Scan平均`-0.000115` |
| Fresh三seed共识蒸馏 | 三位独立老师的平均答案教回一个新学生 | ❌ Stage A | 共识救弱seed但几乎不超强seed，强seed两项CI门失败；未训练student |
| GT最佳老师路由蒸馏 | 每个训练病例听三位老师里soft-Dice最好的那位 | ❌ | Routing覆盖三老师但validation三项均低于control/archive；未进test |
| UCSD患者隔离外部确认 | 把冻结模型带到未见患者的新医院病例上各考一次 | ✅ 外部确认完成 | GCSAM在UCSD75三项点估计领先KD，但paired Dice CI跨0；不得回流调参 |
| UCSD75官方float32来源纠正 | 同一批病例从压缩8位镜像换回官方浮点原件重算 | ✅ 一次性完成并关闭 | 两模型Dice仅`+0.0002–0.0011`且来源CI跨0；GCSAM Point少1；不是新队列 |
| EGD人工二值WT外部候选 | 用374位Erasmus术前人工0/1全肿瘤作跨院验证 | 🔒 DUA/账号阻塞 | 标签语义清楚且有T1GD；须用户/机构签协议，匿名subjects API 403，0病例/模型 |
| UCSF-PDGM专家修订外部候选 | 从501条术前MRI中排除已知BraTS泄漏再验证 | ❌ patient-disjoint门 | 298条明确映射BraTS21；UCSF也是BraTS2024来源站，余199人无2024桥，0NIfTI/模型 |
| UPENN术前corrected外部候选 | 从官方232份人工修订术前标签冻结完整队列 | 🔒 mirror completeness/parity+患者桥 | data-nih仅147 seg、非官方完整232；173旧BraTS且无2024桥，0asset/模型 |
| BraTS-Africa95冻结baseline | 两个冻结模型去尼日利亚四中心公开专家胶质瘤队列考试 | ✅ 一次性完成并关闭 | GCSAM Dice `0.5916`且较Full `+0.0686` CI全正；Full Flip8 Point最高但Dice收益不确定 |
| Burdenko治疗后GBM来源候选 | 用放疗后纵向MRI补足治疗后外部证据 | ❌ target语义 | 180人时点合适但标签为GTV/CTV/PTV、随访GTV仅子集且在CT空间；不是BraTS WT，0下载/模型 |
| CPTAC33 Africa-informed确认 | 在另一公开术前人工队列复核GCSAM checkpoint Dice方向 | ✅ 完成并关闭 | GCSAM仅`+0.0020`且CI跨0，Africa强差不复现；Flip8`+0.00325`且CI全正，Point全33 |
| WAW-GliomAI152冻结面板 | 两个冻结checkpoint在华沙公开术前人工队列只考一次 | ✅ 完成并关闭 | GCSAM/Flip8 Dice几乎tie；GCSAM相对Full不确定，Flip8 `+0.0101` CI全正 |
| WAW术后only-patient31面板 | 排除preop已见患者后每人最早术后scan考冻结模型 | ✅ 完成并关闭 | Flip8/GCSAM相对Full均小正但CI跨0；Point仅15/14/15，早期术后困难 |
| OpenNeuro-GBM334流式面板 | 八家新医院334名术前GBM逐例流式考冻结模型 | ❌ 输入门/0性能 | 332/334过；BO118/KA86各丢4/6个WT crop体素，零丢失门失败，不排例/扩框 |
| MOTUM-HGG29冻结面板 | 两个冻结checkpoint在中国三院厚层HGG队列只考一次 | ❌ independent-canonical geometry门 | Native误差`2.38e-7`，2例canonical后>`1e-5`；0cache/模型，另审共享orientation桥 |
| MOTUM-HGG29共享方向桥 | 三个对齐数组使用图像导出的同一个轴翻转/置换 | ✅ input-only双进程exact | 29/29 arrays/shared affine跨进程exact，等于旧canonical arrays；0插值/模型 |
| MOTUM-HGG29共享方向面板 | 用通过的共享坐标接口运行冻结Full/GCSAM | ❌ positive-support crop门 | 仅0001丢892/21007 WT，但0丢失门失败；0cache/模型，不排例/GT扩框 |
| CFB-GBM v3外部候选 | 用法国264人/347个放疗轮廓补治疗阶段证据 | ❌ target语义 | GTV只含强化+坏死核心、非含FLAIR异常WT；0主体/模型，不下载208GB |
| IMAGO500外部候选 | 用阿姆斯特丹大规模术前/术后四模态队列补外部证据 | ❌ 自动标签+DUA | 249 preop/190 postop mask由PICTURE自动生成、医师只接受筛选未修订；术后仅强化残留，0下载/模型 |
| CoRe-BT250外部候选 | 用华盛顿大学250名完整四模态术前专家修订mask做外部验证 | 🔒 UW预审批/DUA | 标签流程与WT语义初步合格；须用户/机构申请签署，0清单/payload/model |
| UCSD BraTS2024test56冻结面板 | 把冻结模型带到2026年新公开的BraTS-GLI 2024 test包（UCSD部分）新患者上只考一次 | ✅ 完成并关闭 | Full Point56/56；GCSAM−Full=`−0.0123`、Point−4；原CI略负但Scan数值精度敏感，不称稳健显著下降 |
| BraTS-PEDs257外部候选 | 用257名儿科治疗前高级别胶质瘤专家标签做极远域外部 | ❌ 输入前提 | 标签/许可合格，但发布not skull-stripped；canonical brain-crop链前提失败，0像素/0模型 |

## 不额外做N4，教师—学生fold隔离 ❌ Stage A关闭

开卷前家族登记：另有[N4-Otsu mask候选](archive/research_records/2026-09-05_n4_otsu_clean_replication/IDEA.md)共享同一canonical对照；所有候选须完成或先按预设工程门停止后才打开E/F。两方法不是独立患者复制，原no-N4训练/门槛未改。

原因诊断补记：纯比例缩放误差很小，0.057679主要由float32负背景编码往返的近零信号误差被N4放大；第三例低信号相对误差达22.1%，float64往返则与直接缩放exact。不得把它概括为N4执行随机或已确认的库bug。证据：[诊断结果](archive/reviews/2026-09-05_n4_scale_diagnostics/RESULT.md)。

**一句话：** 检查额外自动补光是否必要，让有/无N4学生在真正避开各自教师训练患者的新内部划分上同场比较。

**出发问题：** 同一train MRI换强度单位后，N4完整输入链最大差0.057679；旧fresh内部分割又有教师训练患者重叠，且排除全部旧评测清单后仅38 subjects/87 scans可作新inner。

**具体做法：** E/F各42-case inner（17/19 subjects），各自930 fit；每fold新训独立raw/EMA教师，再训配对canonical/no-N4学生到168。全部4教师/4学生冻结后才一次性打开两inner，同时固定identity与Flip8，使用患者簇bootstrap。没有旧checkpoint或旧评测结果参与新教师训练。

**改变/固定因素：** 每pair只改N4有无，教师权重/初值/抽样/增强/label/模型/损失/168轮/阈值不变。教师重训是两组共享的隔离基础设施，不冒充原历史checkpoint复现。

**预期机制与门：** 减少不必要校正带来的单位敏感性，同时保持有效对比度。两个fold、两种视图均须正Scan/Subject，Point最多少1/42、配对CI下界≥−0.003；跨fold两种Scan平均都>0.001。任一失败不调参、不加fold、不访问旧cohort；小样本会限制结论。

**状态/结果：** 全部训练/配对/缓存/访问工程门通过后一次性开卷。E identity control→candidate Scan/Subject/Point=`0.760810/0.766343/40→0.778854/0.779039/40`，delta=`+0.018044/+0.012696/0`，两项paired CI整体为正；E Flip8 delta=`+0.017634/+0.011707/+1`，门全过。F identity delta仅`+0.000597/+0.002239/−2`，Scan/Subject CI下界`−0.006840/−0.007921`；F Flip8 delta=`+0.000187/+0.001592/0`，CI下界`−0.007965/−0.009649`。F的Point/CI门失败，Stage A关闭。

**直觉是否支持：** 部分支持但不稳定。去N4在E17人上产生明显广泛收益，说明额外N4可能伤害某些分布；在F19人上几乎打平且不确定性超门，不能称可复现普遍收益。跨折identity/Flip8 Scan均值虽为`+0.009320/+0.008910`，预注册明确禁止均值掩盖单折失败。

**代价/下一决定：** 共享教师+control14.426 GPU-hours，no-N4候选新增6.784 GPU-hours，最高20.167GiB；cache9.49GiB，部署1×/8×。Full972 admission在剩余14.46h、8h时间门通过时仍重算eligible为空，0CUDA退出；不调N4/epoch/seed/门、不访问旧cohort。证据：[IDEA](archive/research_records/2026-09-05_no_n4_clean_teacher_replication/IDEA.md)、[RESULT](archive/research_records/2026-09-05_no_n4_clean_teacher_replication/RESULT.md)、[机器结果](archive/research_records/2026-09-05_no_n4_clean_teacher_replication/RESULT.json)、[raw](archive/research_records/2026-09-05_no_n4_clean_teacher_replication/RAW_INNER.json)、[对比图](archive/reviews/2026-09-05_30h_chain_audit/N4_FAMILY_STAGE_A_COMPARISON.png)。

## N4仅用Otsu自动前景拟合 ❌ Stage A关闭

**一句话：** 让N4只参考更可靠的脑组织估计补光场，但保留原完整脑图给模型。

**出发问题：** 3份train图的float32编码往返中，近零信号误差最高22.1%，在原正值mask-N4后放大到0.057679。取消N4还未评测，保留校正但改变拟合参考区是另一个单因素。

**具体做法：** 使用官方示例OtsuThreshold(full image,0,1,200)作为N4拟合mask；shrink2、4×50迭代等不变，校正后只把原非正值清零。最终brain crop和WT标签不由Otsu mask决定。共享当前fold-clean教师与canonical对照，所有候选在E/F首次开卷前冻结。

**改变/固定因素：** 只改N4拟合mask，学生初始seed/fit930/教师权重/loss/抽样/增强/168轮/阈值/固定8视图均与canonical对照匹配。没有百分位裁剪或mask参数搜索。

**预期机制：** 拟合时减少对极弱边缘信号的依赖，使输入编码误差不再明显改变校正场；还必须独立检验Dice/Point是否提高。

**观察结果/直觉：** 数值机制仍支持：最敏感第三例误差0.057679→0.000021。但准确率不支持稳定收益。E identity/Flip8 Scan delta=`+0.004115/+0.005374`，CI门分别失败；F identity Scan/Subject=`−0.009235/−0.007148`且CI整体低于0，F Flip8=`−0.002901/−0.001515`。跨折identity均值为负，完整Stage A失败。说明“输入编码更稳定”没有稳定转化为分割提升，只否定当前Otsu200实现，不否定一般前景mask N4。

**代价/下一决定：** 共享教师+control14.426 GPU-hours，Otsu候选新增6.783 GPU-hours，最高20.167GiB；cache9.49GiB，部署1×/8×。Full972/旧cohort/外部均不进入，不搜索bins/mask/epoch/seed。证据：[IDEA](archive/research_records/2026-09-05_n4_otsu_clean_replication/IDEA.md)、[RESULT](archive/research_records/2026-09-05_n4_otsu_clean_replication/RESULT.md)、[机器结果](archive/research_records/2026-09-05_n4_otsu_clean_replication/RESULT.json)、[raw](archive/research_records/2026-09-05_n4_otsu_clean_replication/RAW_INNER.json)、[对比图](archive/reviews/2026-09-05_30h_chain_audit/N4_FAMILY_STAGE_A_COMPARISON.png)。

## RHUH发布态影像no-N4输入桥接 ✅ input-only工程门通过

**一句话：** RHUH影像已经由发布者统一过亮度，就把被z-score平移的黑背景恢复成0，不再做第二次N4自动补光。

**状态：** 2026-09-05完成80 scans/40 subjects无标签输入QC，80/80及首3例独立重建全过；但no-N4 Stage A失败，条件queue在标签/GPU前正常退出。RHUH标签未下载、模型forward为0。

**出发问题：** 官方80份T1ce的MD5/大小全匹配，但80/80含负背景，历史正值mask N4桥接在3份train×2编码中仅4/6等价，最坏完整链差`0.057679/0.002245`，所以不能把canonical N4模型直接搬过去。

**具体做法：** 官方发布者固定commit明确流程为配准、SynthStrip去颅骨、CaPTK z-score，没有N4。每例从六面边界众数解码黑背景为0，以正值图像支持取bbox，在源脑区重新population z-score，再三线性缩到128³、最近邻脑mask清零。当前只计算输入哈希/QC；若未来no-N4获准，模型才接收同一输入并固定Flip8/阈值0输出。

**改变/固定因素：** 唯一变化是RHUH发布态背景解码后不运行N4。固定时点1/2全部80例、canonical orientation、image-only crop、source非零z-score、128³、未来checkpoint/阈值/Flip8/无SAM/无后处理；标签不用于裁剪或删病例。

**预期机制：** 避开旧失败所定位的N4尺度敏感阶段，同时尊重RHUH本身已经z-score的发布流程；工程可重复不等于跨域准确。

**观察结果：** 边界背景众数占比最小`0.983648`（门0.95），解码后80/80背景exact0且支持非空；标准化最大绝对均值`6.11e-11`、std最大误差`1.79e-10`（门均`1e-6`），全部输出finite。80例均RAS、`240×240×155`、1mm³；首3例image/mask SHA byte exact。

**直觉是否支持：** 支持“no-N4输入可严格定义并重复”的工程直觉；尚不支持模型Dice/Point或外部迁移主张。发布态强度域差异与40人宽CI仍是风险。

**成功/停止门：** 当前所有80例/所有数值门必须全过，已通过。下一阶段只有no-N4 E/F两折identity/Flip8全部门通过且条件Full972完成才允许；否则不下载标签、不跑RHUH。未来WT冻结为发布分割中所有positive tumor labels的union，不因标签编号3/4差异选病例。

**代价/下一决定：** 输入QC为3项合成合同、CPU27.315秒、0GPU/0forward、不建cache。12文件/4项专用链按预注册执行：no-N4失败后queue exit0，EXTERNAL_PREACCESS、label目录、GPU smoke、cache、模型均未创建。输入工程资产保留，但当前候选模型链关闭；不得换旧checkpoint绕过资格。证据：[七问](archive/research_records/2026-09-05_rhuh_no_n4_input_bridge/IDEA.md)、[输入结果](archive/research_records/2026-09-05_rhuh_no_n4_input_bridge/RESULT.md)、[终态](archive/research_records/2026-09-05_rhuh_no_n4_input_bridge/EXTERNAL_QUEUE_TERMINAL.json)。

## RHUH发布态Otsu-N4输入桥接 ❌ 独立重建门失败

**一句话：** 恢复RHUH黑背景后，只让可靠脑组织估计N4补光场，为可能胜出的Otsu模型预先准备外部输入。

**状态：** input-only QC完成并关闭当前实现。80/80首轮工程门通过，但首3例最终float image独立重建0/3 byte exact；不重跑、不改线程/门/病例。RHUH标签和模型均未访问。

**出发问题：** no-N4桥接虽通过，但E/F尚有独立Otsu候选；若只有Otsu胜出，不能看性能后临时设计RHUH输入。旧正值mask N4最坏差0.057679，而Otsu的train机制3/3通过，所以旧负结果未否定此实现。

**具体做法：** 80官方T1ce逐SHA重验、背景归零，固定Otsu200 fit mask、shrink2/四级50次N4，校正场应用到完整正支持，再image-only crop、源z-score和128³；只保存QC哈希，不载入标签/模型。

**改变/固定因素：** 相对RHUH no-N4桥接只加入冻结Otsu-fit N4；病例、背景、方向、支持、crop、z-score、grid及未来阈值/Flip8都固定。

**预期机制：** Otsu mask应避免极弱边缘支配N4并保持输入编码稳定；它必须同时满足独立重建，单次finite不够。

**观察结果：** 80/80单次门通过，fit mask范围1,075,619–1,618,201，标准化mean/std最坏误差`5.61e-11/1.83e-10`。重建中brain mask和Otsu fit-mask均3/3 exact，但最终image SHA 0/3 exact；未保存两次数组，不能量化差值或推断Dice影响。

**直觉是否支持：** 只支持单次支持/finite机制，不支持当前跨进程byte-exact可复现主张。该失败不否定一般Otsu，也不改变已冻结BraTS cache/Stage A。

**成功/停止门：** 80例全部工程门且3例image/mask/fit-mask SHA全exact；image门失败后立即关闭。不调bins/N4迭代/线程/容差，不下载标签、不跑模型。

**代价/下一决定：** 2项合成合同，CPU634.269秒/8 worker，0GPU/0forward、不建cache。即使Otsu Stage A以后通过，本实现也不授权RHUH；需另立不依赖RHUH分数的新工程假设。证据：[七问](archive/research_records/2026-09-05_rhuh_otsu_input_bridge/IDEA.md)、[结果](archive/research_records/2026-09-05_rhuh_otsu_input_bridge/RESULT.md)、[raw QC](archive/research_records/2026-09-05_rhuh_otsu_input_bridge/INPUT_QC.json)。

## UTSW人工标签双版本固定3例桥接 ❌ 唯一pair门失败

**一句话：** UTSW每个人工病例有两版T1ce和两版人工标签，先在固定3例上双下载并叠物理网格，只接受唯一无歧义配对。

**状态：** TCIA官方来源审计已确认625病例、362人工细化、263自动-only，iteration-token完整目录为6349文件/24,667,030,300 bytes且9/11精确集合零偏差。桥接已完成固定3例双传输与数组QC；预注册唯一pair门失败，0模型。

**出发问题：** 362人工子集可逐例确定，但每例同时有`r`/非`r`两份manual correction和FeTS/ANTs两份T1ce。ReMIND物理grid失败说明不能按文件名猜接线；UTSW术前与当前治疗后训练又有真实域差异。

**具体做法：** 固定BT0001/2/3和每例5文件，共28,880,493 bytes；两次独立官方传输15/15 bytes/SHA exact后才读header/array。预列四种image×manual-label组合，只有恰好一个组合在三例shape/affine全过才确定未来路线；标签值/FeTS重叠只描述，不参与择优。

**改变/固定因素：** 只新增固定15文件来源桥接。其余622例、所有checkpoint、训练、旧cohort、阈值/TTA/SAM/后处理均不访问；不做重采样、换例、容差或模型分数选择。

**成功/停止门：** 双传输全exact、图像finite非空、标签整数/含0/含positive，且四配对中唯一一个三例几何全过；零个或多个都关闭当前实现。本结果不授权全包或外部评测。

**观察结果/直觉：** 双传输15/15 exact，所有数组finite/标签整数。两胜者都使用`rtumorseg_manual_correction`：它三例均240×240×155且与两份T1ce affine 0差；非`r`标签在BT0001/2是74×93×63、24×28×36裁剪块。标签全网格谱系得到支持，但两份T1ce均对齐，唯一pair直觉失败。

**代价/下一决定：** 墙钟约35秒、约56MB双副本、0GPU/0forward。当前实现停止，不把唯一pair门改成唯一label门、不用模型分数选FeTS/ANTs影像。保留UTSW362为未来候选；如继续须另立基于官方处理谱系的input-only门。证据：[来源结果](archive/reviews/2026-09-05_utsw_glioma_source_audit/RESULT.md)、[七问/最终填写](archive/research_records/2026-09-05_utsw_label_variant_bridge/IDEA.md)、[结果](archive/research_records/2026-09-05_utsw_label_variant_bridge/RESULT.md)、[机器结果](archive/research_records/2026-09-05_utsw_label_variant_bridge/RESULT.json)。

## UTSW官方主FeTS图no-N4输入桥接 ✅ 固定3例input-only门

**一句话：** 不靠模型分数选图，按UTSW论文的主FeTS流程固定无`_ants` T1ce，再验证它能否逐字节复用当前no-N4输入链。

**状态：** 固定BT0001/2/3的双副本转换已完成，128³ image/WT/geometry均3/3 exact；0模型、0GPU，其余359人工病例未访问。

**出发问题：** 上一几何桥接以两个胜者失败，但两胜者共享全网格`r`人工标签；歧义只剩FeTS/ANTs两种skull strip。UTSW论文把FeTS列为主预处理/初始分割，ANTs是额外版本，允许不用分割成绩预先定图。

**具体做法：** 固定`brain_t1ce`+`rtumorseg_manual_correction`，转RAS并把所有positive标签合为UTSW WT，再调用no-N4的非正清零、脑支持bbox、source z-score、128³图像/标签缩放；copy A/B独立重建。

**改变/固定因素：** 只按官方谱系固定主FeTS图；3病例、标签、双SHA、no-N4、crop/normalization/grid不变。禁用ANTs、N4、checkpoint和所有旧cohort。

**成功/停止门：** 三例源图finite/非负/有零背景与正脑，canonical几何全过，copy A/B输出image/label/geometry全exact；任一失败停止，不换版本或病例。成功也只证明input-only，不授权其余359例/模型。

**观察结果/直觉：** 三例源图均非负、min0、脑支持137.1万/151.9万/147.0万体素；image-label均240×240×155、affine 0差。所有positive UTSW标签映为WT后，两个独立副本的输出image/label/geometry全部exact，支持input-only直觉。首次运行被BraTS-only ID正则拒绝，原失败保留；仅放行固定3个BT ID后数值路径不变并通过。

**代价/下一决定：** 5项合同、约7秒CPU、复用约56MB，0GPU/0forward。当前不下载全362、不跑模型；成功只允许未来另立全量cache协议，且术前SRI24跨阶段证据必须与post-treatment榜分开。证据：[七问/最终填写](archive/research_records/2026-09-05_utsw_primary_fets_no_n4_input_bridge/IDEA.md)、[结果](archive/research_records/2026-09-05_utsw_primary_fets_no_n4_input_bridge/RESULT.md)、[机器结果](archive/research_records/2026-09-05_utsw_primary_fets_no_n4_input_bridge/RESULT.json)。

## UTSW主FeTS图canonical-N4输入桥接 ✅ 固定3例input-only门

**一句话：** 给未来same-seed canonical外部对照按其原训练配方做N4补光，不能让它错读no-N4输入。

**状态：** 固定BT0001/2/3的两个独立进程canonical N4均完成；跨进程image/label/geometry3/3 exact，对no-N4标签/几何相同、图像不同。0模型/0GPU。

**出发问题：** UTSW no-N4输入已3/3 exact，但单报候选绝对外部分数无法归因。UTSW主图非负/黑背景0，因此可直接测试canonical正值mask N4；RHUH负背景与Otsu重建负结果未覆盖这条路径。

**具体做法：** copy A/B在两个独立Python进程转RAS、all-positive→WT，调用原canonical N4 shrink2/四级迭代，再同bbox/z-score/128³；父进程比较输出并要求label/geometry与no-N4相同、image不同。

**改变/固定因素：** 相对已通过的UTSW no-N4桥接只加canonical N4。病例、源文件/标签、线程1、crop/grid和0模型固定。

**成功/停止门：** 跨进程image/label/geometry3/3 exact，且对no-N4 label/geometry3/3 exact、image3/3 different；任一失败停止，不重跑或改N4设置。

**观察结果/直觉：** copy A/B的image/label/geometry全部3/3 exact；相对已通过no-N4，label与crop geometry3/3 exact，N4 image3/3 different。支持“对照按原配方补光但保持同标签/几何”的工程直觉。

**代价/下一决定：** 两独立进程总381.9秒CPU、0GPU/0forward、不下载新源文件。通过只为未来全362 paired cache提供依据；no-N4资格、Full972与RHUH优先链未满足前不扩展。证据：[七问/最终填写](archive/research_records/2026-09-05_utsw_primary_fets_canonical_n4_input_bridge/IDEA.md)、[结果](archive/research_records/2026-09-05_utsw_primary_fets_canonical_n4_input_bridge/RESULT.md)、[机器结果](archive/research_records/2026-09-05_utsw_primary_fets_canonical_n4_input_bridge/RESULT.json)。

## UTSW主FeTS图Otsu-N4输入桥接 ✅ 固定3例input-only门

**一句话：** 在E/F结果前检查Otsu200拟合N4能否对同一UTSW主图跨进程复现，为“只有Otsu晋级”保留外部路线。

**状态：** 固定3例的两个独立进程均完成，image/label/geometry/Otsu mask跨进程3/3 exact；对canonical/no-N4标签/几何同、图像不同。0模型/0GPU。

**出发问题：** RHUH Otsu发布态桥接final image不exact；但其输入经过负背景解码/z-score，UTSW是直接非负主FeTS图。train Otsu机制/cache已过，旧失败未覆盖此来源。

**具体做法：** copy A/B独立进程、线程1运行OtsuThreshold200拟合mask的N4，再同bbox/z-score/128³；所有positive UTSW标签合为WT。与no-N4/canonical已归档输入比较。

**改变/固定因素：** 相对canonical只改N4拟合mask正值→Otsu200；病例/图/标签/shrink/iterations/geometry/grid固定，0模型。

**成功/停止门：** 跨进程image/label/geometry3/3 exact；对另两臂label/geometry同，image均3/3 different；任一失败停止且不改bins/线程/病例。

**观察结果/直觉：** fit mask体素133.8万/149.9万/143.1万，均为完整正支持的严格子集且只用于拟合；跨进程所有门3/3通过，支持UTSW直接非负主图的input-only确定性，不支持准确率主张。

**代价/下一决定：** 两进程371.3秒CPU、复用56MB、0GPU/0forward。Otsu Stage A现已失败，因此停止在input-only资产，不立全362 Otsu模型协议；工程成功与准确率失败必须分开。证据：[七问/最终填写](archive/research_records/2026-09-05_utsw_primary_fets_otsu_n4_input_bridge/IDEA.md)、[结果](archive/research_records/2026-09-05_utsw_primary_fets_otsu_n4_input_bridge/RESULT.md)、[Otsu Stage A](archive/research_records/2026-09-05_n4_otsu_clean_replication/RESULT.md)。

## UTSW362 canonical/no-N4配对外部确认 ❌ 条件门前关闭

**一句话：** 同一批362份异校术前试卷，让same-seed canonical与no-N4各按自己的完整输入配方作答，再逐人比较。

**状态：** IDEA/PREREG、19文件最终Amendment04 manifest与10项专用合同均在其余359例/模型前冻结，全相关56项测试通过；no-N4 Stage A失败后queue在23:35按`no_n4_stage_a_failed`边界正常关闭。EXTERNAL_PREACCESS、全量源、双cache、模型均未创建。

**出发问题：** RHUH条件链只有80 scans/40人且只报no-N4绝对分数；UTSW有362例人工细化，可提供大样本跨机构+跨阶段压力测试。单报候选绝对分无法归因，故必须same-seed paired control。

**具体做法：** 仅no-N4 E/F全门、Full972 canonical/no-N4完成、RHUH优先链终态且剩余≥4小时才冻结动态checkpoint。362个主FeTS T1ce+全网格r标签各双下载一次，724/724 size/SHA exact；构建canonical-N4/no-N4双cache，标签/geometry全同。两模型固定identity+Flip8、阈值0，患者paired bootstrap10k。

**改变/固定因素：** 唯一端到端差异是训练/推理输入是否canonical N4。362病例、all-positive WT、seed20261343、epoch168、架构、128³、阈值、视图和无后处理固定。

**成功/停止门：** Flip8 no-N4−canonical Scan>0、95% CI下界≥−0.003、Point最多少3/362；identity次要。双下载/cache/smoke/6516 forward任一工程门失败即停；模型结果无论正负只访问一次，不调参/重跑。

**代价/下一决定：** 计划双源3.63GB、双cache7.59GB、6516 forward均未发生，实际新增全量成本0。UTSW362来源与三种input-only桥仍是可用资产；当前no-N4专属模型协议关闭，不得换checkpoint。下一独立工作可预注册既有冻结baseline的UTSW一次性跨阶段评测，不能由本E/F选择新权重。证据：[七问](archive/research_records/2026-09-05_utsw_no_n4_paired_external/IDEA.md)、[终态](archive/research_records/2026-09-05_utsw_no_n4_paired_external/QUEUE_TERMINAL.json)、[运行时绑定](archive/research_records/2026-09-05_utsw_no_n4_paired_external/AMENDMENT_05_RUNTIME_BINDING.md)。

## UTSW362冻结baseline跨阶段外部基线 ❌ 标签语义门前关闭

**一句话：** 不再根据新E/F调学生，直接把当前总体代表KD168和1×Scan代表GCSAM184送到UTSW362术前人工队列考试。

**状态：** 两checkpoint synthetic smoke通过，双源724/724 exact；cache在raw label域门失败，0UTSW模型/指标。全362与异常非r探针均显示未定义label3/5，当前路线关闭。

**出发问题：** no-N4/Otsu都失败，不能绕门进外部候选链；但UTSW362高质量来源仍无模型基线，UCSD75又是uint8镜像。既有冻结模型可给出独立描述性外推证据。

**具体做法：** 双下载362主FeTS T1ce+r人工标签、724/724 SHA exact后构建单一canonical-N4 128³ cache。KD168与GCSAM184各跑identity和Flip8，患者bootstrap10k seed20261391；同视图差值只描述，因为两模型训练范围/优化器不同。

**改变/固定因素：** 唯一新增是UTSW362 cohort；模型/权重/N4/标签all-positive WT/128³/阈值0/视图/无SAM后处理均固定。两个模型已有test270和mirror-uint8 UCSD访问，权重没因结果改变。

**成功/停止门：** 工程全过且只评一次；KD Flip8、GCSAM identity各自参考门Scan/Subject≥0.70、Point≥326/362。未过也完整归档，不换模型或view；不授权旧cohort重访/训练。

**观察结果/直觉：** 模型冻结与smoke直觉支持，但来源标签门否定可评测性。全362 `r` manual union为0..5：3见116例/276万体素，5见5例/434体素；35异常病例非`r`原始manual仍0..5，不能归因于`r`重采样。论文只定义1/2/4三肿瘤区，无3/5映射。

**代价/下一决定：** 双源3.63GB已下载；首次单连接0payload失败，16固定批次后成功。partial cache约4KB blocks；synthetic36 forwards，但UTSW model forwards=0。没有发布者语义前不删5例、不排除3/5、不猜positive union。转向已下载UCSD75官方float图的同协议来源纠正，不用其分数选方法。证据：[结果](archive/research_records/2026-09-05_utsw_frozen_baseline_external/RESULT.md)、[标签域](archive/research_records/2026-09-05_utsw_frozen_baseline_external/LABEL_DOMAIN_DIAGNOSTIC_AMENDMENT_01.json)、[非r探针](archive/research_records/2026-09-05_utsw_frozen_baseline_external/NONR_ANOMALY_PROBE.json)。

## ReMIND T1ce/manual tumor来源探针 ❌ NRRD双传输门失败

**一句话：** 从43个合法人工标签候选中预先固定首例，只检查图像、肿瘤答案和脑mask的“插头”是否原生对齐。

**状态：** metadata形成43例候选；ReMIND-002下载门失败后关闭当前NRRD probe，未打开数组或模型。其余42例无像素访问。

**出发问题：** UCSD已揭示且来源量化，RHUH依赖no-N4资格；ReMIND为CC BY 4.0、BWH单站、113例manual preop tumor，但raw DICOM未去颅骨且NRRD几何未实测。

**具体做法：** 用官方series、完整`.sums`和临床表，在像素前固定43例Glioblastoma/Astrocytoma/Oligodendroglioma且T1ce-referenced manual tumor的队列。首例ReMIND-002只下载T1ce、manual tumor及automatic cerebrum；因`.sums`全为空MD5，修订为NRRD两次独立官方传输bytes/SHA exact，再无重采样做UID/几何/二值/包含关系QC。

**改变/固定因素：** 只增加一个固定外部来源探针；病例/series/label路径/1e-5几何门固定，不做预处理选择、不加载checkpoint。

**预期机制：** 先确认源格式与标签物理空间，避免整批下载后才用GT修接线；automatic cerebrum若被未来使用必须标provided-mask assisted。

**观察结果：** 43例为19/15/9三诊断，34例有同T1ce cerebrum。NBIA包192个DICOM bytes与官方exact，另含LICENSE导致实现计数假失败；首轮两个NRRD size通过，第二轮Aspera在文件前SSH连接失败，双传输门失败。未运行几何/标签值QC。

**直觉是否支持：** 支持ReMIND metadata存在高质量候选，不支持“当前NRRD probe已建立可复现内容链”。网络失败不否定一般来源。

**成功/停止门：** 两次NRRD字节exact、DICOM UID/实例、原生几何、二值/非空、tumor⊂cerebrum全过；第二传输失败即按规则停止，不补传/换例/放宽。

**代价/下一决定：** 5项合同，下载8.90MB ZIP和首轮1.95MB NRRD，0GPU/0forward/0数组QC；临时数据保留。若继续只允许另立DICOM SEG交付路线并保持同首例/43例，不修改当前失败实现。证据：[七问](archive/research_records/2026-09-05_remind_t1ce_source_probe/IDEA.md)、[结果](archive/research_records/2026-09-05_remind_t1ce_source_probe/RESULT.md)、[来源审计](archive/reviews/2026-09-05_remind_source_audit/RESULT.md)。

## ReMIND固定首例DICOM SEG来源探针 ❌ 源间隔exact门失败

**一句话：** 不重试失败NRRD，改用同一TCIA collection的DICOM SEG，并按每帧引用的源SOP UID对齐同一首例。

**状态：** 五个NBIA下载与重复内容门全过；T1ce header投影间隔稳定门失败，在任何T1ce/SEG pixel array前关闭。0模型。

**出发问题：** ReMIND43例metadata候选质量高，但NRRD第二轮SSH失败；同一ReMIND-002有明确引用T1ce的manual tumor/cerebrum DICOM SEG，发布论文说明由DCMQI转换。

**具体做法：** 固定pydicom3.0.1 wheel/解压manifest，不安装venv；T1ce重下并与旧192 DICOM逐文件SHA，两个SEG各下载两次；随后计划按ReferencedSOPInstanceUID/物理位置无插值重建mask。

**改变/固定因素：** 只改官方标签交付格式NRRD→DICOM SEG；ReMIND-002、43例定义、series、标签、几何门和0模型不变。

**预期机制：** 图像和标签在同一DICOM引用体系中应消除外部NRRD传输/映射不确定性；必须先过源slice与frame几何门。

**观察结果：** T1ce 192 DICOM与旧副本exact，tumor/cerebrum两次DICOM各exact。192位置/SOP/Instance unique、方向一致、声明spacing/thickness1mm；但投影gap范围0.999348–1.000722mm，相对中位最大误差0.000860mm，超冻结1e-5门。

**直觉是否支持：** 支持DICOM交付字节可复现；不支持当前exact几何实现全门通过。可能是DICOM小数精度，但结果后不调容差。

**成功/停止门：** 下载复现、source gap稳定、SEG UID/frame/geometry、二值/非空、tumor⊂cerebrum全过；source gap先失败即停止，不读pixel array/换decoder/病例/容差。

**代价/下一决定：** 4项合同、临时pydicom不安装，新ZIP约11.28MB，0GPU/0forward/0数组。当前轮不再围绕ReMIND-002调工程门；保留43例metadata但不跑模型。证据：[七问](archive/research_records/2026-09-05_remind_dicomseg_source_probe/IDEA.md)、[结果](archive/research_records/2026-09-05_remind_dicomseg_source_probe/RESULT.md)、[来源审计](archive/reviews/2026-09-05_remind_source_audit/RESULT.md)。

## ReMIND量化感知/物理grid映射 ❌ SEG spacing区间门失败

**一句话：** 先让有限精度DICOM坐标变成可信区间，再像按经纬度叠地图一样把不同SEG grid最近邻映射到T1ce。

**状态：** Source量化区间机制通过；物理grid实现随后在两个SEG内部spacing区间门失败，SEG pixels/映射/模型均未发生。本轮ReMIND线关闭。

**出发问题：** Fixed1e-5 source gap对0.001mm坐标文本过严，而DICOM SEG又是独立裁剪/分辨率grid且无SOP逐帧引用，需要标准物理坐标映射。

**具体做法：** 按每个DS字符串末位计算半量化区间；验证source/SEG frame gap与声明spacing兼容，再构造SimpleITK物理grid，以nearest一次映到T1ce，预设bbox/5%物理体积/tumor⊂cerebrum门。科学记数DS用Decimal exponent解析。

**改变/固定因素：** 只改几何验证/映射为量化感知physical-grid；同ReMIND-002、下载、decoder、nearest、5%、标签和0模型固定。

**预期机制：** 避免用比DICOM刻度更细的固定容差，同时显式处理不同grid，不靠GT或模型分数修接线。

**观察结果：** Source191区间全过；tumor gap0.999994–1.000002mm、最大层残差5.79e-6，cerebrum gap0.399576–0.399591mm、最大层残差2.42e-5，均超各DS严格区间。相邻frame都是一层，并非稀疏跳片。

**直觉是否支持：** Source量化机制支持，但“DS末位区间足够覆盖DCMQI转换误差”不成立；物理resample尚未执行，不能判断映射质量。

**成功/停止门：** source/SEG规则grid、bbox在source内、nearest后非空、物理体积误差≤5%、tumor⊂cerebrum全过。SEG内部门先失败后停止，不再按已见header加第五种容差。

**代价/下一决定：** 两阶段各3项合同、0新网络/0GPU/0forward；T1ce pixel已读，SEG pixel仍未读。43例metadata保留到未来独立窗口，本轮不跑ReMIND模型。证据：[量化结果](archive/research_records/2026-09-05_remind_quantized_geometry_probe/RESULT.md)、[物理grid结果](archive/research_records/2026-09-05_remind_physical_grid_probe/RESULT.md)。

## 标准化MRI背景零点解码 ❌ 完整链等价性失败

**一句话：** 把已线性标准化MRI里的恒定黑背景移回零，再沿用原输入链。

**出发问题：** RHUH80官方影像全含负值，未经解码的正值N4格式门失败。只读抽查证明负数包含被移动的黑背景，不能把所有负值直接置零。

**具体做法：** 取六面精确众数，要求≥95%表面一致且为全图最小值；整图减背景。先在合成图和首3份train972图验证支持/包围盒完全一致、原始标准化及含N4完整输入链误差不超过冻结容差。通过后才检查80例源格式；不读标签、不跑模型。

**改变/固定因素：** 唯一变化为文件零点解码；N4、brain crop、源非零z-score和128³固定。无参数训练或推理配方改变。

**预期机制：** 标准化后的背景值减去后恢复原信号的全局比例，而后续标准化消除比例；这需要数值验证，不声称精确原始强度恢复。

**状态/结果：** 7项合成合同通过；3 train图×2编码中支持/包围盒全部exact、源z-score误差均<1e-6；完整N4输入链仅4/6通过，最大/平均误差最坏0.057679/0.002245，超过冻结0.001/0.00001。无标签/模型或Dice访问。

**直觉是否支持：** 只支持线性信号恢复，不支持“经过当前N4仍数值等价”。不能用数学比例不变替代实际数值检查，也不能据此判断Dice更差。

**代价/下一决定：** 121.50秒CPU、0 GPU/0模型forward，无大cache/新checkpoint。关闭当前等价性主张，不放宽门、不跑RHUH解码QC或外部评测。下一步只审计N4尺度敏感机制，若实施新方法必须新预注册。证据：[七问](archive/research_records/2026-09-05_background_zero_decode/IDEA.md)、[结果](archive/research_records/2026-09-05_background_zero_decode/RESULT.md)、[raw](archive/research_records/2026-09-05_background_zero_decode/TRAIN_EQUIVALENCE.json)。

## GCSAM固定epoch184双新seed复制 ❌ Stage A

**一句话：** 不改变GCSAM配方，把同一场AdamW与GCSAM比赛在两套新的训练划分和随机种子上各重做一次。

**出发问题：** 原GCSAM在train-only inner只提高Scan/Subject `+0.000807/+0.002558`，两项CI下界`-0.005464/-0.005327`失败；它虽是UCSD75三项点估计和internal test270 Scan点估计第一，但优势均不确定且不能回流调参。

**具体做法：** 从Full972 train角色预先生成两个subject-disjoint `864 fit/108 inner`复制。每个复制都以同一初始state、sampling和增强配对AdamW control与固定`rho=0.05` paper-form GCSAM candidate，训练到固定epoch184。四条训练全部完成前不打开inner；随后两个inner各只评一次。

**改变/固定因素：** 每个pair只改变student optimizer update。模型4.70M、T1ce/WT、history/EMA teacher、DiceCE加逐体素KL知识蒸馏、batch8、184轮、阈值0、无测试时增强/Segment Anything Model/后处理和1×部署均固定；不调`rho`或epoch。

**预期机制：** 若GCSAM真的更容易落在宽谷底，不应只在一个seed或少数外部病例上表现好，而应在两个新inner中都同时提高Scan和Subject。

**成功/停止门：** 每个复制Candidate都须严格提高Scan/Subject、Point最多少1/108、两项paired CI下界≥`-0.003`；两个复制平均delta也须为正，工程配对全部exact。任一失败立即停止，不做Full972、不访问fixed validation/test270/UCSD75、不加第三个seed。

**观察结果：** 四条固定184轮训练均正常完成；两个pair的初始state、sampling weights、全部184轮sampler、首增强batch和first-pass loss exact。冻结四个checkpoint后才各读取一次inner。复制A Candidate/Control Scan=`0.737953/0.739167`、Subject=`0.726095/0.722541`、Point=`104/103`，差`-0.001214/+0.003554/+1`；复制B为`0.767425/0.767249`、`0.746052/0.748282`、`102/104`，差`+0.000176/-0.002230/-2`。A Scan/Subject CI=`[-0.008946,+0.006706]`/`[-0.009746,+0.017919]`；B为`[-0.005494,+0.006665]`/`[-0.010571,+0.007987]`。两个复制平均Scan/Subject差=`-0.000519/+0.000662`。

**直觉是否支持：** 不支持。A只提高Subject/Point而降低Scan；B的Scan只升`0.000176`却降低Subject并少2 Point，两组都未形成一致的Dice方向且四个CI安全门失败。旧epoch184在reused test270/UCSD75的高点估计不能解释为稳定、可复制的优化器优势。这否定当前固定配方，不证明所有锐度感知优化永远无效。

**2026-09-06新外部补记：** 原冻结GCSAM184在Africa95相对Full identity Dice=`+0.068617`且CI全正，但Point少3；CPTAC/WAW-pre/WAW-post差值小正且CI跨0，连同internal/UCSD六cohort只有Africa确定。**同日晚些时候新增的UCSD-BraTS2024test56（patient-disjoint、治疗后、近训练域）上GCSAM−Full=`−0.012312`，原冻结scan/subject配对CI略负（后续Scan数值精度审计不支持稳健显著下降）且Point−4；七cohort中方向可逆，"GCSAM外部普遍偏高"措辞作废。**五外部保存计数又显示GCSAM预测体积和recall点估计5/5均高，precision仅2/5正、Point从未增加；recall CI只在Africa/CPTAC全正。这支持checkpoint coverage倾向，不支持optimizer因果或全面冠军。两checkpoint同时不同fit/epoch/seed/选择；不得用已见cohort选择`rho`、volume router或重训。[coverage综合](archive/reviews/2026-09-06_gcsam_coverage_synthesis/RESULT.md)、[跨cohort](archive/reviews/2026-09-06_cross_cohort_checkpoint_synthesis/RESULT.md)、[因果边界](archive/reviews/2026-09-06_gcsam_external_interpretation/RESULT.md)。

**代价与下一决定：** 四条formal共19.14 GPU-hours，双卡分阶段墙钟约9.59h；GCSAM比AdamW慢`1.578×/1.581×`，峰值约20.36GiB，workspace约109M，推理仍1×。Stage A失败后关闭分支，不做Full972、不访问fixed validation108/outer/confirmation/test270/UCSD75，也不调`rho`、epoch、seed或阈值。证据见[七问](archive/research_records/2026-09-03_gcsam_fixed184_replication/IDEA.md)、[结果](archive/research_records/2026-09-03_gcsam_fixed184_replication/RESULT.md)、[机器结果](archive/research_records/2026-09-03_gcsam_fixed184_replication/RESULT.json)与[raw inner](archive/research_records/2026-09-03_gcsam_fixed184_replication/RAW_INNER.json)。

## T1ce脑区固定1–99百分位裁剪 ❌ Stage A

**一句话：** 像拍照前先剪掉最刺眼和最黑的各1%坏点，再统一每例MRI的亮度尺子。

**出发问题：** Full972 KD的fixed validation最高Scan为`0.740013`，而Gamma、GCSAM、Lookahead等训练配方跨seed不稳；brain crop曾有约`+0.054`可靠收益，说明输入表示仍比堆优化器更值得检验。当前N4后直接用所有非零体素做z-score，可能被少量极端强度牵动。

**具体做法：** Candidate在N4与source非零总体z-score之间，按每例脑内固定1st/99th percentile裁剪强度，再z-score并缩至128³。之后仍走同一4.70M SegResNet、history/EMA teacher、DiceCE+逐体素KL知识蒸馏，输出raw logit阈值0和argmax point。

**改变/固定因素：** 唯一变化是是否执行固定分位裁剪。两个新864/108 subject-disjoint split、每pair初始state/采样/增强、模型、teacher、loss、AdamW、固定epoch168、阈值0、无测试时增强/SAM/后处理和1×推理全部固定。

**预期机制：** 若扫描间强度极值正在扰动归一化，确定性去极值应在两个fresh seed中都提高Scan与Subject，而不是只靠少数病例或一个seed。Gamma随机扭曲亮度曲线的失败没有直接测试这个机制。

**成功/停止门：** 每个复制Candidate须严格提高Scan/Subject、Point最多少1、两项paired CI下界≥`-0.003`，两个复制平均Dice差也为正；任一失败立即关闭且不加seed/调百分位。即使全过也只训练Full972冻结候选，不访问已揭示fixed validation/outer/confirmation/test270/UCSD75。

**观察结果：** A Candidate/Control Scan=`0.721939/0.730148`、Subject=`0.704389/0.714339`、Point=`102/106`，差`-0.008210/-0.009950/-4`；B为`0.731047/0.738568`、`0.724639/0.734514`、`103/98`，差`-0.007520/-0.009875/+5`。A Scan/Subject CI=`[-0.016222,-0.001269]`/`[-0.022694,-0.000456]`；B=`[-0.012519,-0.002623]`/`[-0.017121,-0.003004]`。两复制平均Scan/Subject差=`-0.007865/-0.009912`。

**直觉是否支持：** 不支持。两个独立split的Dice均下降且四个CI上界低于0，说明当前固定裁剪稳定损害mask泛化。B的Point多5个、最终fit Dice也略高，但inner Scan/Subject仍下降，说明亮度规整可能保住最高峰却删掉边界判别信息；Point不能代替Dice。

**代价与下一决定：** Cache标签与Control整文件exact、5/5独立重建、11项CPU合同、双卡smoke、四条168轮训练和A/B配对门均通过。Formal共13.55 GPU-hours/双卡6.78h；临时cache9.5G在`/tmp`，workspace110M，峰值约20.29GiB，部署仍1×。Stage A失败后不训练Full972，不调百分位/顺序/epoch/teacher/阈值，不访问fixed validation/outer/confirmation/test270/UCSD75，Leaderboard不变。证据见[IDEA](archive/research_records/2026-09-03_t1ce_percentile_clip_stability/IDEA.md)、[结果](archive/research_records/2026-09-03_t1ce_percentile_clip_stability/RESULT.md)、[机器结果](archive/research_records/2026-09-03_t1ce_percentile_clip_stability/RESULT.json)与[raw inner](archive/research_records/2026-09-03_t1ce_percentile_clip_stability/RAW_STAGE_A_INNER.json)。

## 镜像前后答案一致训练 ❌ fresh C/D失败并关闭

**一句话：** 训练时让同一MRI正着看和镜像看，镜像答案翻回来后必须一致，希望模型正着一次和8镜像平均都更准。

**状态：** 四条fixed168训练、四checkpoint预访问冻结和fresh C/D inner一次性评测全部完成；主门和跨复制门失败。没有访问fixed validation、outer、confirmation、test270或UCSD75。

**出发问题：** test访问前的Flip8 A/B让Scan/Subject平均提高`+0.006828/+0.007132`，说明方向敏感可复现，但8×推理昂贵。这个实验尝试只用新train-role证据把等变性学进模型。

**具体做法：** C/D各864 fit/108 inner、各49 inner subjects，彼此case/subject互斥并排除旧A/B inner的93个subjects。每复制Control/Candidate同初值、sampler、增强、teacher和两次student forward；只把平方分母sigmoid soft-Dice一致性梯度从0改为0.1。四模型全部完成后才打开inner，同时评identity1×与固定Flip8 8×。

**改变/固定因素：** 唯一变化是一致性weight0/0.1。模型4.70M、T1ce/WT、brain crop true128³、DiceCE+逐体素知识蒸馏、native-volume sampling、AdamW、168轮、reflection cycle、阈值0、无Segment Anything Model/后处理全部固定。

**预期机制：** 若Flip8收益来自模型不严格等变，直接减少最终mask镜像差异应同时改善1×与8×。旧KAS是晚期feature投影加三loss，未提前否定本实现。

**成功/停止门：** 每复制Candidate Flip8须严格提高Scan/Subject、Point最多少1、两项CI下界≥`-0.003`；identity也须严格提高Scan/Subject；末5轮一致性≤Control 0.90×。跨复制Flip8 Scan平均>`0.001`且四项Dice平均方向为正。任一失败停止，不调weight/formula/view/epoch/seed，不进旧cohort。

**观察结果：** C identity Candidate−Control=`+0.000316/+0.001386/+1`，Flip8=`-0.001773/+0.000643/+2`，Flip8 CI=`[-0.006636,+0.003320]/[-0.005893,+0.007807]`。D identity=`-0.000546/+0.000085/-1`，Flip8=`-0.002063/-0.003967/-1`，CI=`[-0.005778,+0.001629]/[-0.009182,+0.000867]`。跨复制平均Flip8 Scan/Subject=`-0.001918/-0.001662`，identity Scan/Subject=`-0.000115/+0.000735`；整体失败。

**直觉是否支持：** 不支持准确率主张。机制确实学到：末5轮Candidate/Control一致性loss比为C`0.636`、D`0.648`，即差异少约35–36%；但更一致会把错误也稳定重复，未变成更准。只否定当前weight0.1、平方soft-Dice、单reflection循环实现，不证明所有群等变架构不可能。

**代价与下一决定：** 四训练约20.873 GPU-hours/双卡两波10.484h，每卡峰值42.006GB；inner评测3,456 forwards；workspace110MB，无新cache。原普通soft-Dice公式混入锐化，在训练前以AMENDMENT_01透明修正，未看inner。关闭分支，不调参、不加seed、不进榜。C/D已经揭示，下一内部方法必须新E/F并排除A/B/C/D inner subjects。证据见[结果](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/RESULT.md)、[机器结果](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/RESULT.json)、[raw inner](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/RAW_STAGE_A_INNER.json)、[preaccess](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/INNER_PREACCESS.json)与[修订](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/AMENDMENT_01.md)。

## 全自动Direct Locator固定Flip8测试时增强 ✅ 完成；reused test270三项点估计第1

**一句话：** 同一模型把MRI按三轴全部8种镜像方向都看一遍，把答案翻回原方向后平均。

**状态：** 两个fresh inner复制全门通过；随后唯一一次reused internal test270四个成功门全过。Africa95、CPTAC33、WAW preop152与WAW postop31也按冻结面板运行；五处cohort关闭，禁止调参或重测。

**出发问题：** Full972 KD168旧test Scan=`0.751760`，距GCSAM榜首`0.752003`仅`0.000244`，而Subject/Point更高；训练随机覆盖全部三轴翻转但部署只看identity，可能留下方向偶然误差。

**具体做法：** A/B各用新subject-disjoint 864 fit/108 inner训练一个标准Full972-style KD base到固定epoch168，两个checkpoint全部完成并冻结后才访问inner。同一checkpoint依次看完整8种reflection，把raw logits逆翻回原方向并按固定顺序float32等权平均，阈值0、Point=argmax。

**改变/固定因素：** 唯一变化是推理1×→固定8×。数据、模型、checkpoint、teacher、loss、采样、增强、AdamW、epoch168、阈值0、无Segment Anything Model/连通域/后处理全部固定；没有搜索reflection子集、权重或概率平均。

**预期机制：** 训练翻转没有让卷积网络严格等变时，8种等价视图的逆变换平均会消除方向敏感噪声。旧TTA只在GT-centered refiner支线，未否定全自动direct locator。

**成功/停止门：** 每个fresh inner均须Scan/Subject严格提高、Point最多少1、两项CI下界≥`-0.003`；平均Scan须超过`0.000243638`。全部通过后只允许一次固定test；test须Scan超过GCSAM与identity、Subject不低于identity、Point最多少1。

**观察结果：** A/B fresh平均Scan/Subject=`+0.006828/+0.007132`。Test270 Scan差`+0.005510`、CI全正，Subject CI跨0，Point`+3`。Africa95=`+0.003435`、CI跨0、Point`+1`；CPTAC33=`+0.003246`、CI全正、Point`0`；WAW-pre152=`+0.010116`、CI全正、Point`+2`；WAW-post31=`+0.007312`、CI跨0、Point`0`。

**直觉是否支持：** 支持方向、但强度受限。两fresh inner及五个报告cohort的Scan点估计都正；internal、CPTAC与WAW-pre区间支持，Africa/WAW-post不确定。四个有完整保存计数的外部队列中，Flip8 precision点估计4/4为正，Africa/WAW-pre/WAW-post的precision区间全正；recall仅CPTAC区间全正，预测体积通常更小。这支持“通过镜像共识减少偶然多画”的保守倾向，不证明普适Dice显著。不能合并cohort成总体效应。

**代价与下一决定：** Stage A约6.783 GPU-hours/双A100墙钟3.404h、1,728 forwards；test 2,160 forwards。部署仍4.70M参数/同一checkpoint，峰值近1×但计算延迟约8×。跨外部误差综合只读JSON、0 forward。冻结当前8视图公式；五个已运行cohort全部关闭，绝不按结果调视图/权重/阈值或重测。证据见[原实验结果](archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/RESULT.md)、[跨cohort只读综合](archive/reviews/2026-09-06_flip8_cross_cohort_synthesis/RESULT.md)、[误差机制综合](archive/reviews/2026-09-06_flip8_overlap_synthesis/RESULT.md)、[raw test](archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/RAW_TEST270.json)与[Stage A](archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/STAGE_A_INNER.json)。

## CPTAC33 Africa-informed确认 ✅ 完成并关闭

**一句话：** 在Africa看到GCSAM大幅领先后，用另一套CPTAC公开人工标签复考同两checkpoint，检查方向能否独立复现。

**出发问题：** Africa95 GCSAM−Full identity=`+0.068617`且CI全正，但只在四个中心，两个checkpoint训练也非单因素。TCIA映射给出CPTAC39，其中public Training33有seg、Validation6无seg。

**具体做法：** 原39门因Validation首例缺seg失败后，像素前另冻Training33；官方407,245行`.sums`覆盖66选中文件，双源官方MD5/size/跨副本SHA全过。All-positive术前WT经相同N4/crop/z-score/128³；Full identity+Flip8、GCSAM identity一次性297 forwards。

**改变/固定因素：** 唯一新增CPTAC33 cohort；面板/模型/阈值/视图/统计与Africa相同。Africa-informed primary在CPTAC payload前冻结；Validation6和hidden test均不访问。Public Training labels是external-to-project，不是原challenge validation/blind。

**观察结果：** Full identity/Flip8/GCSAM=`0.841331/0.844577/0.843285`，Point全33/33。GCSAM−Full=`+0.001954`，CI `[-0.006646,+0.012384]`，16胜17负，按门仅弱/不确定支持；Africa的大幅优势未复现。Flip8−identity=`+0.003246`，CI `[+0.000828,+0.005855]`全正，23胜10负，独立支持固定Flip8 Dice方向。

**直觉是否支持：** Africa的GCSAM方向符号相同但强度/确定性不复现，说明其优势域特异；Flip8在CPTAC重新得到小而一致的Dice收益。三个模型target/prediction体积接近且Point全中，与Africa系统漏分不同。

**代价/下一决定：** 训练0；sums47.5MiB、双源149MiB、cache330MiB/N4 288秒、297 forwards 6.55秒/5.08GiB。首次formal证据字段KeyError在模型前0forward修复，V2 preaccess后唯一运行。Source-disjoint/no patient bridge、训练源parity caveat保留。Cohort关闭，不调参/重跑。证据[结果](archive/research_records/2026-09-06_cptac33_africa_informed_confirmation/RESULT.md)、[机器结果](archive/research_records/2026-09-06_cptac33_africa_informed_confirmation/RESULT.json)、[raw](archive/research_records/2026-09-06_cptac33_africa_informed_confirmation/RAW_CPTAC33.json)、[验证](archive/research_records/2026-09-06_cptac33_africa_informed_confirmation/VERIFICATION_RECEIPT.json)与[来源](archive/reviews/2026-09-06_cptac_brats2021_source_audit/RESULT.md)。

## WAW-GliomAI152冻结checkpoint面板 ✅ 完成并关闭

**一句话：** 把两位冻结“医生”和同一位医生的八镜像复核，带到华沙152名公开术前胶质瘤患者只考一次，不看成绩换答案。

**出发问题：** Reused internal270总体Full Flip8=`0.757270`但非盲；GCSAM在Africa95相对Full identity高`0.068617`且CI全正，在CPTAC33却只高`0.001954`且CI跨0。项目缺少另一地区、公开专家人工标签的大队列区分队列特异现象和可复现方向。

**具体做法：** Zenodo final record22205631描述156份preop，但最终coreg T1c与preop mask精确交集实际152人；按最终文件在MRI大包前冻结。Delayed post-contrast T1c与all-positive标签1/2/3经canonical RAS/N4 shrink2/image-only crop/z-score/128³；Full KD168同时输出identity/固定Flip8，GCSAM184 identity，一次性1368 forwards并做10,000次病例bootstrap。

**改变/固定因素：** 唯一新增WAW152 cohort。病例、模型/checkpoint、预处理、WT、阈值0、八视图顺序与float32等权raw-logit融合、无SAM/后处理/模型ensemble、统计与解释门全部冻结。华沙不在BraTS2024七站，但无patient-ID桥，只称institution-disjoint/no-known-overlap；public非hidden blind。

**预期机制与门：** 若GCSAM优势跨术前域稳定，GCSAM−Full应为正且配对Scan CI下界>0；若Flip8方向稳定同理。正差但CI跨0只算不确定，非正算不支持；不设performance promotion。大包官方size/MD5/CRC、152 geometry/label、首例byte复现、cache、checkpoint、smoke和preaccess任一失败即模型前停止。

**观察结果：** GCSAM/Full Flip8/Full identity Scan=Subject Dice=`0.693601/0.693494/0.683378`，Point=`140/143/141`。GCSAM−Full=`+0.010223`，CI `[-0.003005,+0.023481]`、84/63/5、Point−1，按门正向但不确定；Africa强差不稳定复现。Flip8−identity=`+0.010116`，CI `[+0.004472,+0.016274]`、93/54/5、Point+2，支持WAW上的固定八视图收益。GCSAM只比Flip8 Dice高`0.000107`且Point少3，没有统一冠军。

**直觉是否支持：** 部分支持。新队列再次显示GCSAM点估计可高于Full identity，但区间仍跨0，支持“依队列变化”而非普适优势；Flip8获得第三处Scan配对CI全正证据，方向更稳，但付出8×推理。按“空预测precision=1”的统一口径，事后计数分解显示Flip8 precision约`+0.04185`、CI `[+0.01957,+0.06881]`全正，recall近零且CI跨0；GCSAM precision/recall差均跨0。旧V1误记空预测precision为0，V2修正不影响Dice/Point。三模型共同5个零Dice均grade2，4例只有FLAIR label2；train纯only-label2仅3.09%而WAW17.11%。GCSAM差在Architect/其他=`+0.01365/+0.00063`、3T/1.5T=`+0.01611/−0.00341`，组成敏感；体积/grade/label/scanner均不得用作router。

**代价/下一决定：** 训练0；官方大包2.95GiB、选中T1c约749MiB、cache约1.48GiB/N4 630.7秒；正式26.49秒、1368 forwards、峰值5.08GiB。首次formal因预访问字段路径KeyError在checkpoint/model前0forward，V2后唯一运行。终态重hash大包/304源文件/checkpoints/cache/五个bootstrap/榜单全过。WAW关闭，不调参/重跑。证据[结果](archive/research_records/2026-09-06_waw_gliomai152_frozen_panel/RESULT.md)、[机器结果](archive/research_records/2026-09-06_waw_gliomai152_frozen_panel/RESULT.json)、[raw](archive/research_records/2026-09-06_waw_gliomai152_frozen_panel/RAW_WAW152.json)、[验证](archive/research_records/2026-09-06_waw_gliomai152_frozen_panel/VERIFICATION_RECEIPT.json)、[事后误差V2](archive/research_records/2026-09-06_waw_gliomai152_frozen_panel/POSTHOC_ERROR_AUDIT_V2.json)、[失败尾部](archive/research_records/2026-09-06_waw_gliomai152_frozen_panel/POSTHOC_FAILURE_TAIL.json)与[来源](archive/reviews/2026-09-06_waw_gliomai_source_audit/RESULT.md)。

## WAW术后only-patient31冻结面板 ✅ 完成并关闭

**一句话：** 从同一公开库排掉所有已考preop患者，只取31名新患者最早且日期明确的术后scan，给冻结模型做同任务小考。

**出发问题：** Africa/CPTAC/WAW preop均术前，UCSD official是来源纠正；项目缺少与BraTS2024 post-treatment语义接近、又未见患者的公开专家队列。WAW final postop可按同release ID直接排重。

**具体做法：** Final160 exams/92 patients中排57 preop-overlap；35 post-only里排4 blank-day患者，再从31人/48 numeric scans每人固定最早一张，day0–22。WT labels1/2/3、RC4 background；官方3.322GB zip MD5/CRC、31源、首例、全cache、smoke/preaccess过后一次性Full identity+Flip8/GCSAM identity 279 forwards。

**改变/固定因素：** 唯一新增patient-disjoint-from-preop31 early-postop cohort；病例/时间点在性能前冻结。模型、checkpoint、N4/crop/z-score/128³、阈值、view、无SAM/后处理、bootstrap均不变。Public同Warsaw source、相对BraTS七站institution-disjoint/no patient bridge、非hidden blind。

**门与观察：** Full identity/Flip8/GCSAM Dice=`0.591647/0.598959/0.593305`，Point=`15/15/14`。Flip8−identity=`+0.007312`、CI `[-0.001925,+0.016626]`、21/10，Point0；GCSAM−Full=`+0.001657`、CI `[-0.012498,+0.016205]`、17/14，Point−1。两者按门均positive-but-uncertain，无确定winner。

**直觉是否支持：** 支持“新患者早期术后更难、Point与Dice继续分离”；不支持GCSAM或Flip8确定优越。Full/Flip各16个Point miss中13个落GT RC，GCSAM17个中14个；GT RC不可部署。Postop/preop粗域近似；grade4-only pre74/post29 Dice仍相差约`0.144–0.150`且post WT更小，排除grade/大WT充分解释，但患者不配对、RC状态不同，非手术因果。

**代价/下一决定：** 训练0；source3.09GiB、selected147MiB、cache约310MiB/141.9秒；279 forwards5.76秒/5.08GiB。终态重hash官方zip/62源/checkpoints/cache/五bootstrap/榜单全过。Cohort关闭，不换后续/空日期scan、不调参/重跑/GT-RC避让。证据[结果](archive/research_records/2026-09-06_waw_postop_only31_frozen_panel/RESULT.md)、[机器](archive/research_records/2026-09-06_waw_postop_only31_frozen_panel/RESULT.json)、[raw](archive/research_records/2026-09-06_waw_postop_only31_frozen_panel/RAW_WAW_POSTOP31.json)、[验证](archive/research_records/2026-09-06_waw_postop_only31_frozen_panel/VERIFICATION_RECEIPT.json)、[事后错误](archive/research_records/2026-09-06_waw_postop_only31_frozen_panel/POSTHOC_ERRORS.json)与[来源](archive/reviews/2026-09-06_waw_postop_only31_source_audit/RESULT.md)。

## OpenNeuro-GBM334冻结checkpoint流式面板 ❌ 输入门失败/0性能

**一句话：** 把同两份冻结checkpoint带到八家新医院334名术前胶质母细胞瘤患者，只换队列、只考一次；为节省磁盘逐例下载、校验、处理后释放。

**状态：** 2026-09-06已完成一手来源核验、v1.0.1全对象树、像素前334例交集、七问/预注册、5项合同测试、首例双重预处理复现和334例全输入门。332例通过，2例零WT crop loss门失败；未生成MODEL_PREACCESS，0外部模型/性能，关闭。

**出发问题：** GCSAM相对Full identity只有Africa95配对CI全正，在CPTAC/WAW均不确定；Flip8五cohort Scan点估计同向但8×且两处区间跨0。需要一个更大、全新医院、expert-reviewed/refined的公开队列缩窄不确定性。

**具体做法：** OpenNeuro 337 participants中，版本化未N4 MNI T1-CE与最终mask严格交集334；BO199/NSK07缺mask、BO97缺对应影像，像素前排除。每例SHA后走canonical RAS/N4/image-only crop/z-score/128³；全输入门过后才允许Full identity+固定Flip8和GCSAM identity共3006 forwards。

**改变/固定因素：** 唯一新增334例cohort。Checkpoint、N4、WT=`{1,2,3}`、阈值0、八视图和raw-logit融合、无SAM/后处理、bootstrap与解释门全固定。八站不在BraTS2024七站，但无patient桥，只称institution-disjoint/no-known-overlap。

**成功/停止门：** 334/334 versionId SHA、label0/1/2/3、geometry≤`1e-5`、WT非空、零WT crop loss及首例byte复现必须全过；任一失败即0模型或精确记录访问边界，不丢例/换variant/扩框。性能只按配对CI三档完整报告，不触发调参或重跑。

**观察结果：** 首例BA2 shape=`182×218×182`、1mm RAS、affine error0、raw WT76549、crop loss0、resized WT43920；两次独立image/label SHA exact。全量source SHA、label union0/1/2/3和affine全过，但BO118有4/348742、KA86有6/97700个WT体素在image-positive crop外；零丢失门失败。不得写332例性能。

**直觉是否支持：** 不支持当前输入实现的无损接入；尚未检验模型准确率直觉。失败比例虽极小，仍不能看见GT后扩框。它只否定现成canonical image-positive crop直接用于该334例，不否定数据集或未来独立全FOV模型。

**代价/下一决定：** 训练0；12 workers流式2.03GiB、墙钟48.48分钟、常驻cache0；9 synthetic forwards峰值0.66GiB；外部模型forward0。关闭，不排两例、不裁GT、不搜padding/variant。证据[结果](archive/research_records/2026-09-06_openneuro_gbm334_frozen_panel/RESULT.md)、[机器结果](archive/research_records/2026-09-06_openneuro_gbm334_frozen_panel/RESULT.json)、[输入收据](archive/research_records/2026-09-06_openneuro_gbm334_frozen_panel/INPUT_PREFLIGHT.json)、[七问](archive/research_records/2026-09-06_openneuro_gbm334_frozen_panel/IDEA.md)、[预注册](archive/research_records/2026-09-06_openneuro_gbm334_frozen_panel/PREREG.md)、[来源](archive/reviews/2026-09-06_openneuro_ds007045_source_audit/RESULT.md)与[对象清单](archive/reviews/2026-09-06_openneuro_ds007045_source_audit/SOURCE_INVENTORY.json)。

## MOTUM-HGG29独立canonical冻结面板 ❌ geometry门前关闭

**一句话：** 用中国三院29名厚层高等级胶质瘤考冻结面板，但三个对齐文件各自重算RAS原点时微小header差被索引翻转放大。

**出发问题：** GCSAM跨外部队列仅Africa确定为正，Flip8已有四处同向；MOTUM官方67人中可在性能前按Participants表冻结29名WHO 3/4级原发胶质瘤、排除38转移瘤。

**具体做法：** 双官方984,434,960-byte快照SHA/CRC exact；29×T1ce/FLAIR-mask/T1ce-mask native审计。原协议要求三文件各自closest-canonical RAS后shape exact、affine≤`1e-5`，再做N4/cache/冻结模型面板。

**改变/固定因素与门：** 唯一新增MOTUM-HGG29；WT为两个语义mask正类并集，其余checkpoint/预处理/view/阈值/统计冻结。任一canonical geometry失败即0模型停止，不能改容差或丢例。

**观察/直觉：** Native最大误差仅`2.384e-7`、29/29离散orientation相同；但首例独立canonical affine误差`1.4488e-5`越门，全29共2例越门。说明是大索引flip放大header小数；来源可对齐直觉仍可能成立，但当前实现失败。

**代价/下一决定：** 双source约1.83GiB、选中87文件34.7MiB；0cache/checkpoint/GPU/forward/performance。旧协议关闭，只允许另立0插值共享orientation桥。证据[结果](archive/research_records/2026-09-06_motum_hgg29_frozen_panel/RESULT.md)、[失败](archive/research_records/2026-09-06_motum_hgg29_frozen_panel/CACHE_PROBE_FAILURE.json)、[来源](archive/reviews/2026-09-06_motum_hgg_source_audit/RESULT.md)。

## MOTUM-HGG29共享离散方向桥 ✅ input-only通过

**一句话：** 三张已对齐胶片共用图像算出的同一个轴翻转/置换，不让各自header小数重算不同原点。

**出发问题：** 旧协议2例canonical affine越`1e-5`，但29/29 native orientation相同且误差≤`2.384e-7`，旧失败没有检验无插值共享变换。

**具体做法：** 新冻结native affine≤`5e-7`与orientation exact；从image算一次到RAS transform/shared affine，同一`apply_orientation`用于image与双mask。两独立Python进程保存29例transform/affine/array/WT SHA并exact比较。

**改变/固定因素与门：** 只改canonical affine生成接口；source arrays、HGG29、mask语义固定，0插值/N4/cache/model。转换arrays还必须与每文件独立canonical array bit-exact。

**观察/直觉：** 29/29全部门过，copy A/B case payload exact；三转换arrays分别等于旧canonical arrays。支持header传播而非真实网格冲突解释。

**代价/下一决定：** 两进程秒级CPU、低于1MiB JSON、0GPU/forward。桥只授权另立性能协议，不能复活旧记录或消除厚层限制。证据[结果](archive/research_records/2026-09-06_motum_hgg29_shared_orientation_bridge/RESULT.md)、[机器结果](archive/research_records/2026-09-06_motum_hgg29_shared_orientation_bridge/RESULT.json)。

## MOTUM-HGG29共享方向冻结面板 ❌ image-support crop门前关闭

**一句话：** 坐标插头修好后继续走canonical图像正值脑框，但一例人工WT伸进了T1ce全零区，不能不看答案地合法扩框。

**出发问题：** 共享方向桥29/29过且MOTUM性能仍0；因此另立全新面板，唯一输入变化为shared orientation，其他与旧面板相同。

**具体做法：** Shared RAS arrays→N4 positive support→image-only bbox→z-score/128³；预注册额外要求0 raw WT crop loss，随后才可cache/smoke/Full identity+Flip8/GCSAM identity。

**改变/固定因素与门：** 相对旧失败只改通过的坐标接口；29例/WT/模型/view/阈值/统计均固定。GT crop丢失任一体素即停，禁止GT扩框、事后padding、排例、改support或no-N4。

**观察/直觉：** MOTUM-HGG-0001 raw WT21,007，positive-image bbox内20,115，丢892/4.246%；全29仅该例。官方HD-BET+FLIRT spline会产生负值，但nonzero bbox仍丢同892，说明坐标桥成功而强度/标签支持不兼容。

**代价/下一决定：** 0cache/checkpoint/GPU/forward/performance；只读29例量化。当前MOTUM路线关闭，等待发布者一致brain mask/修正版图像或未来新训练全FOV模型；不得用GT修源。证据[结果](archive/research_records/2026-09-06_motum_hgg29_shared_orientation_panel/RESULT.md)、[失败](archive/research_records/2026-09-06_motum_hgg29_shared_orientation_panel/CACHE_PROBE_FAILURE.json)、[桥](archive/research_records/2026-09-06_motum_hgg29_shared_orientation_bridge/RESULT.md)。

## CFB-GBM v3外部候选 ❌ target语义

**一句话：** 法国264人/347个放疗靶区很专业，但“强化+坏死核心靶心”不是含周围FLAIR异常的BraTS全肿瘤。

**出发问题：** WAW补了新术前外部，但治疗后/纵向人工WT仍稀缺；CFB 2026 v3有Stupp治疗、多时点T1Gd和大量radiotherapist轮廓，旧负来源未覆盖。

**具体做法与改变/固定：** 只读TCIA v3、发布者segmentation README和v2论文；0下载。只审计来源标签语义，固定本项目WT定义，不把Gross Tumor Volume（GTV，肉眼肿瘤体积）换名。

**预期/停止门与观察：** 需要逐例专家WT才可预注册。官方GTV由放疗计划软件按指南勾画，补全模型也只预测contrast-enhancing+necrotic core并由五位放疗医师验证；没有non-enhancing FLAIR abnormality，故首要语义门失败。208GB Aspera/自动补全/无患者桥无需再进入。

**直觉、代价、下一决定：** 不支持当前WT用途，不否定GTV任务本身。0主体/label/model/GPU；当前关闭，未来若目标改GTV必须新任务/榜。证据[来源审计](archive/reviews/2026-09-06_cfb_gbm_source_feasibility/RESULT.md)。

## Burdenko治疗后GBM来源候选 ❌ target语义

**一句话：** 想用180位患者放疗后纵向MRI补治疗后外部证据，但它的答案纸是放疗靶区，不是BraTS全肿瘤。

**出发问题：** Africa95是术前域且绝对Dice低；Burdenko第一次/第二次随访约在放疗后3周/2–3个月，治疗阶段更匹配。

**具体做法：** 当前只读TCIA页面，核对患者、时点、标注和几何；没有下载131GB主体、manifest/临床表或运行模型。

**改变/固定因素：** 没有改变模型；只是来源可行性门。目标必须是现有WT，不能为了拿到新队列把Gross Tumour Volume（GTV）、Clinical Target Volume（CTV）或Planning Target Volume（PTV）重命名。

**观察结果/停止门：** RTSTRUCT以topometric CT为基准，MRI在原始空间且无发布的MRI→CT变换；随访GTV只在部分患者。更关键的是GTV/CTV/PTV均非包含FLAIR水肿/浸润的BraTS WT。几何容器只能配准，不能修复语义，所以在payload前关闭。

**代价/下一决定：** 0患者/series/pixel/label/model/GPU。只否定当前WT榜用途；未来若目标改为放疗靶区应建独立任务/榜。证据见[来源审计](archive/reviews/2026-09-06_burdenko_source_feasibility/RESULT.md)。

## BraTS-Africa95冻结baseline ✅ 一次性完成并关闭

**一句话：** 让当前Full972 KD的1×/固定8镜像方案和GCSAM 1×去尼日利亚四个中心的95例公开专家胶质瘤上只考一次。

**出发问题：** 内部test270已复用，UCSD official只是同75例来源纠正；UTSW、EGD和UCSF-PDGM分别卡标签语义、DUA和患者重叠。TCIA package948公开146例专家标注，但发布者目录明确其中95例Glioma、51例OtherNeoplasms。

**具体做法：** payload前递归目录审计得到`146×5=730`文件且95例Glioma均唯一T1CE+seg；冻结只取这190路径277,630,584 bytes。未来双份官方传输exact后，all-positive术前WT走canonical RAS/N4 shrink2/image-only crop/z-score/128³。Full KD168同时报告identity和固定Flip8，GCSAM184只identity；阈值0/argmax point，无SAM/后处理/模型ensemble。

**改变/固定因素：** 唯一新增是public post-challenge `95_Glioma` cohort；51个其他肿瘤按发布者类别在像素前排除。模型/checkpoint、病例顺序、预处理、WT、阈值、视图公式、指标/bootstrap全冻结；原挑战60/15/20全部已公开标签，但本项目不训练/调参，合并为外部95，非hidden blind。

**预期机制与门：** 1.5T、多中心、原始3–5mm层厚能检验低资源域迁移；官方1mm重采样不恢复真实细节，须保留grid caveat。190文件双源SHA、95例geometry/finite/integer/允许域/非空、首例cache复现和GPU smoke全过才跑模型；没有性能晋级门，任何结果只归档一次。

**状态/观察结果：** 双源190/190 exact、全95 cache/标签值域/首例复现/模型前收据均过后一次性运行。GCSAM identity Scan=Subject/Point=`0.591619/84`，Full identity=`0.523003/87`，Full Flip8=`0.526438/88`。GCSAM−Full identity Dice=`+0.068617`且paired CI `[+0.044472,+0.092498]`全正、80/95改善，但Point少3；Flip8−identity=`+0.003435`、CI跨0、Point多1。三模型平均预测体积都明显小于真实WT，GCSAM较大、可能解释Dice优势。

**直觉是否支持：** 支持“新域能暴露旧榜看不见的泛化差异”，但不支持单一全面冠军。GCSAM相对Full的平均recall提高`0.0809`且事后CI整体正，precision差近0且CI跨0，说明Dice主要来自补回范围；Point仍较低。Flip8 precision提高`0.0333`但recall近零且不确定。Africa目标在train972经验分位的中位达95.47%，GCSAM优势在最小WT四分位仅`+0.0039`、其余约`+0.082–0.097`，中大型coverage改善而小病灶仍脆弱；不得在该test上设计体积router。旧GCSAM训练配方双复制失败仍成立，Africa结果不能回流复活优化方法。

**代价/下一决定：** 训练0；双源533MiB、cache951MiB，主N4 worker约783秒+finalizer39秒；855 forwards17.15秒、峰值约5.08GiB。GCSAM/Full identity部署1×，Flip8 8×。原始3–5mm层厚经1mm/128³重采样的grid caveat保留。首次cache因probe/worker线程不配对在byte门失败，V2同线程exact后才finalize，未放宽门。实际95例只来自datainfo四中心，raw“六中心”显式更正。Cohort已关闭，不调参/重跑。证据见[结果](archive/research_records/2026-09-06_brats_africa95_frozen_baselines/RESULT.md)、[机器结果](archive/research_records/2026-09-06_brats_africa95_frozen_baselines/RESULT.json)、[raw](archive/research_records/2026-09-06_brats_africa95_frozen_baselines/RAW_EXTERNAL95.json)、[终态验证V3](archive/research_records/2026-09-06_brats_africa95_frozen_baselines/VERIFICATION_RECEIPT_V3.json)与[来源结果](archive/reviews/2026-09-06_brats_africa_source_audit/RESULT.md)。

## UCSD BraTS2024test56冻结面板 ✅ 完成并关闭

**一句话：** TCIA在2026-01-22公开了原BraTS-GLI 2024隐藏test中UCSD贡献的部分；把两个冻结checkpoint带到其中从未见过的41名新患者（56次治疗后扫描）上只考一次。

**出发问题：** 治疗后阶段的外部证据此前只有WAW-post31，两组比较全部positive-but-uncertain；内部fresh容量只剩2人/3 scans。该包59 studies/42 subjects每study都有官方T1post与BraTS_tumor_seg、标签词典与训练完全同口径（WT=labels 1/2/3、排除切除腔4），且官方v3临床表提供patient桥。

**具体做法：** 来源审计→按any-timepoint规则排除唯一union重叠患者`UCSD-PTGBM-0038`（3 studies；也是全表唯一跨ID歧义所在）→冻结56 scans/41 subjects与七问/预注册→双独立官方传输112文件逐文件SHA一致（包内无`.sums`）→raw标签域扫描（56/56整数、⊆{0,1,2,3,4}、无空WT）→canonical RAS/N4/image-only crop/z-score/128³ cache（零WT crop丢失56/56、首例byte exact）→synthetic smoke→模型前收据→唯一504 forwards。

**改变/固定因素：** 唯一新增该cohort；checkpoint（Full972 KD168、GCSAM184）、阈值0、identity+固定Flip8视图、无SAM/后处理、指标与bootstrap seed全部性能前冻结。subject级聚合与患者簇bootstrap因56 scans/41 subjects（多时点患者）而与既有UCSD75评测器同口径。

**观察结果：** Full Flip8/identity/GCSAM Scan=`0.732973/0.730002/0.717689`，Subject=`0.733625/0.730740/0.715545`，Point=`56/56、56/56、52/56`。Flip8−identity=`+0.002971`、scan CI `[-0.002740,+0.008750]`跨0、32/24；**GCSAM−Full=`−0.012312`（Subject`−0.015194`）、scan CI `[-0.025822,-0.000092]`与subject CI `[-0.031507,-0.000268]`首次整体为负**、21/35、Point−4。

**直觉是否支持：** Flip8方向第六次为正但本队列不确定。GCSAM相对Full的负点估计与Africa方向不同；原冻结区间略负，但Scan结论对bootstrap数值精度敏感，不能概括为稳健显著变差。这与覆盖范围/precision取舍的观察相容，但fit864/epoch/seed/optimizer共变，不能单因归因。Full972 KD168在这批56例Point全命中；其命中比例与CPTAC33的全命中并列，不保证未来同谱系患者定位都稳定。

**边界：** public post-challenge、非hidden blind；患者独立已由官方映射证明，但UCSD是BraTS2024七来源站且与UCSD75同机构，**非institution-disjoint**，不能当跨机构泛化证据；独立分榜。训练源Kaggle mirror caveat继承。

**代价/下一决定：** 训练0；双源2×312MB、cache 587MB（/tmp）、N4 325.7秒；正式10.15秒/504 forwards/峰值5.07GiB。Cohort关闭，不据此调参或复活/否决GCSAM配方；点估计方向依cohort可逆。2026-09-07数值精度补记：原Scan CI上界贴0，20组10k bootstrap有17组跨0，百万次Scan CI跨0；Subject4/20跨0且百万次仍略负。故保留原主报告，但不得称两项“稳健确定反向”。补充审计0新forward，不改变当前辅助训练。证据：[数值精度](archive/reviews/2026-09-07_ucsd56_bootstrap_precision/RESULT.md)、[原结果](archive/research_records/2026-09-06_ucsd_brats2024test_frozen_panel/RESULT.md)、[raw](archive/research_records/2026-09-06_ucsd_brats2024test_frozen_panel/RAW_UCSD_BRATS2024TEST56.json)、[验证](archive/research_records/2026-09-06_ucsd_brats2024test_frozen_panel/VERIFICATION_RECEIPT.json)、[来源审计](archive/reviews/2026-09-06_ucsd_brats2024test_source_audit/RESULT.md)。

## BraTS-PEDs257外部候选 ❌ 输入前提（skull未剥离）

**一句话：** TCIA package 1217公开257名treatment-naive儿科高级别胶质瘤的专家迭代修订标签（ET1/NET2/CC3/ED4、CC BY 4.0），本可作极远域外部，但官方发布明确not skull-stripped（仅defaced）。

**出发问题/做法：** 只读完整分页inventory（348例=Training257有seg+Validation91无、1,649文件/35.17GB、无.sums、全部唯一subject）；核一手论文标签字典与collection页预处理描述。0像素、0模型。

**关闭原因：** 本项目冻结的canonical输入链（image-only positive bbox "brain crop"→源非零z-score→128³）只对skull-stripped、背景为零的输入定义；训练数据与既往七个已评cohort全为skull-stripped。含颅骨图像会使crop退化为头颅crop、z-score被颅骨/头皮高信号主导——与RHUH发布态、MOTUM brain-support同类的输入前提失败，不能用大caveat掩盖后评测；自加skull-strip属未经门检的新输入组件，当前无fresh容量为其建立开发证据。另记录儿科官方WT=ET+NET+CC（不含ED）与成人WT的语义分歧，未来须冻结all-positive目标。

**下一决定：** 登记为[阻塞#14](archive/reviews/2026-09-05_30h_chain_audit/EXTERNAL_BLOCKER_REGISTER.md)；等发布者skull-stripped版本或独立过门的skull-strip桥。证据[来源审计](archive/reviews/2026-09-06_brats_peds_source_audit/RESULT.md)。

## UCSF-PDGM专家修订外部候选 ❌ patient-disjoint门

**一句话：** 想从UCSF的501条高质量术前MRI中排掉已进BraTS的人，再考冻结模型；但没有BraTS2024患者交叉表，不能证明剩余者独立。

**出发问题：** UTSW标签3/5无定义、EGD需机构签DUA；UCSF-PDGM公开多区人工修订标签与T1增强，官方还专门增加BraTS21映射用于防泄漏。

**具体做法：** 当前只读TCIA v5页面、501行metadata和glossary。官方说明分割由BraTS获胜模型ensemble先生成、受训放射科医生人工修正并经2位专家批准。按官方6组随访关系合并患者，再统计任何时间点是否已有BraTS21 ID；没有请求142GB package或像素。

**改变/固定因素：** 没有改变模型或训练，只审计来源身份。不会把空BraTS21字段当作与BraTS2024无重叠，也不会用图像相似度猜跨治疗时点的患者身份。

**预期机制与门：** 标签质量和T1GD输入足以建立跨阶段评测，但首要门是患者隔离。501行中298条明确映射BraTS21；合并6组随访后495位真实患者里296位至少一个时点已映射，表面余199位。UCSF又是本地BraTS2024七个来源站之一，官方没有PDGM↔2024 patient crosswalk，所以余199人的patient-disjoint门失败。

**状态/观察结果：** 停在metadata。0 package token、0 inventory、0 NIfTI/标签体素、0模型/GPU。只否定当前把它称为新独立外部队列，不否定数据和标签质量。

**代价/下一决定：** 约60KB公开CSV和网页；无训练/推理/cache。只有发布者或BraTS组织者提供PDGM↔BraTS2024患者表并能在像素前排除重叠，才可重开；当前不下载大包。证据见[来源结果](archive/reviews/2026-09-06_ucsf_pdgm_external_feasibility/RESULT.md)与[机器摘要](archive/reviews/2026-09-06_ucsf_pdgm_external_feasibility/SOURCE_EVIDENCE.json)。

## UPENN术前corrected外部候选 🔒 镜像完整性/parity/患者桥

**一句话：** 官方说有232份术前专家修订答案，但方便下载的逐文件镜像只拿出147份，不能把缺失的85份静默忘掉后开考。

**出发问题：** 旧UPENN审计只关闭60个无标签follow-up；WAW/CPTAC证明术前外部仍有价值。2026更新的data-nih GitHub release提供逐asset SHA，看似可绕过69GB Aspera。

**具体做法与改变/固定：** 只读release API并按文件名统计T1GD/seg配对、bytes和digest；0asset下载/模型。模型与WT不变，不按镜像标签存在性选择患者。

**门与观察：** API实际4424 assets/62.874GB且全SHA；630份`ce-gd_T1w`但仅147份seg/147 pairs共427.7MB。官方availability明确preop611中232份corrected seg，镜像少85；网页又称4426 assets。Provider正文自称Pittsburgh Fiber Data Hub整理，UCSD已证其可量化官方图。论文另有173例进入旧BraTS、无2024患者桥。

**直觉/代价/下一决定：** 当前不支持完整、官方parity且排重的外部队列。0asset/NIfTI/seg/GPU；等待官方NIfTI manifest/member checksum与患者crosswalk，不能因147对下载小而运行。证据[镜像复审](archive/reviews/2026-09-06_upenn_release_mirror_followup/RESULT.md)与[旧随访审计](archive/reviews/2026-09-05_upenn_source_audit/RESULT.md)。

## EGD人工二值WT外部候选 🔒 DUA/账号阻塞

**一句话：** 用Erasmus MC另一家医院374位术前患者的人工0/1全肿瘤mask考冻结模型，避开UTSW多类数字3/5语义不明的问题。

**出发问题：** UTSW362图像/传输均就绪，但116例含发布者未定义label3、5例含label5，不能猜all-positive；当前又没有新的同口径人工外部队列可运行。EGD一手论文声称774位患者中374位有人工whole-tumor分割。

**具体做法：** 当前只做来源门审计。论文确认每人有T1GD及其他三模态，manual `MASK.nii.gz`为`1=肿瘤/0=背景`，基于T2或FLAIR由四位观察者之一勾画；400个AUTO mask明确排除。未来若用户/机构签署Data Usage Agreement（数据使用协议，DUA）并获Health-RI账号，必须先冻结manual+quality acceptable清单和BraTS重叠审计，再另行预注册冻结模型评测。

**改变/固定因素：** 本轮不改变任何模型或数据；只查询公开论文与官方项目metadata。没有把自动标签冒充人工，也没有因规模挑病例。未来的模型/checkpoint、T1GD预处理、阈值、视图和指标尚未授权。

**预期机制与门：** 二值WT无需解释多类编号，374人也比40–75人的队列有更窄区间潜力。但T2/FLAIR定义真值与T1GD-only输入存在跨模态caveat；且必须先证明可用患者/机构隔离，不能只凭Erasmus名称推断零重叠。

**状态/观察结果：** 科学和标签语义初门通过；官方XNAT description要求邮件联系`servicedesk@health-ri.nl`并签DUA后才建账号，匿名subjects API为HTTP403。Codex不能代表用户/机构签协议或发外部承诺，因此0 subject IDs、0影像/标签、0模型/GPU停止。

**代价/下一决定：** 仅数十KB论文XML/项目JSON和网络查询；无训练、推理、cache或部署变化。保留为授权后最高优先级候选；在用户完成外部协调前不反复轮询、不用400例自动mask替代。证据见[来源结果](archive/reviews/2026-09-06_egd_source_feasibility/RESULT.md)和[机器摘要](archive/reviews/2026-09-06_egd_source_feasibility/SOURCE_EVIDENCE.json)。

## CoRe-BT250专家修订外部候选 🔒 UW预审批/DUA

**一句话/状态：** 华盛顿大学250名完整四模态术前病例具备专家纠错三分区mask，但官方要求申请、预审批和机构DUA；当前0清单/payload/model。

**出发问题与直觉：** GCSAM跨外部只有Africa确定，Flip8虽五cohort同向但8×；需要比现有152例更大的新医院人工标签队列。直觉是把冻结模型带到250名新患者只考一次，而不是从旧test继续选参数。

**具体做法：** 本轮只读一手论文/官方仓库。310名患者中250人有术前T1/T1c/T2/FLAIR；三分区由BraTS2023优胜ensemble初始化，有经验标注者逐例纠错，可合成WT。未来获权后先核实际mask数、label、recurrent混入、patient overlap、checksum和许可，再另写七问/预注册。

**改变/固定因素与预期机制：** 当前只新增来源可行性知识，模型/checkpoint/阈值/预处理均未授权。UW不在BraTS2024七站但无patient桥；若清单/标签/隔离过门，较大样本可缩窄checkpoint/Flip8差异区间。

**成功/停止门与观察：** 科学标签流程初步合格；但必须由用户提交form、联系人信息并签DUA。Codex不能代填身份、发邮件或签署，故在0数据处停止；不得用ImageCLEF placeholder labels、embedding或第三方镜像绕过。

**直觉/代价/下一决定：** 尚未检验模型直觉；仅网页/PDF读取、0 GPU/磁盘/推理。保留为用户可解除的第二优先级；证据[来源审计](archive/reviews/2026-09-06_corebt250_source_feasibility/RESULT.md)。

## IMAGO500外部候选 ❌ 自动标签与DUA

**一句话/状态：** 阿姆斯特丹500人大队列影像很有价值，但mask是自动模型输出经医师accept/reject，不是逐体素人工真值；0下载/model关闭当前榜路线。

**出发问题与直觉：** 500人术前四模态、381人早期术后看似能补足大样本和治疗阶段证据；日常类比是先确认“标准答案是谁写的”，不能拿另一台机器的答案给本模型判卷。

**具体做法与改变/固定：** 只读Scientific Data论文，核标签流程、数量和访问条款。249份术前/190份术后segmentation由PICTURE自动生成；一位6年经验神经放射科医师只接受或剔除，未写人工修订；术后只含强化残留。无模型因素改变。

**门与观察：** 人工WT真值门失败；自动错误病例还在发布前被排除，会产生选择偏差。EUCAIM/Health-RI另要求registration、negotiator与DUA。0账号、清单、payload、GPU。

**直觉/代价/下一决定：** 不支持作为当前外部Leaderboard真值；不否定无监督域研究或未来人工修订版。仅网页/PDF成本。等待发布者给出未按自动质量筛选的逐体素人工修订标签，且用户完成DUA；证据[来源审计](archive/reviews/2026-09-06_imago500_source_feasibility/RESULT.md)。

## UCSD75官方float32来源纠正 ✅ 一次性完成并关闭

**一句话：** 把已经评过的同75份UCSD扫描从uint8压缩镜像换回TCIA官方float32原件，让同两个冻结模型按同一协议只重算一次，量出来源量化的影响。

**出发问题：** 旧mirror-uint8上GCSAM184为Scan/Subject/Point=`0.755647/0.755152/71`，Full972 KD168为`0.745554/0.739055/70`；全75例直接官方核验却显示图像`0/75`数组exact、18,156个官方非零体素在镜像中变0，只有标签`75/75` exact。旧结果因此不能代表官方float32输入。

**具体做法：** 固定同75 scans / 53 subjects及顺序，官方T1增强MRI经canonical RAS、N4 shrink2、image-only正值包围盒、源非零population z-score和128³重建；WT仍为原标签`{1,2,3}`并排除4。工程门全过后，Full972 KD168和GCSAM184各跑identity一次，以阈值0输出mask/point，报告绝对指标和相对旧镜像的同模型配对差值。没有训练、teacher、融合、测试时增强、Segment Anything Model或后处理。

**改变/固定因素：** 唯一变化是影像字节由Pittsburgh Fiber Data Hub的uint8整理镜像换成TCIA包1019直接float32文件。病例、标签数组/几何、模型/checkpoint、N4、crop、标准化、128³、插值、阈值、视图、指标/bootstrap全部固定；旧cache/report不可修改。

**预期机制与门：** 量化可能先改变N4拟合和非零crop边界，再改变模型输入，不能用源相关系数代替实测。75映射、150源SHA、几何、标签域及旧标签exact、cache finite/empty、独立重建、synthetic GPU smoke与checkpoint哈希须在模型前全过；否则0 forward停止。没有性能晋级门，结果无论升降均只归档一次，禁止据此调参或重跑。

**状态/观察结果：** 75映射/150源SHA、完整cache和首例独立重建全过；新旧crop/128³ label均75/75 exact，128³ image为0/75 exact。Official-float上GCSAM184 Scan/Subject/Point=`0.755890/0.755312/70`，Full972 KD168=`0.746255/0.740165/70`；GCSAM−Full Dice=`+0.009634/+0.015147`但配对CI跨0。相对mirror，GCSAM=`+0.000242/+0.000160/−1`，Full=`+0.000701/+0.001110/0`，四个来源Dice CI均跨0。

**直觉是否支持：** 部分支持。量化差异确实穿过N4/标准化，但这两个模型的聚合Dice变化很小且不确定；GCSAM Point改变1例，故不能说预测完全相同，也不能推广到别的模型/协议。

**代价/下一决定：** 训练0；cache CPU868.8秒、约751MiB在`/tmp`，两个模型150 forwards、正式进程峰值约4.86GiB，部署仍各1×。同一UCSD75重复来源评测已关闭；不访问internal test270/outer/confirmation、不回流模型选择。证据见[七问](archive/research_records/2026-09-06_ucsd_official_float_source_correction/IDEA.md)、[结果](archive/research_records/2026-09-06_ucsd_official_float_source_correction/RESULT.md)、[机器结果](archive/research_records/2026-09-06_ucsd_official_float_source_correction/RESULT.json)、[raw](archive/research_records/2026-09-06_ucsd_official_float_source_correction/RAW_OFFICIAL_FLOAT.json)与[独立重算](archive/research_records/2026-09-06_ucsd_official_float_source_correction/VERIFICATION_RECEIPT.json)。

## UCSD患者隔离外部确认 ✅ 完成

**一句话：** 不再在已经看过很多次的内部 test270 上找更漂亮的数，而是把两份早已冻结的模型带到 53 位从未进入项目的 UCSD 患者上各评一次。

**出发问题：** 当前 GCSAM 和 Full972 voxel-KD 在复用 test270 上的 Scan Dice 只有 `0.000244` 差距且配对置信区间跨 0；同时近期 LAWA6、Gamma 增强和 Lookahead 都再次暴露单个 seed/inner 正结果难以跨 seed 保持。现有 validation 和 test 均不能再当独立证据。

**具体做法：** 用映射表把UCSD ID映射回BraTS ID，并从184个T1对比增强MRI/标准BraTS分割配对中，按患者排除所有进入过本项目1,350例清单的人。冻结差集是75 scans / 53 subjects。历史实际影像来自Pittsburgh Fiber Data Hub整理的uint8镜像；患者映射后来由TCIA官方临床v3表复核，源标签与官方75/75 exact。图像沿用canonical RAS、N4、image-only brain crop、128³和相同标准化；WT仍为标签`{1,2,3}`并排除切除腔`4`。分别运行冻结Full972 KD168与GCSAM184，阈值0、无测试时增强、无Segment Anything Model、无后处理、各1× forward。

**改变/固定因素：** 唯一改变是评测 cohort；checkpoint、输入通道、预处理、标签定义、模型、阈值和推理全部固定。不训练、不用外部结果挑模型或调参数。

**预期机制：** 这不是提高模型分数的模块，而是提高证据可信度的实验。若内部更稳的 Full972 KD 在新患者上仍保持绝对门并不劣于 GCSAM，说明内部差异至少能跨队列；若失败，则说明当前“leaderboard领先”主要局限在已复用内部数据。

**观察结果：** 官方临床表为178 subjects/243 timepoints，主包为136 subjects/184 studies；109 scans/83 subjects已经暴露，严格差集为75 scans/53 subjects且病例/患者重叠均为0。在历史mirror-uint8输入上，GCSAM的Scan/Subject/Point=`0.755647/0.755152/71/75`，Full972 KD为`0.745554/0.739055/70/75`；GCSAM−KD=`+0.010093/+0.016097/+1`，但paired Dice 95% CI `[-0.002465,+0.027164]`/`[-0.000631,+0.038954]`均跨0。官方float32结果单列在上一卡片，不覆盖本历史行。

**成功/停止门：** Full972 KD绝对 Scan/Subject/阳性Point需分别达到`0.70/0.70/0.90`；相对GCSAM的Scan/Subject点估计不低、配对CI下界均不低于`-0.003`、Point少不超过1。任何asset数、患者差集、SHA-256、geometry、finite或标签域门失败都在模型前停止；结果门失败照实归档，禁止调阈值、病例、N4或checkpoint重测。

**直觉是否支持：** 部分支持。两模型都过绝对迁移门，说明约0.75的表现能带到未见患者；但原先看似更综合稳健的Full972 KD没有保持相对领先。GCSAM三项点估计更高，提示其优化可能更可迁移，但CI跨0，不能称显著。

**代价与下一决定：** 训练0；历史mirror选择下载103.18MB，cache约751MB；两个模型共150次forward，部署仍1×。UCSD是256³ FOV、训练用BraTS副本是182×218×182，存在外部grid/FOV caveat。External confirmation已访问一次，后续official-float只作为同cohort来源纠正；两次均没有新增fixed validation、outer、internal confirmation或reused test270访问。关闭本cohort，禁止调阈值、病例、N4、epoch、`rho`或checkpoint后重测。证据见[历史结果](archive/research_records/2026-09-02_ucsd_ptgbm_external_confirmation/RESULT.md)、[mirror raw](workspace/BraTS/ucsd_ptgbm_external_confirmation_20260902/UCSD_EXTERNAL_CONFIRMATION.json)和[官方float纠正](archive/research_records/2026-09-06_ucsd_official_float_source_correction/RESULT.md)。

## 方法1：Vanilla SAM-Med3D基线 🔒

**一句话：** 先照官方交互方式跑起来，知道“什么都不创新”能到哪里。

**出发点：** 没有可信基线，就无法判断后续提升是真创新还是流程差异。

**具体做法：** T1ce输入SAM-Med3D，使用GT-centered裁剪和最多五次由真实标签导出的交互点击。

**为什么用：** 它是上游模型的标准工作方式，可检查数据、预处理、训练和推理链路是否正确。

**结果：** test最佳五点击Scan Dice `0.594226`，Subject Dice `0.580970`。

**是否支持直觉：** 支持“基线可复现”；不支持把它当全自动最终方案，因为它依赖真实标签提示。

**代价/下一步：** 多次交互forward，且不全自动。它只保留为历史基准。

## 方法2：Image-only brain crop ✅

**一句话：** MRI周围有很多纯黑背景，先把脑区域裁出来，别让模型把计算浪费在空气上。

**出发点：** 整幅图直接缩到128³时，肿瘤和脑组织被压得太小，大量模型token覆盖背景。

**具体做法：** 只根据图像非零区域找脑部包围盒，再把这个包围盒缩放到128³；裁剪位置不看肿瘤标签。

**唯一变化：** 空间预处理；标签、模型和训练协议保持匹配。

**结果：** 旧SAM三点击协议test从`0.571109`提升到`0.625359`，约`+0.054250`。

**是否支持直觉：** 明确支持。先去黑背景是项目最可靠的早期改进之一。

**代价/下一步：** 需要保存crop几何以还原原图，但推理成本不增加。现在已成为主线预处理。

## 方法3：GT crop与residual refiner 🔒

**一句话：** 如果有人提前告诉模型“肿瘤大概在这里”，再用高分辨率小区域做精修。

**出发点：** 全脑定位困难，而局部区域内精细分割更容易。

**具体做法：** 用GT中心或GT导出的点击裁局部块，冻结粗模型，再训练一个residual网络修正粗logit。

**结果：** residual refiner test Scan `0.703798`、Subject `0.686637`，相对粗结果有明显提升。

**是否支持直觉：** 支持“局部精修有效”；但不证明全自动系统更好，因为定位信息来自GT。

**代价/下一步：** 多阶段且oracle。不能进入全自动主榜，只作为机制证据。

## 方法4：Prompt-local refiner ❌

**一句话：** 只在自动point附近开一个小窗口，做便宜的局部修补。

**出发点：** 想保留局部高分辨率收益，又不做昂贵的全图精修。

**具体做法：** 围绕point裁64³小块，训练局部refiner。

**结果：** validation只提高约`+0.000190`，没有达到有效门槛。

**是否支持直觉：** 基本不支持。局部窗口得到的信息不足以稳定修正全局mask。

**代价/下一步：** 额外模型和数据链路换来极小收益，方向停止。

## 方法5：Direct plain locator ✅

**一句话：** 不再让小网络只负责给SAM提示；直接把小网络自己的肿瘤mask当最终答案。

**出发点：** 实验发现好的locator mask接入旧SAM后，Dice反而从约`0.716`降到约`0.599`，说明SAM decoder与当前locator分布不匹配。

**具体做法：** 一个4.70M参数SegResNet直接输入brain-crop T1ce，输出WT logit；阈值固定0，最高logit作为point。DiceCE中CE权重降到0.1，避免全背景坍缩。

**结果：** epoch99 test Scan `0.725820`、Subject `0.709413`、Point `0.900000`。

**是否支持直觉：** 强支持。任务对齐的小locator比“locator→旧SAM”链路更好、更快、更自动。

**代价/下一步：** 单模型1×推理，成为所有后续研究的主线。

## 方法6：CAPG中心头 ❌

**一句话：** 除了分割mask，再加一个专门预测肿瘤中心的输出头。

**出发点：** 希望point更稳定，从而帮助后续SAM。

**具体做法：** SegResNet同时输出分割logit和中心热图，中心头学习GT内的中心锚点。

**结果：** locator Scan有小幅变化，但中心头Point hit仅`0.638889`，plain分割argmax为`0.935185`；接SAM后也更差。

**是否支持直觉：** 不支持。人为定义的“中心”不一定是网络最容易、最可靠的肿瘤点，多病灶/不规则病灶尤其如此。

**代价/下一步：** 额外head和loss没有价值，停止；后续point研究应直接约束最终segmentation logit。

## 方法7：灰度/intensity residual ❌

**一句话：** 让模型根据T1ce灰度特征再修正一次分割结果。

**出发点：** 肿瘤区域可能有与周围组织不同的强度模式，灰度或许能纠正边界。

**具体做法：** 在plain输出上增加matched residual/control和intensity residual分支。

**结果：** intensity相对control Scan `-0.002638`、Subject `-0.003331`，Point hit下降约`-0.370370`。

**是否支持直觉：** 不支持当前实现。强度线索可能不稳定，且容易制造GT外高峰。

**代价/下一步：** 不进入selector/test，不继续调灰度权重。

## 方法8：Late-cosine长时程到epoch168 ✅

**一句话：** 模型跑100轮时还没真正收稳；后面慢慢降低学习率，让它在好区域里精修。

**出发点：** 100轮可能只是训练预算截断，而不是模型能力上限。

**具体做法：** epoch1–100保持`2e-4`，epoch101–200用cosine逐渐降到`1e-5`；fresh-inner选择epoch168，再用全756例固定重训。

**结果：** test Scan从epoch99的`0.725820`升到`0.741204`，Subject从`0.709413`升到`0.723179`。

**是否支持直觉：** 强支持。低学习率尾程是当前单模型主进步来源。

**代价/下一步：** 训练更久但推理不变；epoch168成为历史强基座。

## 方法9：True-160³高分辨率 ❌

**一句话：** 把128³网格换成160³，希望看清更多小细节。

**出发点：** 128³缩放会丢失边界和小病灶信息。

**具体做法：** train和test都从原始NIfTI重新生成true-160³ cache，不放大128 cache；全756例训练200轮。

**结果：** test Scan `0.737060`、Subject `0.719729`、Point `0.874074`，均低于128³ epoch168；显存约翻倍，体素计算约1.95×。

**是否支持直觉：** 不支持“全图一律加分辨率”。更多体素没有转化为更好泛化，反而让最高峰更不稳定。

**代价/下一步：** 大磁盘、大显存、低收益；cache已在保留归档后删除，不继续该方向。

## 方法10：EMA权重平滑 🟡

**一句话：** 不只用最后一刻的模型，而是保存最近一段训练权重的平滑平均。

**出发点：** 随机采样和增强会让模型参数来回抖动；平均后可能更稳。

**具体做法：** epoch100后复制模型，从epoch101起每步执行`EMA=0.999×旧EMA+0.001×当前模型`。

**结果：** EMA能明显改善同seed的较差raw轨迹，但没有超过历史最佳Dice。单独test为Scan `0.740696`、Subject `0.724905`、Point `0.948148`。

**是否支持直觉：** 部分支持。EMA确实让point更稳、提供不同错误模式，但单独不是最强模型。

**代价/下一步：** 训练多一份影子权重，推理仍1×。最有价值的是作为ensemble的互补成员。

## 方法11：固定logit ensemble ✅

**一句话：** 让历史强模型做主判断，再听25%的EMA意见。

**出发点：** 历史模型Dice强，EMA point强，二者错误不完全一样。

**具体做法：** 对同一输入分别得到两份logit，计算`0.75×history+0.25×EMA168`，阈值仍为0，point取混合logit最高点。权重先在fixed validation选择，再冻结进test。

**结果：** 当前总体test最高：Scan `0.745054`、Subject `0.727345`、Point `0.955556`；相对历史模型三项都提高。

**是否支持直觉：** 强支持。互补错误能通过输出平均稳定地结合。

**代价/下一步：** 推理要跑两个locator，约2×计算。后续目标是蒸馏回单模型。

## 方法12：Cross-seed parameter soup ❌

**一句话：** 既然输出能平均，试试直接平均两个模型参数，推理只跑一次。

**出发点：** Model Soup在同一权重盆地内可能接近ensemble而没有2×推理。

**具体做法：** 固定`0.75×历史参数+0.25×EMA参数`，不搜索比例。

**结果：** validation Scan从`0.731601`崩到`0.431343`，108/108病例全部下降。

**是否支持直觉：** 不支持。两个独立seed的内部神经元排列和权重盆地不同，参数不能直接相加；输出可以平均不代表参数可以平均。

**代价/下一步：** 零训练快速排除；不扫更多alpha、不进test。

## 方法13：2025/2026 voxel-logit KD ✅

**一句话：** 让一个学生模型在训练时看ensemble的答案，学会后只部署学生。

**出发点：** ensemble有效但推理2×；希望把它的互补知识压进一个模型。

**具体做法：** 学生从历史epoch168开始再训练32轮。固定teacher为`0.75 history+0.25 EMA`；除了真实标签DiceCE，再加`0.3×`每体素KL蒸馏，温度`T=2`。同时跑完全匹配的无KD continuation控制。

**结果：** 当前最高单模型test：Scan `0.742976`、Subject `0.725179`、Point `0.955556`。Point完全追平ensemble，Dice吸收部分收益；配对Dice CI跨0。

**是否支持直觉：** 支持，但收益来源要诚实拆分：继续低LR训练贡献较大，KD相对matched control再提供约`+0.00078/+0.00076` validation增益。

**代价/下一步：** KD训练约control的1.79×，推理恢复1×。这是当前推荐单模型。

## 方法14：2026 ReCo-KD区域归一蒸馏 ❌

**一句话：** 教师不要被海量背景淹没；把肿瘤、边界、背景分开教。

**出发点：** 普通每体素KD中，背景体素远多于肿瘤，模型可能主要学背景。

**具体做法：** 在voxel-KD上增加前景、两体素边界带、非边界背景分别归一的SmoothL1 logit loss，边界权重2，总权重0.1。

**结果：** 同epoch192相对base KD小幅提高Scan `+0.000441`、Subject `+0.000688`并多命中1例，但最佳绝对Dice低于base KD epoch200。

**是否支持直觉：** 有很弱的局部迹象，但不足以成为更好最终模型。

**代价/下一步：** 训练与KD同量级，推理1×；不进test、不调region权重/边界宽度。

## 方法15：Safe adapter、boundary与Domain-SAGE 🟡/🔒

**一句话：** 冻结强base，只训练一个很小、初始为零的修正层，避免新模块一上来破坏原结果。

**出发点：** 大改模型风险高，小adapter可以“先不改、再慢慢学”。Domain-SAGE还尝试按潜在域选择修正。

**具体做法：** zero-initialized adapter叠加在冻结base上；再做boundary和latent-domain gate消融。

**结果：** plain safe adapter在oracle validation相对base约`+0.003`且多seed复现；boundary额外收益未过门槛；Domain-SAGE分类很准但主Dice不升。

**是否支持直觉：** 支持“小的安全残差层”，不支持boundary/gate/domain conditioning是收益来源。

**代价/下一步：** 这是GT-centered oracle支线，不直接进入全自动主榜。

## 方法16：ELPGR显式error gate ❌

**一句话：** 预测“哪里会错”，再让修正器只处理这些地方。

**出发点：** 不希望refiner改坏已经正确的区域。

**具体做法：** error predictor产生gate，控制refiner残差。

**结果：** 两seed平均比matched control Scan低约`0.001780`。

**是否支持直觉：** 当前实现不支持。预测错误本身也会错，gate会漏掉应修区域或错误放大。

**代价/下一步：** 多模块无收益，停止。

## 方法17：Robust shift、TTA、MCTS与推理剪枝 🟡/🔒

**一句话：** 对oracle residual-refiner做多seed、多翻转ensemble，再研究少跑几次能否保持分数。

**出发点：** ensemble通常稳但贵，希望找到精度/速度平衡。

**具体做法：** robust-shift训练、3 seed×8 reflection TTA；MCTS/random/greedy搜索成员与视图；固定1/4/8/12/24 forward比较。

**结果：** locked 24-forward validation约`0.7004`；8-forward略高且部署约2.78×加速，1-forward refiner阶段约13.66×加速。MCTS没有胜过random/greedy，算法优越性为负。

**是否支持直觉：** 支持“存在大量推理冗余”；不支持“MCTS搜索本身更聪明”。

**代价/下一步：** 属GT-centered refiner支线，只能报告工程效率，不与全自动主榜混排。

## 尚未正式运行的2025+候选 🧪

### Full972 保统计Gamma强度增强 ❌fresh final反向

**一句话：** 训练时随机调MRI的非线性明暗曲线，同时把均值和标准差调回原样，让模型少依赖某一种T1ce对比度。

**状态：** 2026-09-02两阶段完成。Stage A fresh inner六门全过，但fresh Full972 final fixed validation相对Control与archived均反向，最终11门过2失9；不访问test。

**出发点：** 项目最大可靠收益来自数据呈现；当前增强只有翻转/平移/线性亮度/噪声，没有非线性曲线。LAWA与Lookahead说明optimizer/权重平均不能解决seed差异。旧intensity residual是在输出后修mask，不否定训练期输入不变性。

**具体做法：** Control保持现有增强；Candidate在完全相同原增强后，以独立RNG对30%病例做Gamma`0.7–1.5`、非反相、保均值/总体标准差。标签、模型、teacher、loss、seed、sampler、LR、epoch168、阈值与部署固定。

**成功/停止门：** 新864/108同seed配对，须胜Control Scan/Subject、Point不降、两CI安全，并过原增强exact、Gamma概率/范围/统计/label/RNG门；过门才fresh Full972 final。Final还须胜matched Control与archived champion、Point>=103、四CI安全；失败不调Gamma设置或进test。

**代价：** 双阶段最多约7–8h墙钟，额外逐体素pow但无额外forward，不建cache，推理仍1×。证据：[七问](archive/research_records/2026-09-02_full972_gamma_aug_stability/IDEA.md)、[预注册](archive/research_records/2026-09-02_full972_gamma_aug_stability/PREREG.md)、[nnU-Net论文](https://www.nature.com/articles/s41592-020-01008-z)、[官方Gamma接口](https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/transforms/color_transforms.py)。

**Stage A结果：** Candidate/Control=`0.708356/0.710096/101` vs `0.702205/0.704526/100`，即`+0.006151/+0.005570/+1 Point`；Scan/Subject CI下界`-0.001553/-0.001596`均过安全门。48/108 scans、25/52 subjects改善，说明正均值由较大个案收益驱动，必须fresh final复现。证据：[决定](archive/research_records/2026-09-02_full972_gamma_aug_stability/STAGE_A_DECISION.json)、[raw inner](workspace/BraTS/sammed3d_t1ce_wt/full972_gamma_aug_stability_20260902/STAGE_A_INNER.json)。

**Final结果/解释：** Gamma/Control=`0.729001/0.720571/101` vs `0.732850/0.722882/101`，即`-0.003848/-0.002312/0`；相对archived seed32为`-0.011011/-0.010695/-2`且两项CI整体低于0。Stage A正向没有跨seed复现，原直觉不成立；禁止调Gamma概率/范围/顺序/retain-stats。两阶段formal约13.56 GPU-hours，peak20.30GiB，workspace289M，推理1×。证据：[结果](archive/research_records/2026-09-02_full972_gamma_aug_stability/RESULT.md)、[机器结果](archive/research_records/2026-09-02_full972_gamma_aug_stability/RESULT.json)、[raw validation](workspace/BraTS/sammed3d_t1ce_wt/full972_gamma_aug_stability_20260902/STAGE_B_VALIDATION.json)。

### Full972 Lookahead-AdamW慢权重稳定优化 ❌Stage A失败

**一句话：** AdamW每走5个快步，就让一套慢权重向当前位置走一半并从慢位置重新出发，把稳定中心从第一轮贯穿到epoch168。

**状态：** 2026-09-02已完成配对fit864 formal与唯一一次fresh train-only inner。Candidate Scan/Subject低于AdamW，四项Dice门失败，`full972_final_authorized=false`；未运行Full972 Stage B，fixed validation/outer/confirmation/test均未访问。

**出发点：** LAWA6在三个fresh seed都相对自身raw提高Scan/Subject`+0.00046～+0.00277`且Point不降，但final仍低于archived强seed`0.00709/0.00490`。说明轨迹平滑有一致局部价值，却只平均最后六轮太晚；seed差异更早形成。

**具体做法：** Control为原AdamW；Candidate保持完全相同AdamW作为fast optimizer，固定每5步把slow向fast插值50%并同步fast，最终部署slow。模型、teacher、loss、数据、seed、采样、增强、LR、epoch168、阈值与1×推理固定。

**成功/停止门：** 先在新864/108 train-only split同seed配对，Candidate须胜Control Scan/Subject、Point不降、两CI安全并过初值/sampler/首batch/20328步/4065 slow update合同；全过才fresh Full972配对final。Final还须胜matched Control和archived Full972 seed32、Point>=103、四CI安全；否则不进test且不调`k/alpha`。

**为什么既有负结果未否定：** GCSAM测试的是每步锐度扰动且多一次forward/backward；LAWA6只在最后六轮事后平均；Lookahead从随机初始化开始每5步反馈slow中心。Cross-seed soup混不同盆地，本方法始终在同一run内。

**观察结果：** Candidate/Control=`0.738678/0.716695/104` vs `0.742575/0.721451/104`，即`-0.003897/-0.004757/0`。Scan CI`[-0.009837,+0.001641]`，Subject CI`[-0.011494,+0.001552]`；51/108 scans改善、56变差，27/52 subjects改善、24变差。工程配对九门全过，因此是方法泛化失败而非接线失败。

**是否支持直觉：** 不支持。Candidate最终fit loss/train Dice优于Control，但fresh inner Dice更差；默认slow反馈过度约束了当前AdamW表示学习。它与LAWA6的末期小幅正收益一起说明“最后轻微平滑”和“全程反复拉回”不能等同。

**实际代价/下一决定：** 两臂各约3.38–3.39h，formal约6.77 GPU-hours，peak20.30/20.32GiB，workspace217M，推理仍1×。分支关闭，不调`k/alpha/fast-slow/epoch/LR/seed/threshold`，不进test；后续离开优化器/权重平均轴。证据：[结果](archive/research_records/2026-09-02_full972_lookahead5_stability/RESULT.md)、[机器结果](archive/research_records/2026-09-02_full972_lookahead5_stability/RESULT.json)、[raw inner](workspace/BraTS/sammed3d_t1ce_wt/full972_lookahead5_stability_20260902/STAGE_A_INNER.json)、[checkpoint](workspace/BraTS/sammed3d_t1ce_wt/full972_lookahead5_stability_20260902/formal_lookahead5_candidate/raw_lookahead5_candidate_epoch168.pth)。

### Full972 同轨迹最近六轮权重平均 ⚠️局部机制正向/final未晋级

**一句话：** 同一个模型训练到最后时会小幅抖动，把epoch163–168六次参数等权平均，尝试得到更靠近同一权重盆地中心的单模型。

**状态：** 2026-09-02已完成三个fresh seed、Stage A train-only inner和Stage B fixed validation。三个seed的LAWA6都超过各自raw Scan/Subject且Point不降，但final低于archived Full972 seed32并少1 Point，最终11门通过6、失败5；分支关闭，不访问test。

**出发点：** Reused test最高Scan的GCSAM只比Full972 KD高`0.000244`，Subject近乎持平且Point少10；fresh seed31/32/33 inner Scan为`0.746412/0.736109/0.732792`，显示末期轨迹/seed稳定性仍是瓶颈。容量、160³、困难loss和多seed融合均已关闭。

**具体做法：** Full972式random-init voxel-KD训练、模型、数据、teacher、loss、AdamW、晚余弦schedule和固定epoch168全部不变；只在同一run捕获epoch163–168六份student参数，以CPU float64逐tensor等权平均，最终仍部署一个4.70M student。Stage A在全新train-only inner用两个fresh seed验证，二者全过才以第三个fresh seed做Full972 final/fixed validation。

**唯一变化/固定项：** 唯一比较是同run raw epoch168 vs LAWA6平均参数；固定数据、split、每run seed/初始state、968 draws/121步、增强、teacher、DiceCE+voxel-KL、LR、阈值0、无TTA/SAM/postprocess和1×推理。

**成功/停止门：** 两个fresh seed的LAWA都必须同时超过各自raw Scan/Subject、Point不降且四个CI下界`>=-0.003`；final还须超过matched raw和archived Full972 seed32 `0.740013/0.731266`，Point至少103并过四CI门。任一失败不改`k=6`、窗口、epoch、seed或阈值。

**为什么既有负结果未否定：** Cross-seed parameter soup混的是不同盆地；这里仅平均同一seed连续六轮。旧EMA是Full756旧轨迹、step级0.999长记忆；这里是Full972完整轨迹、最近六个epoch等权。GCSAM改变每步更新但未平均同轨迹后期解。

**实际代价：** Stage A两臂各约3.37–3.39h、Stage B约3.39h，formal总约10.15 GPU-hours，peak20.30GiB/卡；workspace181M、不建新cache；部署仍4.70M参数、1×forward。

**Stage A观察结果：** seed61 LAWA/raw=`0.747123/0.733388/106` vs `0.744804/0.731433/103`，delta=`+0.002319/+0.001955/+3`；seed62 LAWA/raw=`0.739130/0.728890/107` vs `0.737862/0.726117/106`，delta=`+0.001268/+0.002772/+1`。seed61 Scan CI整体为正；其余CI跨0但下界均高于冻结`-0.003`安全门。两个seed均三指标同向，支持继续fresh Full972 final，但尚不能称超过fixed-validation或test榜。证据：[Stage A决定](archive/research_records/2026-09-02_full972_lawa6_stability/STAGE_A_DECISION.json)、[raw inner](workspace/BraTS/sammed3d_t1ce_wt/full972_lawa6_stability_20260902/STAGE_A_INNER.json)。

**Final观察结果：** fresh seed63 LAWA/raw=`0.732919/0.726370/102` vs `0.732456/0.725793/102`，仍为`+0.000464/+0.000576/0`，CI下界`-0.001387/-0.001494`均过安全门；但相对archived Full972 seed32为`-0.007093/-0.004897/-1 Point`，两项CI下界约`-0.0154`且Point未达103。说明盆地内平均机制可复现，但收益远小于跨seed差异，不能解决主稳定性瓶颈。

**是否支持直觉/下一决定：** 局部直觉支持，完整超榜目标失败。停止当前`k=6, epoch163–168`实现，不搜索窗口、权重、epoch、seed或阈值，不访问test。证据：[完整结果](archive/research_records/2026-09-02_full972_lawa6_stability/RESULT.md)、[机器结果](archive/research_records/2026-09-02_full972_lawa6_stability/RESULT.json)、[raw validation](workspace/BraTS/sammed3d_t1ce_wt/full972_lawa6_stability_20260902/STAGE_B_VALIDATION.json)、[checkpoint](workspace/BraTS/sammed3d_t1ce_wt/full972_lawa6_stability_20260902/formal_stageb_full972_seed20261063/lawa6_stageb_full972_seed20261063_epoch168.pth)。

### 训练标签引导的每病例最佳老师蒸馏 ❌validation全面失败

**一句话：** 训练时有真实标签，就让student对每个病例听三位Full972老师里分得最准的那位；部署只保留student。

**状态：** 2026-09-01已完成paired Full972 formal和fixed validation108。Routing持续覆盖三老师，但Candidate Scan/Subject/Point全部低于matched Control与archived seed32；九项promotion门全失败，不访问test。

**出发点：** Fresh固定共识相对弱seed提高`+0.0108–0.0162` Dice，却只比强seed31高`+0.000530/+0.000063`且CI失败；82/108 scans改善但26变差。固定平均不知道每例该听谁。

**具体做法：** Full972 paired students都用fresh seed20261040与固定epoch168。Control让anchor seed32真实forward3次；Candidate让Full972 seed32/40/41各forward1次，按每病例GT soft-Dice loss argmin选择整幅teacher raw logit。Student loss仍为DiceCE+固定`T=2, weight=0.3`voxel-KL；teacher全部frozen/no-grad。

**唯一变化/固定项：** 唯一arm差异是teacher target来自固定anchor还是per-case GT最佳teacher。Teacher forward数3、student initial state/数据/sampler/增强/model/loss/LR/epoch/阈值/推理全部固定。

**成功/停止门：** Candidate必须同时超过matched control与archived Full972 seed32 validation的Scan/Subject，Point≥103，四CI下界≥`-0.003`；失败不调routing/teacher/seed/KD/epoch/threshold。Test需再次授权。

**观察结果：** Candidate=`0.732910/0.725998/100`，Control=`0.736013/0.728973/103`，archived seed32=`0.740013/0.731266/103`。Candidate相对Control=`-0.003104/-0.002975/-3 Point`，相对archived=`-0.007103/-0.005268/-3 Point`；四个CI安全门均失败。全程routing counts seed32/40/41=`47215/63869/51540`，机制确实工作。

**是否支持直觉：** 不支持。训练GT oracle能为每例找到局部更优teacher，但跨病例/epoch切换target使同容量student学到更不一致的函数，validation泛化反而下降。不能把teacher oracle质量等同于student可学习知识。

**实际代价/下一步：** 两臂各约4.13h，峰值20.31/20.37GiB，无新cache，推理1×。分支关闭，不进test、不改routing。证据：[结果](archive/research_records/2026-09-01_gt_best_teacher_routing_distillation/RESULT.md)、[机器结果](archive/research_records/2026-09-01_gt_best_teacher_routing_distillation/RESULT.json)、[raw validation](workspace/BraTS/sammed3d_t1ce_wt/gt_best_teacher_routing_20260901/GT_BEST_ROUTING_VALIDATION.json)。

### Fresh三seed共识蒸馏 ❌Stage A强seed CI门失败

**一句话：** 重新训练三位从未见过fresh inner的独立seed老师，让一个新student学习三人的固定平均答案，最终部署仍只跑student一次。

**状态：** 2026-09-01已完成三位fit864老师训练和唯一一次fresh inner评测。固定共识点估计超过三单模且Point等于最佳，但相对最强seed31的Scan/Subject CI门失败；`student_stage_authorized=false`，未训练student或Full972 final，未访问fixed validation/outer/confirmation/test。

**出发点：** 旧Full972三seed固定等权logit ensemble相对seed32 validation提高`+0.005829/+0.004418/+1 Point`，88/108 scans、41/50 subjects改善；但Subject CI下界`-0.005660`阻止了旧Stage C。GCSAM又显示单轨迹Scan小幅领先会换来Point少10，说明下一步应利用seed共识而不是继续押单轨迹optimizer。

**为什么现在选择：** 2025 ensemble distillation支持把多模型预测分布压入单student；2026 MTSS-KDNet与官方代码直接支持多老师单学生脑肿瘤分割；项目自身已有两模型logit ensemble→1×voxel-KD成功先例。Parameter soup失败不否定输出蒸馏，feature/region KD负结果也未测试固定多seed output-only consensus。

**具体做法：** 新建Full972的fresh subject-clean `864 fit/108 inner`。先在fit864训练3个固定epoch168 seed老师并独立复现`1/3`raw-logit ensemble；过门后训练计算匹配students：Control把同一anchor teacher真实前传3次，Candidate把3个独立teacher各前传1次。两臂都用真实标签DiceCE+固定`T=2, weight=0.3`逐体素KL。Inner过门后才用已有Full972 seed32/40/41跑fresh paired final并访问fixed validation。

**唯一变化/固定项：** 唯一arm差异是三次teacher forward来自同一anchor还是三个独立seed。Teacher forward数、student初始state/数据/batch/增强/model/loss/LR/168轮/阈值0/无TTA-SAM-postprocess全部固定。

**成功/停止门：** Fresh ensemble必须同时超过三个单模Scan/Subject且Point不低于最佳单模、全部CI安全；三老师student必须超过matched单老师student三项且CI安全；Full972 final必须超过matched control和archived seed32 Scan/Subject、Point≥103及四CI安全门。任一阶段失败立即停，不换seed/老师/权重/epoch。

**观察结果：** Fresh inner共识=`0.746942/0.735130/106`；seed31/32/33分别=`0.746412/0.735067/106`、`0.736109/0.722151/104`、`0.732792/0.718894/103`。共识相对弱seed32/33的Scan/Subject增益约`+0.0108/+0.0130`和`+0.0142/+0.0162`且CI整体为正；但相对强seed31只有`+0.000530/+0.000063/0`，CI下界`-0.009103/-0.016105`失败。

**是否支持直觉：** 部分支持。Seed错误确实互补，平均能显著救弱轨迹；但固定共识没有稳定超过强轨迹，旧validation上的明显ensemble增益未在fresh split复现。当前核心瓶颈仍是“偶然强seed不可稳定复制”，不是单纯缺一个student压缩步骤。

**实际代价/下一步：** 三老师各约3.38–3.40h，双卡Stage A墙钟约6.79h；峰值各20.29GiB，不建cache。分支停止，不训练student、不剔除seed或调权重。证据：[IDEA](archive/research_records/2026-09-01_fresh_multiseed_consensus_distillation/IDEA.md)、[预注册](archive/research_records/2026-09-01_fresh_multiseed_consensus_distillation/PREREG.md)、[结果](archive/research_records/2026-09-01_fresh_multiseed_consensus_distillation/RESULT.md)、[机器结果](archive/research_records/2026-09-01_fresh_multiseed_consensus_distillation/RESULT.json)、[raw inner](workspace/BraTS/sammed3d_t1ce_wt/fresh_multiseed_consensus_20260901/TEACHER_CONSENSUS_INNER.json)。

### Full972 GCSAM跨seed稳定优化 ❌inner稳定性门失败

**一句话：** 不再改模型或loss；每步先沿“去掉每个卷积核平均噪声后的梯度”轻推参数，再要求被推后的模型也把同一批病例分好，尽量找到更宽、更不挑seed的解。

**状态：** 2026-08-31已完成全链路审视、七问/预注册、实现合同、双卡smoke和200轮paired train-only inner。两项CI门失败，未跑fresh-seed final。用户随后明确授权一次failed-gate exploratory reused test270；测试完成并已冻结，不再调参。

**出发点：** Full972 seed32/40/41 validation单模Scan为`0.740013/0.733408/0.736000`；width24在train-only inner一度领先约`+0.010634/+0.012193`，fresh-seed final却变成`-0.002872/-0.003525`。这表明当前明确瓶颈是优化轨迹/seed稳定性，而不是继续增加4.70M学生容量。

**为什么现在选择：** 2025 GCSAM论文用gradient centralization（梯度中心化）抑制SAM ascent中的噪声，并报告在通用/医学影像分类上比Adam和SAM更稳定；官方代码公开。它只改变优化更新，正好避开已判负的分辨率、宽度、额外监督和困难样本loss轴。论文没有3D分割实验，因此这里只是paper-form 3D适配，不冒充复现。

**具体做法：** Full972、history/EMA teacher、DiceCE+voxel-KD和4.70M width16全部不变。Control继续AdamW；candidate固定`rho=0.05`、non-adaptive，第一次反传后对所有多维参数梯度按非输出维减均值，用中心化梯度本身的L2 norm归一化扰动，再在同一batch/teacher logits上第二次反传、恢复参数并由同一AdamW更新。部署仍只有普通student、1×forward。

**唯一变化/固定项：** 唯一arm差异为AdamW vs paper-form GCSAM-wrapped AdamW。固定Full972、fresh train-only inner、两阶段独立seed、同初始state/batch序列、模型、teacher、loss、LR、epoch规则、阈值0及无TTA/SAM/postprocess。

**成功/停止门：** 先在fresh train-only inner由candidate选择post100 epoch，并要求同epoch超过control Scan/Subject、Point最多少1且CI安全；再fresh-seed full972重训。Final必须同时超过matched control和archived Full972 seed32 `0.740013/0.731266`，Point至少103/108及四CI安全门。失败不调rho/epoch/阈值，不访问test。

**观察结果：** GCSAM按Scan选epoch184=`0.753587/0.736140/102`，same-epoch AdamW=`0.752780/0.733582/102`；平均差`+0.000807/+0.002558/0`。但Scan/Subject paired 95% CI为`[-0.005464,+0.006866]`/`[-0.005327,+0.009841]`，两项下界均低于`-0.003`。58/108 scans改善、49变差；26/50 subjects改善、23变差，稳定性主张不成立。

**探索性test：** GCSAM fit864 epoch184=`0.752003/0.737266/252/270`，archived Full972 KD168=`0.751760/0.737327/262/270`。GCSAM差`+0.000244/-0.000061/-10 points`；Scan/Subject paired CI `[-0.005655,+0.005740]`/`[-0.009153,+0.008094]`均跨0。按Scan排序是新的reused test点估计第1，但不是Subject/Point最佳，不是Full972 final或显著提升。

**是否支持直觉：** 只支持“优化轨迹能略改平均Dice”，不支持“结果更稳”。改善/变差接近对半且CI宽，因此不能把正点估计解释为解决跨seed稳定性。只否定当前`rho=0.05`、non-adaptive 3D适配，不证明所有sharpness方法不可能；但不得在同一inner搜索rho或实现。

**来源/实现边界：** 论文公式要求用中心化梯度范数归一化；官方示例代码的执行顺序先取原梯度norm。预注册在任何访问前选择论文公式，并要求扰动范数/restore合成合同通过，禁止结果后比较实现。

**实际代价/下一步：** Control/GCSAM `4.166h/6.474h`，GCSAM为`1.554×`；峰值`20.292/20.358GiB`。推理仍1×。当前分支停止，不补跑final、不调rho/epoch/seed/Point门；test不能回流调参。完整证据：[IDEA](archive/research_records/2026-08-31_gcsam_full972_stability/IDEA.md)、[预注册](archive/research_records/2026-08-31_gcsam_full972_stability/PREREG.md)、[结果](archive/research_records/2026-08-31_gcsam_full972_stability/RESULT.md)、[机器结果](archive/research_records/2026-08-31_gcsam_full972_stability/RESULT.json)、[raw test](workspace/BraTS/sammed3d_t1ce_wt/gcsam_full972_stability_20260831/GCSAM_INNER_E184_TEST270.json)。来源：[论文](https://arxiv.org/abs/2501.11584)、[官方代码](https://github.com/mhassann22/GCSAM)。

### LoMix-3D 三尺度训练监督 ❌

**一句话：** 一个模型内部把粗尺度“看全局”和细尺度“看边界”的输出学会混合，推理只保留最终头。

**状态：** 完成预注册、实现、12项合成测试、batch8 GPU smoke、32轮formal continuation和fixed validation108。promotion失败，test未访问；当前实现停止。

**出发点：** fixed validation 上 voxel-KD epoch200 为 Scan/Subject `0.734384/0.724434`，相对 matched continuation 只多 `0.000779/0.000762`；160³全图加密失败，但同一128³网络已有的多尺度信息仍可能有用。LoMix不需要两个最终模型。

**为什么现在选择：** NeurIPS 2025论文给出二维主实验和三维MedNeXt附录初步结果；三维LoMix Dice `87.19`，高于末层监督`86.22`和普通深监督`86.63`。这只支持“值得在本管线验证”，不保证BraTS有效。160³负结果改变输入网格，不等于已经否定训练期多尺度监督。

**具体做法：** 同一T1ce进入原SegResNet；32³、64³、128³三个decoder stage各自产生raw logit并上采样对齐。所有二元/三元subset做加法、乘法、注册的`1×1×1`拼接卷积和逐体素attention，共19张训练监督图；每张用现有DiceCE并乘`softplus(alpha)`。没有teacher。部署只保存原最终raw logit，阈值、point和1×forward不变。

**唯一变化/固定项：** 唯一增加训练期LoMix监督；history168初始化、756例manifest/cache、seed20260846、batch8、采样/增强、AdamW、late-cosine epoch169–200、validation108、阈值0、无SAM/TTA/后处理全部固定。matched control复用完全同协议的监督continuation。

**成功/停止门：** 四个冻结epoch按validation Scan选择；必须同时超过同epoch control和当前KD200的Scan/Subject，Point至多少1/108且配对CI守门，才允许冻结一次reused test270候选。NaN/Inf、注册参数无梯度、batch8超过72GiB、部署logit不等价或平均Softplus权重低于初值25%立即停止。未过门不调operator、alpha、loss、epoch、阈值或后处理。

**论文与实现边界：** 官方仓库只公开二维代码；三维结果是论文附录初步证据。公开代码的concat卷积在forward内临时创建，且attention共享与论文表述不同；本项目将预先注册每个subset的3D模块。论文未归一化Softplus目标会直接推动所有alpha下降，因此必须记录权重动态，不能把结果无条件解释成“搜索到了最好分支”。

**观察结果：** 四个validation checkpoint中epoch200最高：Scan `0.732479468`、Subject `0.721607103`、Point `0.935185`。相对同epoch matched control为Scan `-0.001126193`、Subject `-0.002064797`；相对voxel-KD200为`-0.001904840/-0.002827040`。相对KD的Scan 95% CI `[-0.004242,+0.000247]`，Subject `[-0.005073,-0.000651]`。Point持平，但43/108 scans和20/50 subjects改善，65 scans和30 subjects变差。

**是否支持直觉：** 不支持当前实现。所有新增模块都有finite nonzero gradient，checkpoint/deploy logit差为0，所以不是代码没工作；但最终head overlap下降。平均Softplus从初始`0.693147`降到epoch200的`0.654070`，19项权重仍很接近，未显示强的“自动选择好组合”。这不证明所有LoMix或规范化/从头训练版本不可能，只证明本次忠实raw-logit、未归一化、late-continuation协议失败。

**实际代价：** formal训练`1045.64s`（17.43min），约为matched control的`1.58×`；峰值allocated约`23.85GiB`，LoMix workspace约144M。推理仍为一个4.70M参数locator、1×forward，不建新cache。

**下一决定：** 当前LoMix实现停止，不调operator、alpha规则、loss、epoch、threshold或后处理，不进test。下一候选转向2026 DCD detail distillation。证据：[预注册/最终填写](archive/research_records/2026-08-19_lomix3d/IDEA.md)、[结果](archive/research_records/2026-08-19_lomix3d/RESULT.md)、[机器结果](archive/research_records/2026-08-19_lomix3d/RESULT.json)、[论文](https://papers.nips.cc/paper_files/paper/2025/file/5f0e54e34ba4c30fbc6f229986dbffb4-Paper-Conference.pdf)、[官方实现](https://github.com/SLDGroup/LoMix)。

### DCD方向细节蒸馏 ❌

**一句话：** 不让学生照抄教师全部feature，只学习三维图像中与边界/方向有关的高频细节。

**状态：** 完成一手源码审计、预注册、11项合成测试、真实batch8 GPU smoke、32轮formal训练和fixed validation108。promotion失败，本方法的新test未访问；当前适配停止。

**出发点：** fixed validation上voxel-KD相对matched control只增加Scan/Subject `0.000779/0.000762`，而LoMix decoder supervision为负。普通logit KD可能保住粗定位，却没有专门保护encoder下采样时易丢失的边界和小结构。

**为什么现在选择：** MICCAI 2026 DCD直接针对3D MRI encoder detail。论文BraTS压缩student mDice从`63.60%`到`68.51%`，NETC从`41.26%`到`54.36%`。但论文是大teacher→压缩nnU-Net、四模态多类别、1000轮；本项目只是同容量T1ce/WT的32轮增量适配，不能冒充复现。

**具体做法：** 保留现有`DiceCE + 0.3×voxel-KD`。另用validation已冻结且与history168同初始化路径的KD200作detail teacher；student/KD200四级encoder feature用`db4`做3层3D小波分解，只保留band index1–6，排除LLL和HHH，再逆变换回空间feature做MSE。每stage内3 level取mean、四stage求sum。same-channel遵官方实现不加projection。训练后删除所有teacher和wavelet，只部署原student。

**唯一变化/固定项：** 唯一增加KD200→student DCD loss。student init、history/EMA logit teacher、T2、voxel-KD weight0.3、train/cache、seed20260846、batch8、sampler/增强、AdamW/LR、global169–200、validation108、threshold0和无SAM/TTA/postprocessing全部固定；不新增student EMA。

**权重冻结：** 论文写`mu=0.05`，官方BraTS YAML使用max`0.01` cosine-healing。选择可执行BraTS YAML，并按训练进度映射到32轮：zero-based index start5/ramp26；global176/184/192/200的mu固定为`0.000145/0.003227/0.007840/0.010000`，不看validation修改。

**实现审计边界：** 官方`dec`→`conv3d`、`rec`→`conv_transpose3d`在PyTorch cross-correlation语义下不能全频重建。执行前已冻结使用同一db4 decomposition bank做analysis与adjoint synthesis，并设全8-band max error `<1e-5`门；这是论文detail projection语义修正，不会在validation后比较方向。

**成功/停止门：** selected DCD必须同时超过同epoch base KD和KD200的Scan/Subject，Point至多少1/108且paired CI守门，才能进test。DWT/IDWT shape/filter不一致、非有限loss、任一student stage无detail gradient、teacher有gradient、batch8超过72GiB或部署不等价立即停止。

**观察结果：** 四个validation checkpoint随epoch提高，epoch200最高：Scan `0.733603344`、Subject `0.723798160`、Point `0.935185`。相对同epoch base KD200为`-0.000780964/-0.000635983`；Scan 95% CI `[-0.001888,+0.000058]`，Subject `[-0.001581,+0.000123]`。49/108 scans与20/50 subjects改善，59 scans与30 subjects变差。安全CI和Point门通过，但四个必须超过base/KD200的点估计门失败。

**是否支持直觉：** 不支持当前适配。raw DCD从`3.9181`降到`0.5489`，约下降86%，四个stage都有DCD专属梯度，所以不是loss没工作；但detail matching没有转化为更高最终Dice。selected结果几乎回到matched supervised control，未保住base voxel-KD的小增益。这不否定论文的大teacher→压缩student场景。

**实际代价：** formal训练`5954.21s`（99.24min），为base voxel-KD的`5.04×`；峰值allocated约`30.59GiB`，workspace约72M。部署仍4.70M参数、1×forward，不建新cache。

**下一决定：** 当前DCD适配停止，不调teacher、wavelet、band、stage、mu、schedule、epoch、threshold或postprocessing，不进test。后续KAS也已validation判负，当前下一候选转向hard-binned mL1-ACE。证据：[预注册/最终填写](archive/research_records/2026-08-19_dcd_detail_distillation/IDEA.md)、[结果](archive/research_records/2026-08-19_dcd_detail_distillation/RESULT.md)、[机器结果](archive/research_records/2026-08-19_dcd_detail_distillation/RESULT.json)、[论文](https://arxiv.org/html/2605.26382v1)、[官方代码](https://github.com/ClinicaAlpha/DCD-3D-MedSeg)。

### KAS-on-voxel-KD teacher-free语义同化 ❌

**一句话：** 先向学生灌入teacher知识，再减弱teacher约束，让学生重新发挥自己原本的优势。

**状态：** 完成论文/源码审计、预注册、12项合成测试、2×A100 paired smoke、control/KAS双臂formal训练和fixed validation108。promotion失败，本方法的新test未访问；当前适配停止。

**出发点：** voxel-KD相对control只提高约`0.000779/0.000762`；DCD继续模仿teacher detail后反而下降。可能不是teacher知识不足，而是外部约束覆盖student自身能力。

**具体做法：** 把KD200作为已有infusion起点。control和KAS都teacher-free继续32轮；KAS额外把最深encoder feature经registered `1×1×1 Conv3d`投影到最终logit空间，用raw MSE、sigmoid Dice consistency和per-volume contrastive self-optimization让student自我整理。总目标固定为`DiceCE+0.01*(MSE+Dice+Contrastive)`，不detach。部署删除projector。

**唯一变化/固定项：** 唯一增加KAS训练期projector/loss；起点、数据、seed20260860、batch8、sampler/增强、fresh AdamW、LR`1e-5`、32轮、validation、threshold和推理全部固定。两臂分别在两张A100-40GB独立进程并行，不用DDP。

**成功/停止门：** selected KAS必须同时超过同epoch control和原KD200的Scan/Subject，Point/paired CI与batch-hash/reload门全过才进test。projector未注册、任一分量/梯度非有限、双臂batch hash不同、任一卡超过32GiB或部署不等价立即停止。

**代码边界：** 官方release的KIS/KAS块大多被注释、`loss`未定义，projector在batch内临时创建；binary注释权重也与accepted paper冲突。本项目预先选择paper的`gamma=0.01`等权KAS公式、binary sigmoid negative形式和per-volume cosine，只能称论文语义的3D适配。

**观察结果：** KAS按Scan选择epoch232：Scan `0.732915355`、Subject `0.723199040`、Point `0.916667`。相对同epoch control为`-0.001218828/-0.001418980/0`；相对KD200为`-0.001468952/-0.001235103/-0.018519`。相对KD200的Scan CI `[-0.002870,-0.000151]`整体低于0，Subject CI下界`-0.003062`也未过安全门。

**是否支持直觉：** 不支持当前适配。raw KAS从`48.06`降到`1.54`，说明自一致目标被学会；但paper等权gamma初期远大于主loss，最终Dice和Point都变差。teacher-free control epoch224 Scan/Subject `0.734413/0.724670`略高于KD200点估计，但Point少1例，且control没有预注册test promotion，不能借此访问test。

**实际代价：** control/KAS并行formal约19.3min；峰值allocated约`20.29/20.41GiB`；workspace约217M。部署删除projector，仍4.70M参数、1×forward。

**下一决定：** 当前KAS适配停止，不调gamma/内部权重/projector/detach/contrast/LR/epoch。下一候选为hard-binned mL1-ACE等校准/难例loss。证据：[预注册/最终填写](archive/research_records/2026-08-20_iad_kas_assimilation/IDEA.md)、[结果](archive/research_records/2026-08-20_iad_kas_assimilation/RESULT.md)、[机器结果](archive/research_records/2026-08-20_iad_kas_assimilation/RESULT.json)、[论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Hong_From_Infusion_to_Assimilation_Distillation_for_Medical_Image_Segmentation_CVPR_2026_paper.pdf)。

### Hard mL1-ACE前景校准 ❌

**一句话：** 把前景概率分20档，惩罚每档“平均自信”和“真实肿瘤频率”的差距。

**状态：** 13项合成测试、A100 smoke、32轮formal训练、完整control配对和一次fixed validation108均完成；校准门通过但Dice promotion失败，停止且不访问新test。

**出发点：** 多次方法的Dice变化很小，但Point被少数错误高峰影响；现有loss不直接约束置信概率是否可靠。

**具体做法：** 从KD200按与既有teacher-free control完全相同的seed/LR/32轮训练，只增加foreground sigmoid probability的20-bin hard mL1-ACE。hard assignment detached，bin内mean probability保持梯度；空bin忽略。总loss固定为历史DiceCE加`1.0×ACE`。

**唯一变化/固定项：** 唯一增加ACE；起点、数据、seed20260860、batch8、sampler/增强、fresh AdamW、LR1e-5、grid208/216/224/232、validation、threshold和推理全部固定。直接复用KAS实验的teacher-free control。

**预期机制：** 减少错误但过度自信的体素峰，同时尽量保留Dice；如果只改善ACE却降低Dice，就说明校准可靠性与当前WT重叠目标发生冲突，不能仅凭Point或ACE放行。

**成功/停止门：** selected hACE必须超过same-epoch control和KD200的Scan/Subject、Point至少100/108、Dice CI安全，同时macro-ACE至少降低0.005且CI上界<0、official-micro ACE更低，才进test。ACE无logit梯度、NaN、配对hash不同、峰值超过24GiB或deploy不等价立即停止。

**来源边界：** 官方是多通道softmax、Dice:CE:ACE=1:1:1；项目为单通道sigmoid并保持历史Dice:CE=1:0.1，所以实际1:0.1:1是单因素适配，可能过强。论文BraTS hard版本DSC保持0.905，macro-ACE从0.146降到0.132。

**观察结果：** hACE按Scan选择epoch224，Scan/Subject/Point=`0.723870/0.713278/0.953704`（103/108）；same-epoch control为`0.734413/0.724670/0.925926`，KD200为`0.734384/0.724434/0.935185`。相对control的Scan/Subject delta为`-0.010544/-0.011392`，95% CI分别`[-0.014594,-0.006693]`和`[-0.017180,-0.006043]`。但macro ACE从`0.230865`降到`0.125927`，delta CI `[-0.112170,-0.097648]`；official-micro ACE从`0.197534`降到`0.071770`。

**是否支持直觉：** 部分支持校准机制，但不支持整体方法。train ACE下降约65%，validation校准和Point都改善，证明loss有效；同时Dice显著下降。最可信推断是项目`1:0.1:1`中weight1校准压力与主loss同量级，却不直接奖励边界/区域重叠，因此用Dice换了更可靠的概率。不能由此事后降权重重跑。

**实际代价：** 单armformal `1354.37s`（22.57min），峰值allocated/reserved约`20.29/21.53GiB`，新增workspace约76M、不建cache；推理仍4.70M参数、1×forward。

**下一决定：** 当前适配停止，不搜索bins/right/weight/起点/LR/epoch/threshold/postprocessing，不进test。下一候选为单因素PM Dice难体素调制，需重新核验一手论文/实现并预注册。证据：[预注册/最终填写](archive/research_records/2026-08-20_hard_ml1ace/IDEA.md)、[结果](archive/research_records/2026-08-20_hard_ml1ace/RESULT.md)、[机器结果](archive/research_records/2026-08-20_hard_ml1ace/RESULT.json)、[论文](https://arxiv.org/html/2506.03942v4)、[官方代码](https://github.com/cai4cai/Average-Calibration-Losses)。

### 3D VM Dice困难体素调制 ❌

**一句话：** 让Dice少看已经分对的体素，把梯度集中到当前分错的边界和碎小病灶。

**状态：** 19项合成测试、双卡paired smoke、gamma0/gamma1双臂32轮formal和一次fixed validation108已完成；总体及bottom-tail门失败，停止且不访问test。

**出发点：** KD200相对control的收益被少数困难case抵消；边界复杂度最高四分位平均delta为`-0.001202`，仅9/27改善。KD Dice与边界/体积代理Spearman为`-0.675`，说明高边界、碎片化tail是稳定瓶颈，单纯小病灶采样证据不足。

**具体做法：** 显式构造foreground/background概率，用论文公式`m=|y-stopgrad(p)|^gamma`调制平方分母Dice，并保留`0.1×CE`。新跑gamma0 paper-Dice control与gamma1 VM Dice；两张A100一臂一卡，无teacher。

**唯一变化/固定项：** 两臂唯一差异是`gamma 0→1`。background、平方分母、epsilon1e-6、KD200起点、seed20260870、batch8、sampler/增强、fresh AdamW、LR1e-5、global201–232、grid208/216/224/232、validation、threshold和推理全部相同。历史standard Dice不能替代gamma0 control。

**预期机制：** detached modulation让高误差体素权重更高，但正常概率路径继续反传；若overall及冻结hard strata同时改善，支持难体素机制。若只改Point/边界小组却不提高overall Scan/Subject，不进test。

**成功/停止门：** selected gamma1必须同时超过same-epoch gamma0和KD200的Scan/Subject、Point至少101/108、两组CI安全、boundary-top27与KD-bottom27均有正delta。公式/gradient/pairing/reload/deploy、24GiB和45min门任一失败立即停；不得搜索gamma或阈值。

**来源边界：** 论文无官方代码、无BraTS、无真正3D网络实验；ACDC/MSSEG均按2D切片训练。项目是3D独立实现和KD200后32轮continuation。论文默认gamma1；MSSEG的fg2/bg1来自数据集敏感性搜索，本项目禁止照搬。

**观察结果：** gamma1按Scan选择epoch224，Scan/Subject/Point=`0.732178/0.721938/0.953704`；同轮gamma0为`0.732818/0.722534/0.953704`，KD200为`0.734384/0.724434/0.935185`。相对gamma0为`-0.000641/-0.000596/0`；相对KD200为`-0.002207/-0.002496/+0.018519`，两项CI安全门失败。

**是否支持直觉：** 部分支持局部机制、整体不支持。boundary-top27 mean delta为`+0.000903`，但CI跨0且仅12/27改善；KD200-bottom27为`-0.001495`，只有10/27改善。VM loss下降66.4%，说明目标学动，但默认gamma1没有把局部边界收益转成overall提升。

**实际代价：** 两臂formal `1121.52/1123.44s`，双卡墙钟18.72min；每臂peak allocated约`20.48GiB`，workspace约145M；部署仍4.70M参数、1×forward。

**下一决定：** 当前gamma1适配停止，不搜索gamma/epsilon/CE/LR/epoch/threshold/postprocessing，不进test。后续Batch-CVaR50病例尾部聚合也已validation判负。证据：[七问/最终填写](archive/research_records/2026-08-20_vm_dice/IDEA.md)、[结果](archive/research_records/2026-08-20_vm_dice/RESULT.md)、[机器结果](archive/research_records/2026-08-20_vm_dice/RESULT.json)、[错误审计](archive/research_records/2026-08-20_vm_dice/ERROR_AUDIT.json)、[论文](https://arxiv.org/pdf/2506.15744)。

### Batch-CVaR50病例尾部聚合 ❌

**一句话：** 每批8例不再全部平均，只让loss最高的4例决定这一步更新。

**状态：** 15项合成测试、GPU legacy parity、双卡paired smoke、uniform/CVaR双臂32轮formal和一次fixed validation108均完成；overall与两个tail门失败，停止且不访问test。

**出发点：** KD200最差10例占全部负收益约63%；VM Dice虽改善boundary-top27，却让更广bottom27下降，说明下一步应从体素级转为整例tail风险。

**具体做法：** 两臂对每个scan独立计算原Dice+`0.1×CE`。新control对8例求均值；candidate以stable detached排序选择最高4例，对原attached loss求均值。无teacher、无新增参数。

**唯一变化/固定项：** 唯一变化为`mean8→top4 mean`。KD200起点、seed20260860、batch8 replacement sampler、增强、fresh AdamW、LR1e-5、global201–232、grid208/216/224/232、validation和推理相同；重复case按draw计，不去重。

**成功/停止门：** selected candidate必须超过同轮uniform和KD200的Scan/Subject、Point≥101/108、两组CI安全、KD-bottom27改善且boundary-top27不恶化。per-case mean与legacy DiceCE不等价、stable top4/梯度/pairing/reload/24GiB门任一失败立即停止。

**来源边界：** 2025论文是对全训练图像排序的AT10%或N→5% curriculum，并配合平滑和pixel过滤；ACDC还是2D slice，固定hard AT10%反而降分。当前top4/8只是保守的batch经验CVaR50适配，不能称论文复现或population/subject CVaR。

**观察结果：** candidate按Scan选择epoch208，Scan/Subject/Point=`0.731801/0.721922/0.925926`；same-epoch uniform=`0.733194/0.723334/0.944444`，KD200=`0.734384/0.724434/0.935185`。相对uniform的Scan/Subject=`-0.001393/-0.001412`且95% CI整体低于0；相对KD200=`-0.002583/-0.002512`。

**是否支持直觉：** 不支持。KD200-bottom27 mean delta=`-0.002275`，boundary-top27=`-0.001551`，病例tail没有改善。top4确实把首batchloss和grad norm从约`0.187/0.436`提高到`0.218/0.759`，但瞬时batch hardness与真实validation hard cases不稳定对应。

**实际代价：** uniform/CVaR formal约`1107.86/1106.53s`，双卡墙钟18.46min；每卡peak allocated约20.29GiB，workspace约145M；部署仍4.70M参数、1×forward。

**下一决定：** 当前top4/8适配停止，不搜索比例、curriculum、smoothing、loss、LR、epoch或阈值，不进test。近期loss/teacher约束均未超过KD200，下一步离开loss轴，重新核验局部原生分辨率或架构信息瓶颈。证据：[七问/最终填写](archive/research_records/2026-08-20_batch_cvar50/IDEA.md)、[结果](archive/research_records/2026-08-20_batch_cvar50/RESULT.md)、[机器结果](archive/research_records/2026-08-20_batch_cvar50/RESULT.json)、[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12428111/)。

### FairDRO（暂缓）

**一句话：** 用临床subgroup路由专家，并在每组内部提高困难患者权重。

**状态：** 不进入正式实验。公开论文没有代码、KL半径/求解器/dMoE关键配置；当前BraTS也没有部署可用的临床/机构group。强行做只能成为自创global DRO，不能称FairDRO。

### NMSW / 固定局部原生patch 🚫可行性未过

**一句话：** 不把全图都升高分辨率，只选择少量full-grid局部patch精修。

**状态：** 不进入正式训练。完整NMSW需要global+local双模型、Gumbel Top-K和learnable fusion，单肿瘤/当前128³会退化成单patch，且部署不是1×。自定义自动argmax中心96³原生patch的validation GT覆盖只有74/108达到90%，最差15.9%，固定单patch不可接受。

**出发点：** 160³全图成本高且失败；但一手论文明确没测试tumor/single-organ，因为sampling会退化。扩大、多patch或改ROI会同时改变几何、融合和forward数，无法形成当前要求的单因素。

**下一决定：** 暂缓局部patch，不恢复旧prompt-local。可行性证据见[`archive/research_records/2026-08-21_dmrfb_bottleneck_adapter/FEASIBILITY_SCREEN.json`](archive/research_records/2026-08-21_dmrfb_bottleneck_adapter/FEASIBILITY_SCREEN.json)、[NMSW论文](https://papers.miccai.org/miccai-2025/paper/0413_paper.pdf)、[官方代码](https://github.com/Youngseok0001/open_nmsw)。

### Single-bottleneck DMRFB dilation adapter ⚠️机制正向/未晋级

**一句话：** 不加像素，在最深特征处并行看1/3/5尺度上下文。

**状态：** 11项合成测试、CUDA identity/two-step-gradient smoke、same-parameter双臂formal和fixed validation108均完成；超过control并改善两个hard strata，但KD Scan/Point门失败，不访问test。

**出发点：** 多条loss/teacher目标学动却不升overall；局部patch覆盖不足。DMRFB在3D单因素消融约`+0.48 Dice`，作用于感受野而非loss。

**具体做法：** KD200最深`128@16³`后加57,088参数DMRFB residual adapter。两臂均有相同branch/norm/GELU/zero projection；control有效RF `[1,3,3]`，candidate仅把第三支dilation1改2，成为`[1,3,5]`。

**唯一变化/固定项：** 唯一差异是第三支dilation/padding metadata。两臂参数shape/初值、KD200起点、DiceCE、数据、seed20260900、base LR1e-5、adapter LR1e-3、32轮和推理均相同。

**成功/停止门：** candidate须超过同轮control和KD200的Scan/Subject、Point≥101、四项CI安全、boundary-top27改善且bottom27不恶化。epoch0 logits必须与plain完全相同；step1 projection、step2全部adapter参数必须有finite nonzero gradient；24GiB/45min/reload门失败立即停。

**来源边界：** 这是单bottleneck source-inspired DMRFB，不是完整EfficientMedNeXt。来源BrainTumour是四模态/多区域、60k steps；项目是T1ce-only WT、KD200后32轮双LR continuation。

**观察结果：** candidate按Scan选择epoch216，Scan/Subject/Point=`0.734242/0.724742/0.925926`；same-epoch control=`0.733997/0.724596/0.935185`，KD200=`0.734384/0.724434/0.935185`。相对control为`+0.000245/+0.000146/-0.009259`；相对KD为`-0.000142/+0.000308/-0.009259`。

**是否支持直觉：** 部分支持。Boundary-top27=`+0.000107`，KD-bottom27=`+0.001184`且19/27改善；candidate同时超过同参数control的Scan/Subject，说明dilation2有小的机制收益。但CI跨0、Scan没超过KD200且Point少1例，不能进test。

**实际代价：** 两臂formal约18.55/18.58min，双卡墙钟18.58min；每臂peak allocated约20.40GiB、4,757,985参数、workspace约146M；部署1×forward。

**下一决定：** 当前single-block协议停止，不搜索dilation/block/LR/epoch。后续同盆地weight interpolation已取得Dice正结果，但Point门失败，仍未test。证据：[七问/最终填写](archive/research_records/2026-08-21_dmrfb_bottleneck_adapter/IDEA.md)、[结果](archive/research_records/2026-08-21_dmrfb_bottleneck_adapter/RESULT.md)、[机器结果](archive/research_records/2026-08-21_dmrfb_bottleneck_adapter/RESULT.json)、[论文](https://papers.miccai.org/miccai-2025/paper/4895_paper.pdf)、[官方代码](https://github.com/SLDGroup/EfficientMedNeXt)。

### KD200–DMRFB216同盆地weight interpolation ⚠️Dice正向/Point未过

**一句话：** 在单模型内按比例混合KD200稳定性与DMRFB的Subject/tail收益。

**状态：** 10项合成测试、3个checkpoint生成/reload和fixed validation108已完成；selected alpha0.5同时超过两个端点的Scan/Subject，但Point少1例，停止且不访问test。

**出发点：** DMRFB216超过same-param control并改善两个hard strata，但Scan比KD200低`0.000142`且Point少1例。两端主干relative-L2仅1.02%，且DMRFB从KD200 exact identity起点训练，属于同盆地轨迹。

**具体做法：** 把KD200装入dilation2+zero-adapter架构作为alpha0，DMRFB216为alpha1；对全部同构state tensor插值，固定只评alpha`0.25/0.5/0.75`，无训练。

**唯一变化/固定项：** 唯一变化是alpha。架构、4,757,985参数、dilation、threshold、无SAM/TTA/postprocess和1×推理固定。

**成功/停止门：** selected interior必须同时超过KD200和DMRFB216的Scan/Subject、Point≥101、两端CI安全；否则停止，不增加alpha或改logit ensemble。

**观察结果：** alpha0.5 Scan/Subject/Point=`0.734683/0.724934/0.925926`；相对KD200=`+0.000299/+0.000500/-0.009259`，相对DMRFB216=`+0.000441/+0.000192/0`。三个interior的Point均为100/108。

**是否支持直觉：** Dice折中直觉支持。Selected boundary-top27相对KD=`+0.001053`、bottom27=`+0.002011`；但Point稳定性没有恢复，完整目标不支持。

**实际代价：** 无训练，3个checkpoint约55M；生成CPU strict-forward数分钟，GPU validation约19秒；部署4,757,985参数、1×forward。

**下一决定：** 当前alpha协议停止，不换alpha、不借KD单独出Point、不进test。若未来明确改成Dice优先且允许Point少1例，alpha0.5可作为development-selected候选，但属于新目标。证据：[七问/最终填写](archive/research_records/2026-08-21_dmrfb_weight_interpolation/IDEA.md)、[结果](archive/research_records/2026-08-21_dmrfb_weight_interpolation/RESULT.md)、[机器结果](archive/research_records/2026-08-21_dmrfb_weight_interpolation/RESULT.json)。

### 固定计算量的972例voxel-KD数据扩容 ⚠️数据机制正向/未晋级

**一句话：** 不再改loss，而是在相同3008个优化步内让模型从972例而不是756例真实病例中学习。

**状态：** 11项合成测试、双卡GPU smoke、cache-matched双臂formal和fixed validation108已完成。相对新control明显改善，但KD200 Scan和Point门失败，不访问test。

**出发点：** 当前KD200只用756 scans/346 subjects；近期loss、teacher、tail weighting和adapter多只带来约`+0.0005`以内变化。旧selector/confirmation合计216例已结束历史角色，与fixed validation108零重叠。

**具体做法：** 从historical168开始，保留`0.75 history + 0.25 EMA`的voxel-KD teacher、DiceCE和32轮late-cosine；新control/candidate均从同一旧972 cache读取，分别暴露756/972例，每轮都抽756次、94个optimizer step。

**唯一变化/固定项：** 唯一变化是可抽样病例池`756→972`。Student/teacher、loss、seed20260846、batch8、draw数、optimizer/LR、epoch grid、增强、阈值和1×推理全部固定。

**成功/停止门：** selected必须同时超过新same-epoch cache-matched 756 control和历史KD200的Scan/Subject，Point≥101/108，四个配对CI安全门和缓存/loss/reload/24GiB门全过才允许一次test。

**观察结果：** candidate按Scan选择epoch184，Scan/Subject/Point=`0.734049/0.724694/0.916667`（99/108）；same-epoch control=`0.730752/0.720408/0.925926`，历史KD200=`0.734384/0.724434/0.935185`。相对control为`+0.003297/+0.004287/-0.009259`，Scan/Subject 95% CI分别`[+0.001027,+0.005549]`/`[+0.001546,+0.007158]`；相对KD200为`-0.000335/+0.000260/-0.018519`。

**是否支持直觉：** 部分且比较明确地支持。扩大病例池对新完全匹配control有广泛、CI为正的改善，说明数据多样性确实是真实瓶颈之一。但它没有超过使用原生标签体积weighted sampling的历史KD200 Scan，Point还少2例，所以完整目标未成功。

**实际代价：** 双臂各约32.3min，两卡并行墙钟约32.3min；每卡peak allocated约20.29GiB；复用cache、workspace约181M；部署仍4.70M参数、1×forward。

**下一决定：** 当前固定756 draws协议停止，不事后改成972 draws或换采样权重。下一数据问题已收窄为“972例+原生体积感知sampling能否同时保住Point”，必须重新预注册。证据：[七问](archive/research_records/2026-08-21_voxel_kd_data972/IDEA.md)、[结果](archive/research_records/2026-08-21_voxel_kd_data972/RESULT.md)、[机器结果](archive/research_records/2026-08-21_voxel_kd_data972/RESULT.json)。

### 972例原生体积感知voxel-KD采样 ⚠️Point恢复/未晋级

**一句话：** 保留972例多样性，但用原始1mm标签中的真实WT体积决定每例被抽到的频率。

**状态：** train-only工件、756例体积/权重exact、11项测试、V2 GPU smoke、双卡formal和fixed validation108均完成。Point门恢复且超过matched control，但KD200 Scan仍低`0.000324`，不访闯test。

**出发点：** 上一数据扩容相对matched control有Scan/Subject `+0.003297/+0.004287`的正CI，但仍比历史KD200 Scan低`0.000335`且Point少2例。已知唯一协议差异是原生标签体积vs 128³缓存标签体积权重。

**具体做法：** 仅从972个train NIfTI label统计canonical RAS、image-only brain crop内WT `{1,2,3}`原生体素数；新跑native756/native972双臂，都用`sqrt(median/volume)` clip`[0.5,4]`，其余复用voxel-KD。

**唯一变化/固定项：** 两臂唯一差异是可抽样病例池756/972；两臂都用同一原生体积权重定义。固定student/teacher/loss/seed/batch/756 draws/32轮/LR/阈值/1×推理。

**成功/停止门：** selected需同时超过same-epoch native756 control和KD200的Scan/Subject，Point≥101/108、四CI安全门和24GiB/reload/access门全过才能test。任一756原生体积/权重不exact立即停止。

**观察结果：** candidate按Scan选epoch200，Scan/Subject/Point=`0.734060/0.724703/0.935185`（101/108）；native756 control=`0.733371/0.724036/0.944444`，KD200=`0.734384/0.724434/0.935185`。相对control为`+0.000689/+0.000666/-0.009259`；相对KD200为`-0.000324/+0.000269/0`。四项CI安全门全过，只有KD200 Scan点估计门失败。

**是否支持直觉：** 部分支持。原生体积权重把上次的99/108 Point恢复到101/108，并保持相对matched control的Dice正增益；但没有让overall Scan超过KD200，说明固定756 draws下每病例更少更新可能成为新瓶颈。

**实际代价：** 双臂各约32.3min，每卡peak allocated约20.29GiB；原生体积工件433KB、不新建大cache；workspace约217M；部署仍4.70M参数、1×forward。

**下一决定：** 当前native fixed-compute协议停止，不改选epoch184、不调weight clip、不进test。若继续，只剩“972 draws/epoch”的数据+计算预算扩容问题，必须新预注册。证据：[结果](archive/research_records/2026-08-21_voxel_kd_data972_native_sampling/RESULT.md)、[机器结果](archive/research_records/2026-08-21_voxel_kd_data972_native_sampling/RESULT.json)。

### 覆盖优先的小病灶回放蒸馏 ❌完整晋级失败/覆盖机制部分支持

**一句话：** 每轮先轮流覆盖680个不同病例，再用约一成训练名额按原生肿瘤体积回放72个小病灶，既扩大见闻又保留小目标练习。

**状态：** 2026-08-24已完成文献检索、预注册、18项sampler测试、5项validation合同测试、V4 smoke、双卡formal和fixed validation108。Selected candidate200超过same-epoch control Dice，但KD200 Scan、Point和KD Scan CI门失败；不访问test。

**出发点：** native972 candidate在fixed validation108达到`0.734060/0.724703/101/108`，相对same-epoch control Dice为正并恢复Point，但每轮带放回只消费752 draws，使972病例池的覆盖不足，Scan仍比KD200低`0.000324`。MICCAI 2025 CSAL-3D支持把分布覆盖与少量信息病例选择结合；项目Batch-CVaR50负结果又要求不能把大比例预算压在难例上。

**具体做法：** 两臂都保留972病例、原生体积权重、history/EMA教师、voxel-KD和94步×32轮。Control按原生体积全带放回；candidate首轮随机覆盖680例，后续每轮纳入上轮遗漏的292例并从上轮已见病例保留388例，再给72个按原生体积加权的small-lesion replay draws，另4个replay固定置尾并由drop-last丢弃。

**唯一变化/固定项：** 唯一arm差异是sampler policy。数据、cache、模型、student/teacher、loss、seed20260884、batch8、752有效draw、94步、增强、AdamW/LR、epoch grid、threshold0、无TTA/SAM/postprocess和1×推理全部固定。

**成功/停止门：** selected candidate必须同时超过same-epoch replacement control和archived KD200的Scan/Subject，Point≥101/108，四项paired 95% CI下界≥`-0.003`；两轮覆盖、determinism、24GiB、45min和reload门全过才允许一次reused test270。失败后不调680/72比例。

**观察结果：** selected candidate200 Scan/Subject/Point=`0.734158/0.724950/98/108`；same-epoch replacement control=`0.732678/0.723060/99/108`；archived KD200=`0.734384/0.724434/101/108`。Candidate-control Scan/Subject=`+0.001481/+0.001890`，95% CI分别`[-0.000209,+0.003246]`/`[-0.000525,+0.004556]`。Candidate-KD200=`-0.000226/+0.000516/-3 points`，Scan CI下界`-0.003098`略低于`-0.003`门。

**是否支持直觉：** 部分支持。相同计算量下，强制覆盖相对带放回control改善Dice，说明减少重复有价值；但9.57%小病灶回放没有保住Point，也没有超过KD200 Scan，所以完整晋级直觉失败。只否定当前680/72、seed20260884、32轮实现，不证明所有覆盖采样无效。

**实际代价：** formal两臂`1935.0/1938.5s`，两卡并行约32.3min；每卡20.29GiB；workspace约253MB、不建cache；部署仍4.70M参数、1×forward。

**下一决定：** 停止当前配方，不搜索680/72比例、epoch、seed或阈值。若继续数据轴，应独立预注册972 effective draws/epoch的matched-compute实验，不能用本validation调设置。证据：[结果](archive/research_records/2026-08-24_coverage_small_lesion_replay_kd/RESULT.md)、[机器结果](archive/research_records/2026-08-24_coverage_small_lesion_replay_kd/RESULT.json)、[七问](archive/research_records/2026-08-24_coverage_small_lesion_replay_kd/IDEA.md)、[raw report](workspace/BraTS/sammed3d_t1ce_wt/coverage_small_lesion_replay_kd_20260824/COVERAGE_REPLAY_VALIDATION.json)、[CSAL-3D论文](https://papers.miccai.org/miccai-2025/paper/2315_paper.pdf)、[官方代码](https://github.com/HiLab-git/CSAL-3D)。

## 当前推荐

### Full972 width24完整轨迹 ❌final validation失败

**一句话：** Full972教材变厚后，把单模型从4.70M扩到10.57M参数，测试是否解除train-only欠容量。

**状态：** Inner曾通过并选择epoch166；随后Full972 width16/24用新seed重训到166并完成fixed validation108。Width24的Scan/Subject/Point均低于matched width16，promotion失败，不访问test。

**出发点：** Full972 width16 train Dice低于Full756；此前160³和late adapter负结果未测试随机初始化的全程宽度扩展。

**具体做法：** 同candidate-fit864/common-inner108、seed20260960、968 draws/121步、teacher/loss/schedule；仅init_filters16→24。Inner过门后full972新seed重训同一epoch。

**唯一变化/固定项：** 只改模型宽度与参数量；数据、计算步数、teacher、loss、epoch选择、阈值与后处理固定。两者均1×forward。

**成功/停止门：** Inner需超过width16同epochScan/Subject、Point within1和CI门；final需超过matched width16及seed32 validation。Peak>32GiB、单臂>8h或工程失败即停。

**观察结果：** Matched width16=`0.739335/0.731320/105/108`；width24=`0.736464/0.727795/104/108`。Width24相对width16 Scan/Subject=`-0.002872/-0.003525`，95% CI分别`[-0.014258,+0.005947]`/`[-0.013330,+0.005515]`。相对archived seed32也为`-0.003549/-0.003471/+1 point`。四个点估计门和四个CI门均失败。

**是否支持直觉：** 不支持。Inner提升没有在新seed的完整重训中复现；更大模型耗时更长但泛化更差，说明当前配方的主要瓶颈不是4.70M容量不足。

**实际代价：** Final width16约3.56h，width24约4.86h；width24参数约2.25倍。Fixed validation已访问；outer/confirmation/test未访问。

**下一决定：** 关闭width24，不搜索width20/32、不改epoch或阈值。保留width16容量配置，下一实验换机制并新预注册。证据：[结果](archive/research_records/2026-08-26_full972_width24_fulltrajectory/RESULT.md)、[机器结果](archive/research_records/2026-08-26_full972_width24_fulltrajectory/RESULT.json)、[raw report](workspace/BraTS/sammed3d_t1ce_wt/full972_width24_20260826/WIDTH_VALIDATION.json)、[七问](archive/research_records/2026-08-26_full972_width24_fulltrajectory/IDEA.md)、[预注册](archive/research_records/2026-08-26_full972_width24_fulltrajectory/PREREG.md)。

### Full972多seed等权ensemble并蒸馏 ❌点估计正向/CI门失败

**一句话：** 新训练seed40/41并与seed32固定各1/3平均logits；若稳定过门再蒸馏回1×。

**状态：** 两个新seed168轮、工程门和固定ensemble validation已完成。Ensemble点估计明显超过seed32，但Subject CI安全门失败，按预注册不蒸馏、不访问新test。

**出发点：** Full972 seed32内部test为`0.751760/0.737327/262`，但只有一个随机轨迹；旧parameter soup不否定logit互补。

**具体做法：** seed40/41复用Full972协议；与seed32按raw logits各1/3，validation不搜索seed或权重。

**唯一变化/固定项：** 训练只变seed；ensemble只加固定logit均值。数据、模型、epoch、draw、teacher、loss、阈值和后处理固定。

**观察结果：** Ensemble=`0.745841/0.735684/104`；seed32=`0.740013/0.731266/103`；seed40=`0.733408/0.722941/101`；seed41=`0.736000/0.724980/103`。相对seed32 Scan/Subject=`+0.005829/+0.004418`，88/108 scans改善；Subject CI下界`-0.005660`失败`-0.003`门。

**是否支持直觉：** 部分支持。seed错误互补带来明显点估计收益，但50-subject层不确定性不足以授权条件蒸馏。

**实际代价：** 两卡约3.61h、每卡20.29GiB、workspace217MB；ensemble3×，未进入1×student阶段。

**下一决定：** 停止，不调权重、不剔除seed、不蒸馏、不访问新test。证据：[结果](archive/research_records/2026-08-25_full972_multiseed_ensemble_distill/RESULT.md)、[机器结果](archive/research_records/2026-08-25_full972_multiseed_ensemble_distill/RESULT.json)、[raw report](workspace/BraTS/sammed3d_t1ce_wt/full972_multiseed_20260825/THREE_SEED_ENSEMBLE_VALIDATION.json)。

### 972病例随机初始化完整voxel-KD轨迹 ✅探索性内部test新高/failed-gate标记

**一句话：** 新增216例不再期末补课，而是从随机初始化第一轮就参与；先用公共train-inner选epoch，再重训full756/full972。

**状态：** 两阶段完成；validation只失败KD200 Subject CI门。用户随后明确授权一次failed-gate exploratory reused test270，取得`0.751760/0.737327/262/270`，相对旧ensemble和KD200的Scan/Subject CI均整体为正，成为当前内部总体/单模新高；不是盲测或正式promotion。

**出发点：** budget968 continuation达到`0.734259/0.724956/100`，但新增病例只在history168后出现。完整轨迹测试新增病例从表示学习早期参与。

**具体做法：** Inner control648/candidate864共享108-case subject-clean tune，两臂同968 draws/121步跑200轮并由candidate选择epoch168；随后full756/full972以新seed、相同随机初始state和计算量重训到168。

**唯一变化/固定项：** 只改eligible病例池；inner tune、计算量、随机初始state、模型、teacher、loss、LR、增强、阈值与1×推理固定。

**成功/停止门：** Final必须超过full control与KD200 Scan/Subject、Point≥101、四项CI下界≥-0.003才授权test。

**观察结果：** Candidate=`0.740013/0.731266/103`；full756 control=`0.729070/0.723284/104`；KD200=`0.734384/0.724434/101`。相对control Scan/Subject=`+0.010943/+0.007982`，Scan CI整体为正；相对KD200=`+0.005628/+0.006832/+2 points`，但Subject CI下界`-0.004484`低于`-0.003`门。

**是否支持直觉：** 强支持。新增216例从随机初始化参与明显优于late continuation；失败的是保守subject CI授权，不是点估计或机制。

**实际代价：** Inner约4.42h、final约3.61h，两卡总墙钟约8h；每卡20.29GiB；推理仍1×。

**Test结果：** 相对ensemble=`+0.006705/+0.009982/+4 points`，相对KD200=`+0.008783/+0.012149/+4 points`，四项paired CI下界均大于0。禁止用test继续调参。

**下一决定：** 保留为failed-gate exploratory内部新高，寻求外部/独立确认。证据：[结果](archive/research_records/2026-08-24_voxel_kd_fulltrajectory_972/RESULT.md)、[机器结果](archive/research_records/2026-08-24_voxel_kd_fulltrajectory_972/RESULT.json)、[raw report](workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824/FULLTRAJECTORY_VALIDATION.json)。

### 972病例配额一致voxel-KD ❌预算机制支持/未晋级

**一句话：** 972例训练池不再沿用756例的做题量，把每轮有效draw从752增加到968，测试数据多样性是否需要相称计算预算。

**状态：** 2026-08-24已完成预注册、10项合同测试、双卡smoke/formal和fixed validation108。Candidate超过same-epoch control Dice且Subject超过KD200，但KD200 Scan和Point门失败；不访问test。

**出发点：** KD200用756例取得validation Scan`0.734384`；三次972实验都显示多样性相对matched control有益，但每轮仍只有752有效draw。最新coverage candidate相对control`+0.001481/+0.001890`，却仍未超过KD200 Scan且Point下降。

**具体做法：** 两臂都用972病例、原生体积权重和相同voxel-KD。Control每轮752有效draw/94步；candidate每轮968有效draw/121步，32轮内总step增加28.72%。

**唯一变化/固定项：** 唯一变化是每轮draw/step预算；病例、cache、模型、teacher、loss、seed20260924、batch、epoch grid、增强、优化器、阈值和1×推理固定。

**成功/停止门：** selected candidate需同时超过same-epoch control和KD200 Scan/Subject，Point≥101/108、四CI安全门及24GiB/60min/reload门全过才允许test。

**观察结果：** selected candidate200=`0.734259/0.724956/100/108`；same-epoch control=`0.733480/0.724003/100/108`；KD200=`0.734384/0.724434/101/108`。相对control Scan/Subject=`+0.000779/+0.000954`；相对KD200=`-0.000125/+0.000522/-1 point`，四CI安全门均通过。

**是否支持直觉：** 部分且明确支持。增加28.72%训练预算同时提高matched-control Scan/Subject，说明972病例此前确实训练不足；但仍没有超过KD200 Scan，Point也少1例，完整晋级失败。

**实际代价：** control/candidate 32.1/41.3min；每卡20.29GiB；workspace约181MB；推理无变化。

**下一决定：** 停止当前968-draw×32轮协议，不事后延长。证据：[结果](archive/research_records/2026-08-24_voxel_kd_972_draw_budget/RESULT.md)、[机器结果](archive/research_records/2026-08-24_voxel_kd_972_draw_budget/RESULT.json)、[raw report](workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_972_draw_budget_20260824/DRAW_BUDGET_VALIDATION.json)。

- 当前reused internal test270总体三项点估计最高：Full972 KD168固定Flip8，0.757270/0.741254/265，单checkpoint、8×推理；不是盲测或外部结果。
- 当前1×推理范围：GCSAM184的Scan点估计最高0.752003；Full972 KD168的Subject/Point最高0.737327/262。
- UCSD75 official-float中GCSAM184 Scan/Subject点估计最高0.755890/0.755312，Full/GCSAM Point同70/75；这是同已访问cohort来源纠正，不是第二次独立外部。
- Africa95中GCSAM184 identity Dice点估计最高0.591619、相对Full identity `+0.068617`且CI全正，但Point低3；Full Flip8 Point最高88/95且Dice收益CI跨0。Public post-challenge、四中心机构独立/no patient bridge，不与UCSD/internal混排。
- CPTAC33中总体点估计最高为Full KD168固定Flip8 `0.844577/33/33`；最高1×是GCSAM `0.843285`，但相对Full identity仅`+0.001954`且CI跨0。Public Training labels external-to-project、非hidden blind，不与其他cohort混排。
- WAW152中GCSAM identity与Full Flip8 Dice=`0.693601/0.693494`近乎tie；GCSAM相对Full identity `+0.010223`但CI跨0，Flip8相对identity `+0.010116`且CI全正、Point+2。华沙public preop、institution-disjoint/no patient bridge、非hidden blind，独立分榜。
- WAW early-postop31中Full Flip8/GCSAM/Full identity Dice=`0.598959/0.593305/0.591647`、Point=`15/14/15`；两配对差均小正但CI跨0。患者与WAW-pre152零重叠、day0–22、public非hidden；Point miss约80%落GT RC但禁止GT避让。
- 972 immutable-role fresh评测容量只剩2人/3 scans；不得从历史fit或已揭示E/F重启新方法。下一步取决于EGD DUA、UTSW数据字典、Africa2024新发布等[外部阻塞](archive/reviews/2026-09-05_30h_chain_audit/EXTERNAL_BLOCKER_REGISTER.md)。
- 每个新方法必须先按[`EXPERIMENT_IDEA_TEMPLATE.md`](EXPERIMENT_IDEA_TEMPLATE.md)写小白说明，再实施。
