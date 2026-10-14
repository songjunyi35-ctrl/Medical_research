# 下一份外部数据的最低验收规范

目的：在任何大下载、cache或模型前，用统一门判断新来源是否真的提高证据等级。它不授权外部联系、签协议、调参或评测。

## A. 法律与来源

1. 官方记录必须明确open license，或由用户/机构本人完成Data Usage Agreement；Codex不能代签。
2. 冻结collection/version/DOI、文件清单、bytes、publisher checksum。无publisher checksum时至少两次独立官方传输SHA/byte exact并做容器CRC。
3. 第三方整理镜像只能称mirror。必须有官方member parity才能升级为official；UCSD已证明data-nih镜像可把float32变uint8，UPENN又只有147 seg而非官方232。
4. 下载前检查磁盘：source+第二份传输（若需）+提取+128³ cache+至少2GiB reserve；只能按授权删除已hash归档、raw可重建的派生cache。

## B. Cohort与数据角色

1. 在像素前冻结case/subject/timepoint清单、排除原因与顺序；不能按模型、图像质量或标签内容事后挑例。
2. 列出训练七站与候选机构；有patient crosswalk时必须排重。没有桥只能写institution-disjoint/no-known-overlap，不能写patient-ID-proven disjoint。
3. 明确public、post-challenge、Training-label、reused source correction或hidden blind；不同身份分榜。
4. 新cohort只访问一次；不得回流checkpoint、threshold、view、fusion、N4、postprocessing或病例选择。

## C. 标签语义

1. 发布者必须逐数值定义所有raw labels；出现UTSW式未定义3/5立即停止，不能猜all-positive。
2. 当前WT必须包含necrotic/nonenhancing core、FLAIR abnormality和enhancing tumor，并排除resection cavity。GTV/CTV/PTV、自动mask或只有enhancing+necrotic core不能混入人工WT榜。
3. 自动初始+专家修订、完全人工、自动融合分别标证据等级；不能用论文总体“人工QC”代替逐病例corrected availability。
4. 统计only-FLAIR label2与enhancing label3是否存在，但只用于解释可观测性，不能用GT组成排例或router。

## D. 图像—标签工程门

1. 每例native image/label shape exact、affine误差在预冻容差内、离散orientation一致；canonical转换不得插值GT。
2. 若独立canonical affine受浮点放大，可另立共享离散orientation桥：先用更严格native门、同一image-derived transform作用所有arrays、跨进程SHA exact；不能在失败协议中放宽门。
3. 目标必须100%落在不看GT定义的图像支持/crop内。MOTUM即使nonzero bbox仍丢892 WT，说明仅affine exact不够；不得用GT扩框或事后padding。
4. 报告source grid/spacing/slices、publisher resampling和项目128³ rasterization；21/24层厚层不能与1mm数据按绝对Dice混排。
5. 首例probe与全量worker使用相同线程/库设置，首例最终image/label byte exact；Africa曾证明线程不一致可使N4差`4.48e-5`。

## E. 评测与样本量

1. 模型面板、checkpoint SHA、阈值、TTA、SAM/后处理、bootstrap seed和解释门在性能前冻结；started receipt先于checkpoint/model。
2. 报告Scan、Subject、Point、absolute CI、paired CI与improve/worse/tie；CI跨0只写不确定。
3. 不把cohort点估计pool成总体排名。Sample-size正态敏感性只作粗规划：
   - GCSAM−Full若关心Dice `+0.01`，加入WAW-post后的observed external SD场景约需313–1149 independent subjects；关心`+0.02`约79–288。
   - Flip8−Full若关心`+0.01`约79–353；关心`+0.005`约314–1411。
   - 新数据前必须先定义最小有意义delta；这些数字不是功效保证。
4. 推理成本必须随结果写：identity 1×、固定Flip8 8×；参数不增不等于延迟免费。
5. Postoperative cohort若有RC4，结果后可离线报告argmax落WT/RC/背景，但GT RC只作错误归因；不得在同cohort据此加RC avoidance。当前binary WT训练虽814/972 scans见过RC，却把RC与背景同编码，未来显式RC head需全新开发证据。

## F. 当前优先解除顺序

1. EGD374：用户/机构完成Health-RI DUA后，先取manual/quality/overlap metadata；其规模最接近有意义的Flip8外部设计。
2. UTSW362：发布者给3/5官方字典后重冻target，不能沿用失败cache。
3. Africa2024新20病例：TCIA公开后只取与95旧病例ID差集。
4. UPENN232：需官方完整manifest/member parity与BraTS患者桥；不能按镜像147选择。
5. MOTUM29：需发布者一致brain support/修正版，或未来从零、全FOV且独立过门的模型；不GT修框。

来源登记与解除主体见[外部阻塞表](EXTERNAL_BLOCKER_REGISTER.md)，粗样本量证据见[规划报告](../2026-09-06_future_external_sample_size_sensitivity/RESULT.md)。
