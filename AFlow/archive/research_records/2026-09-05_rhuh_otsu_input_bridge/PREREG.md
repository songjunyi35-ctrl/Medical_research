# RHUH Otsu-N4输入桥接冻结协议

状态：2026-09-05 UTC，在任何RHUH标签下载、模型输出或Dice/Point之前冻结。

## 来源与访问角色

- 固定TCIA package684后时点1/2全部80 scans/40 subjects。复用`RHUH_IMAGE_QC_PREACCESS.json` SHA256=`65892e4d90a577ff2623cd2786d5142aeed86be809a5b273faf245321a4481c9`与下载收据SHA256=`c6f17decbc81e8a2bc8ec385438b7e83547644aed26009fb376bd27a218d725f`。
- 输入阶段不下载/stat/open segmentation，不加载checkpoint，不import模型，不计算指标。No-N4桥接结果不能替代本Otsu-N4输出门。
- 冻结Otsu机制收据SHA256=`2b0f6e4b2b2b008d2669b79423203a2f348fee981bc3e50176721d9047d720da`；算法必须复用当前`n4_otsu_correct`实现。

## 变换与门

1. 重验80图SHA，canonical RAS/finite 3D/正spacing。
2. 固定背景解码：六面众数≥0.95且等于global min，支持exact。
3. 固定`OtsuThreshold(image,0,1,200)`、shrink2、N4迭代`[50,50,50,50]`；拟合mask非空且不含非正输入，校正后finite且完整正支持exact。
4. 正支持image-only bbox；源正值float64 population mean/std、float32标准化；crop后trilinear resize128³，mask nearest-exact清零。标准化abs mean/std error≤1e-6，输出finite/mask非空。
5. 80/80所有门通过；固定下载顺序前3例再次独立运行，image、brain mask与Otsu fit-mask SHA均exact。
6. 20分钟CPU时间门、持久新增≤20MB。失败不调任何参数、不删病例、不运行模型。

## 后续锁

- 格式通过只证明可执行。只有Otsu候选E/F两折identity/Flip8全门通过且条件Full972 Otsu完成，才能另冻checkpoint/标签/metric preaccess并下载80标签。
- 未来WT为所有positive published tumor labels union；病例全纳入。Primary固定Flip8、threshold0、Point=argmax、无SAM/后处理、患者簇bootstrap10,000 seed20261362；绝对描述门Scan/Subject≥0.70、Point≥72/80。
- 无论结果正负只运行一次；不回流选择训练、输入、checkpoint、view或阈值。不访问fixed validation、outer、confirmation、test270或UCSD75。
