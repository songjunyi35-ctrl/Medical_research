# 患者等概率采样：前瞻冻结的复用开发实验

2026-09-07冻结于当前辅助监督dev揭示之前。依据仅为fit-only采样概率审计；无论辅助监督实验dev支持或失败，本方案内容均不改变。只在前一完整流程成功完成工程/评测归档后运行，不与其GPU训练竞争。

## 数据与对照

- 授权角色完全不变：fit684/313，SHA75504df75fa734723945a9f07d76696fdcca89b8c29563dc0ad02a2560718d61；dev180/78，SHA53885b7e300f915d81eecb115b82123e25330b7da422d3ff0b889b0489a2cdac。
- 对照预先指定为`structured_background_aux_20260907/seed2026090711_control`与`seed2026090712_control`的fixed_epoch168；不选择其他模型。两control由前一预注册产生，同fit、无teacher、双通道结构但RC辅助系数0。控制模型的checkpoint/report SHA将在本候选训练前收据冻结。
- 复用原控制180例输出，避免重复推理；明确reused control/reused development，不宣称同时间重训或独立患者重复。

## 唯一改动

原始w_i为native WT inverse-sqrt、clip[0.5,4]。候选w'_i=w_i/sum_{j属于同患者}(w_j)；全局归一化后每患者概率=1/313，患者内扫描比值保持原w_i/w_j。不新增clip、温度、混合系数、病例剔除或临床分组。扫描数分布1–10只影响每位患者内部如何分配机会。

## 匹配控制项

复用前一冻结训练实现，新增入口只替换采样权重函数和实验输出/证据目录，使用新arm名subject_balanced；原训练代码对该arm辅助系数为0，与control一致。不得修改旧哈希绑定代码。

模型与export沿用StructuredBackgroundLocator，两通道分别计算主输出，末层导出主通道；WT DiceCELoss(sigmoid=True,squared_pred=False,lambda_dice1,lambda_ce.1)，aux仍计算但系数0，无教师/KD。缓存图像/WT/RC、fit、初值、batch8、972 draw→968有效draw/121步、168轮、AdamW lr2e−4/wd1e−5、100恒定后既有200轮cosine到168、min1e−5、BF16、gradclip5、max_shift8和所有随机增强均不变。seeds固定2026090711/12。

新入口每seed模型初始SHA必须与相同seed旧control一致；增强RNG每轮必须相同；采样索引和首batch图像/标签预期不同，不能错误要求两臂batch byte exact。候选不得加载任何历史checkpoint暖启动。

## 计算和工程门

- 两候选在双A100同时从零训练，每臂固定168轮，上限6h/24GiB。当前辅助监督完整流程结束后才进行本实验GPU smoke和formal。
- CPU合成权重门：每患者总权重=1到atol1e−12、患者内比值不变、所有权重finite/positive、相同seed序列可复现；native identity恢复门验证未做归一化时旧权重exact。
- GPU smoke：同模型初始state、两种采样权重合规、增强RNG一致、loss/gradient有限、导出主输出error0。训练前冻结新IDEA/PREREG、包装入口、采样函数、原训练实现、split/cache/control receipts及smoke SHA。
- 不重建缓存；复制已有AUX_CACHE_RECEIPT仅作证据路径适配，不复制1.34GiB数据。新训练输出与旧对照分目录。

## 开发评测与停止

两候选全部完成并冻结后，只运行各自identity一次，共360单例forwards，dev180同顺序、batch8/BF16、WT阈值0、Point=主logit最大点；无TTA/SAM/后处理。原control逐例输出直接从前一RAW_DEVELOPMENT读取并核SHA、case/subject/target计数。

主要Subject Dice，次要Scan Dice/Point。10,000次患者重采样percentile，保留患者内所有扫描，Subject等权、Scan用该次重采样总分/扫描数；配对seed分别2026090830/2026090930；候选绝对seed2026091830/2026091831。两个训练seed共用同78位dev患者，不合并成156独立患者。

每seed candidate−对应control Subject≥.003、Scan>0、Point≥−1；Subject与患者聚类Scan CI下界均≥−.003；两seed平均Scan≥.003。全部通过仅reused-development supported；否则停止当前患者等概率实现，不搜patient temperature、混合比例、epoch或seed。任何工程门失败先记录边界，不进入dev；不自动授权test/confirmation/outer/外部新访问。

## 证据与限制

新增TRAIN_PREACCESS、IMPLEMENTATION_FREEZE、SMOKE、两条TRAIN_DONE、DEV_PREACCESS、RAW_DEVELOPMENT、RESULT.md/JSON、VERIFICATION。新模型代码仍来自原二头实现但aux=0，不能把效果归因RC学习。训练源Kaggle与官方Synapse parity限制保留。概率覆盖更均匀不保证精度；减少全局小病灶偏重可能损害Point。
