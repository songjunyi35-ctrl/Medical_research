# Experiment idea: 972例原生体积感知voxel-KD采样

Status: planned and frozen before native-volume generation, implementation, training, and validation.

## 1. 现在的问题是什么？

固定计算量数据扩容已经证明972病例池相对cache-matched 756 control可提高fixed validation108 Scan/Subject `+0.003297/+0.004287`，两项95% CI均高于0。但selected candidate184仍比历史KD200 Scan低`0.000335`，Point只有99/108、比KD200少2例。

两条训练轨迹的关键已知协议差异是采样体积口径：历史KD200用原始1mm标签中的WT体积计算困难病例权重；现有972旧cache只暴露resize到128³后的标签体积。高边界、小病灶在resize后的体积偏差可能更大，也正是当前Point和hard-tail较弱的病例。

## 2. 一句话直觉

病例数量已经增加了，下一步把每份病例的“抽题频率”从缩小后的近似大小改回原始1mm空间里的真实肿瘤大小。

## 3. 为什么选择这个idea？

- 972 candidate已经在Subject上超过KD200 `+0.000260`，Scan只差`0.000335`，不是从明显失败方向重新下注。
- 相对严格matched control的正CI说明增加病例多样性是真机制；现在应修复唯一已知的sampling口径差异，而不是再换loss。
- 历史成功KD200本身使用原生体积权重并保持101/108 Point，提供直接项目内证据。
- 当前实验不会把上次validation用来搜索clip、指数或draw数；权重公式固定恢复为历史`sqrt(median/volume)`并clip到`[0.5,4]`。
- LoMix、DCD、KAS、hACE、VM Dice、CVaR和DMRFB负/部分正结果都没有测试“972病例＋原生体积权重”，因此没有否定这条数据处理方向。

## 4. 具体怎么做？

先从972个development-train原始NIfTI标签读取WT `{1,2,3}`体素数，生成小型JSON。对旧756重叠病例，这些计数必须与historical formal128 index保存的`per_case_source_positive_voxels`逐例完全一致。

正式训练仍有两臂：

- `native756_control`：旧756病例，按原生WT体积加权抽样。
- `native972_candidate`：全部972病例，按原生WT体积加权抽样。

两臂Student从historical168初始化；Teacher仍是`0.75 history168 + 0.25 EMA168` raw-logit融合；loss仍为Dice+0.1交叉熵加`0.3*T²*voxel KL`、`T=2`。每轮都从各自population加权抽756次，drop-last后752 draws / 94 steps，共32轮。输出仍是plain 128³ raw logit单模型。

## 5. 唯一变化与固定项

- 两臂之间唯一变化：可抽样病例池`756→972`。
- 相对上次数据实验的唯一协议修正：两臂统一使用原生1mm WT体积，而不是128³缓存标签体积计算sampling weight。
- 固定：同一972 cache的image/label、history168初始化、两个teacher、KD公式、DiceCE、seed20260846、batch8、756 sampler draws、94步、32轮、late-cosine LR、AdamW、weight decay、shift8、clip5、epoch `176/184/192/200`、threshold0、无TTA/SAM/postprocess、1×部署。

## 6. 成功、失败和停止规则

- Candidate只按fixed validation108 Scan最大选择；Subject exact tie，再以更早epoch作最终tie。
- Selected candidate必须同时严格超过same-epoch `native756_control`和历史KD200的Scan/Subject，Point至少101/108。
- 相对两个comparator的Scan/Subject 10,000次paired percentile bootstrap 95% CI下界均须不低于`-0.003`。
- 原生体积756 overlap非exact、任何体积<=0、weight公式/历史756 weight非exact、loss或梯度非有限、teacher有梯度、batch8 peak allocated>24GiB、单臂机械外推或formal>45min、checkpoint reload非exact立即停止。
- 只有全部门通过，才冻结唯一checkpoint并访问一次reused internal test270。失败后不得改weight指数、clip、draw数、epoch、LR、阈值或后处理。

## 7. 成本与风险

- 原生体积处理：读取972个train标签，输出几十KB JSON，不建新cache；预计数分钟。
- 训练：两张A100并行，每臂约32分钟，硬门45分钟。
- 显存：预计每卡20.3GiB，硬门24GiB。
- 磁盘：复用9.5GB cache；8 checkpoint和报告约180MB。
- 推理：4,700,897参数、1×forward，零新增部署成本。
- 最大风险：上次candidate相对KD200的差距可能来自不同病例轨迹而非sampling；原生权重可能恢复Point，也可能过度重复小病灶而降低overall。

## 8. 最终结果（完成后填写）

- Validation：待运行。
- Test：仅全部门通过后允许。
- 是否支持直觉：待运行。
- 下一步：待运行。

