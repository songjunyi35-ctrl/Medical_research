# RACE164专属108验证推理校准完成

固定RACE164、单T1ce、原108/50、原生1mm裁剪全GT评分，32个事前冻结全局配置。最高配置为`equal8_logit_t05`，Scan **0.768011881**、Subject **0.758259859**、Point **98/108**。全部为既有test-tuned研究历史上的复用验证选择，非独立泛化确认。

| 配置 | Scan | Subject | Point | 部署前向/例 |
|---|---:|---:|---:|---:|
| raw_identity_t05 | 0.763265668 | 0.753084628 | 97/108 | 1 |
| equal8_logit_t05 | 0.768011881 | 0.758259859 | 98/108 | 8 |
| legacy_full | 0.764705181 | 0.755743131 | 99/108 | 8 |

对当前完整配方Scan差+0.003306699，患者整簇配对描述性95%区间[8.769527911095618e-06, 0.00815019853538794]；Subject差+0.002516727，区间[-0.0009719713780851592, 0.006612862033148605]。Point少1，Subject配对区间跨0，不能称三项联合稳定提升。最高配置Scan区间[0.7279056461996511, 0.8030896060275887]。均为10000次患者整簇bootstrap、seed20261029，未校正32配置选择。

固定5折按患者隔离，四折选参/留出折汇总Scan 0.768011881、Subject 0.758259859、Point 98/108；留出折既有配方Scan 0.764705181。各折与全部候选表现保存在RANKING.json，并非新独立确认。

108使用160×208×160历史画布，4个实际载荷SHA与全部几何匹配旧108。RACE镜像参照矩阵T R T共轭、非零支撑同步翻转。一次864病例前向，batch8/尾4，耗时90.9秒、reserved峰值28.600GiB；32解码不增加网络前向。Point的raw/equal采用logit argmax；完整配方及变体保持历史概率argmax和饱和并列首位置规则，二者没有混为同一定位规则。

全部32配置逐例TP/FP/FN、Point、实际阈值和视图权重保存在RAW_VALIDATION108.json；相对三个参考的全部变差病例保存在ALL_COMPARISONS.json。三个预定108槽位0/53/107的旧完整配方CPU/GPU解码掩膜、Point和量化cut全部相同，神经前向增加0。原始推理与旧训练验证的执行差异见BASELINE_EXECUTION_COMPARISON.json，不把新batch/hardware声称为逐位同轨迹。

旧270四载荷、index、权重SHA及2160/2160既有logits的SHA/shape/dtype/finite全部核验通过，可在总控冻结最多两个候选后复用。本阶段新增270前向0、新270解码评分0。

[计划](PLAN.json) · [32配置逐例](RAW_VALIDATION108.json) · [排名与分折](RANKING.json) · [完整比较](ALL_COMPARISONS.json) · [入选配方](SELECTED_RECIPE.json) · [270缓存只读审计](EXISTING270_CACHE_AUDIT.json)
