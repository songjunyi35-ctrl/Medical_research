# Batch-CVaR50 fixed-validation负结果

## Outcome

source-inspired Batch-CVaR50完成了一手Top-k/FairDRO可行性核验、七问预注册、15项合成测试、三轮GPU smoke审计、uniform/CVaR双臂32轮formal、完整配对和一次fixed validation108决策。

所有implementation、legacy parity、两臂pairing、24GiB显存和8个checkpoint reload门最终通过；但candidate在overall、Point、KD200-bottom27和boundary-top27均失败。因此不授权、不访问test270。

## Method and only changed factor

- 两臂都从KD200开始，每个scan draw独立计算历史`Dice+0.1×BCEWithLogits`。
- uniform control对8个draw求均值。
- candidate用stable detached排序选loss最高4个draw，对原attached loss求均值。
- 重复case按draw分别参与，不去重。
- 唯一变化是`mean8→top4 mean`；seed20260860、sampler、增强、optimizer、LR`1e-5`、201–232轮、grid和推理完全相同。
- 无teacher、额外参数或部署变化。

## Implementation audit history

首次smoke只检查了top4/gradient/memory，审计后发现缺少真实GPU legacy parity门。补门后的v2显示FP32 per-draw与历史autocast DiceCE loss差`0.000335157`，超过冻结`1e-7`门，故在formal前停止。train-only诊断确认保持历史autocast语义时，loss和raw-logit gradient差均精确0；v3修正精度语义后双臂重跑，所有门通过。公式、top4比例和训练schedule未调。

## Fixed validation108 / 50 subjects

Candidate仅按Scan Dice选择epoch208。

| Epoch | Uniform Scan / Subject / Point | CVaR50 Scan / Subject / Point |
|---|---|---|
| **208 selected** | `0.733194 / 0.723334 / 0.944444` | **`0.731801 / 0.721922 / 0.925926`** |
| 216 | `0.732892 / 0.723054 / 0.925926` | `0.729521 / 0.720119 / 0.935185` |
| 224 | `0.734358 / 0.724619 / 0.935185` | `0.731400 / 0.721739 / 0.962963` |
| 232 | `0.734086 / 0.724591 / 0.916667` | `0.731182 / 0.722046 / 0.898148` |

Selected CVaR208相对same-epoch uniform：

- Scan `-0.001393491`，95% CI `[-0.002441700,-0.000461465]`；43/108改善、65变差。
- Subject `-0.001412096`，95% CI `[-0.002704109,-0.000193553]`；16/50改善、34变差。
- Point `-2/108`，仅100/108。

相对KD200：

- Scan `-0.002583438`，95% CI `[-0.004649004,-0.000649688]`。
- Subject `-0.002512257`，95% CI `[-0.004860688,-0.000311994]`。
- Point `-1/108`。

## Tail mechanism

- KD200-bottom27：candidate−uniform mean `-0.002274960`，14改善、13变差，CI `[-0.006041857,+0.001045631]`。
- Boundary-top27：mean `-0.001550842`，10改善、17变差，CI `[-0.003760735,+0.000616588]`。

因此top4聚合没有修复预指定hard tail。部分极差病例改善，但少数较大退化抵消了收益，与overall负结果一致。

## Promotion decision

通过：全部前置门，以及相对uniform的四个CI安全下界。

失败：超过uniform Scan/Subject、超过KD200 Scan/Subject、Point≥101、相对KD200两项CI、bottom27正delta、boundary27非负delta。

`candidate_authorized_for_test=false`；无test evaluator、test receipt或Leaderboard更新。

## Interpretation

CVaR确实提高了困难draw的训练权重：首batch聚合loss从uniform约`0.187`升到`0.218`，grad norm从约`0.436`升到`0.759`。但batch8内hardness是瞬时、augmentation依赖的，WeightedRandomSampler又已做一次重采样；二次强调没有稳定对应validation困难病例。

来源论文也显示ACDC 2D slice上的固定hard AT10%会降分，只有全数据curriculum和平滑组合才恢复。当前top4/8保留更多样本但仍是hard batch选择，负结果与该风险一致；不能外推为所有CVaR、全数据Top-k或完整论文方法无效。

## Cost and deployment

- Uniform/CVaR formal `1107.86/1106.53s`，双卡墙钟约18.46min。
- 两臂peak allocated均约20.29GiB。
- workspace约145M，无新cache。
- 8个checkpoint reload/deploy差均0。
- 部署仍4,700,897参数、1×forward。

## Evidence and access

- Raw validation SHA-256 `cd7a081daf3ae73eb54cb4b881864274125c401e00458d4e99a2204a71862a95`。
- Validation started SHA-256 `faaad0e76fb09ba0932717948ab5de287e2b0c5804cbbcb109966f95d247452d`。
- Pairing receipt SHA-256 `2989f132481b7280bce3ec129a11516652116777cd520c3bb0498e5b89642fe1`。
- Uniform/CVaR TRAIN_DONE SHA-256 `79e246a3...ba229` / `fec271ce...5f673`。
- Selected CVaR208 checkpoint SHA-256 `791ca9d00ed1fc319ceaa34a3ccec7e97c691233ce71e2dc4c77c9cd94bfafb1`。

本方法的新fixed validation108已访问；起点KD200有历史validation及随后test访问。本方法未访问outer selector、confirmation或新test，也未使用test选择top比例、epoch、LR或阈值。

当前Batch-CVaR50停止，不调参。近期多条loss/teacher约束均未超过KD200；下一步应离开loss轴，重新核验局部原生分辨率或架构信息瓶颈。

Primary source: [Top-k medical segmentation paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12428111/).
