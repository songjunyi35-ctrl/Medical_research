# 固定Flip8跨cohort只读综合：结果

状态：五个已有raw报告离线综合完成；0训练、0图像/checkpoint读取、0模型forward、0选择或调参。

## 初学者结论

固定Flip8测试时增强（Test-Time Augmentation，TTA）让同一个模型从三轴全部八种镜像看同一MRI，把输出翻回后平均。五个已运行cohort的Scan Dice点估计都提高了：reused internal270、CPTAC33和WAW preop152的配对区间整体高于0；Africa95与WAW early-postop31跨0。

| cohort | identity Scan | Flip8 Scan | Flip8−identity Scan（95% CI） | Point | 证据身份 |
|---|---:|---:|---:|---:|---|
| internal test270 | `0.751760` | `0.757270` | `+0.005510` `[+0.002751,+0.008022]` | `262→265` | reused、post-test development、非盲/非外部 |
| BraTS-Africa95 | `0.523003` | `0.526438` | `+0.003435` `[-0.010536,+0.016815]` | `87→88` | public post-challenge、非hidden blind |
| CPTAC33 public Training | `0.841331` | `0.844577` | `+0.003246` `[+0.000828,+0.005855]` | `33→33` | external-to-project、非hidden blind |
| WAW-GliomAI152 | `0.683378` | `0.693494` | `+0.010116` `[+0.004472,+0.016274]` | `141→143` | public preop、institution-disjoint/no patient bridge、非hidden blind |
| WAW early-postop31 | `0.591647` | `0.598959` | `+0.007312` `[-0.001925,+0.016626]` | `15→15` | preop患者零重叠、day0–22、public非hidden blind |

外部四个队列每人只有一张scan，所以Scan与Subject Dice相同。Internal270的Subject提升为`+0.003926`，其区间跨0；不能因Scan区间为正就声称所有指标显著。

## 能说与不能说

- 能说：同一冻结Full972 KD168 checkpoint在五处的Scan点估计方向均为正；internal270、CPTAC33、WAW preop区间支持，Africa95与WAW postop不确定。Point差没有变差（`+3/+1/0/+2/0`）。
- 不能说：五个cohort不是同一抽样总体，不能合并成总体效应、显著性或排名；均已揭示，不能据此搜索视图/权重/阈值/router。
- UCSD75没有运行固定Flip8，未填补、未推断。
- 旧“群等变一致性训练”失败不矛盾：它否定的是当前训练损失能把8×收益压回1×；本综合描述的是昂贵的部署期八视图平均。

## 代价与决定

本综合新增存储低于1 MiB，新增GPU/forward为0。真实部署仍需每例8次locator forward，计算和延迟约8×；参数、checkpoint和峰值显存近似不变。固定Flip8保留为“有跨队列方向支持、但代价高且Africa不确定”的部署选项；不再运行已见cohort，也不把结果回流为新调参依据。

## 证据

- [分析冻结](IDEA.md)
- V3（5 cohort）[机器结果](RESULT_V3.json)，SHA-256 `d836b441a3a6b352adb143c8699a1c09d1b9e4794d240bafc76be0c40c95e348`
- V3 [图](COMPARISON_V3.png)，SHA-256 `432f06d1775bb0bd46f1e695825c701adb07f3720f8fdc3572de4bb696159f74`
- V3 [PDF](COMPARISON_V3.pdf)，SHA-256 `07b56fa6a4b2ec504dba63be1d09f420bc492955d07b23b84a203219f87a2110`
- V2（postop访问前）[机器结果](RESULT_V2.json)，SHA-256 `e8c9cac404c43c753e9130a81b8afd975b4e7036fe0309b252551f43d3522675`；[图](COMPARISON_V2.png)，SHA-256 `389ab49233e586f64c15cc69bc415995607aeddf327661796ec4ad87cac450ad`
- V1（WAW访问前）[机器结果](RESULT.json)，SHA-256 `1fd9ca3869342dc0c78ca7a14212874b4d1a0ef55e0019ab7fde3aa1a2c582a7`；[图](COMPARISON.png)，SHA-256 `430ef376ff0b4a6e2e3352e619f5c76f30ef7daac5966ffe9bf360ef99dabae6`
- internal raw SHA `29efc557aa12b6f82fd079edd84cc3c8c1e5ad4536880c016b664932234f45ad`
- Africa raw SHA `c9fb07be4412c160129037104936e4c34b6390fa733b998b8a8f35524ccd30b3`
- CPTAC raw SHA `4b9d8c90e1b5ec45b0e664e560301db20affd6a96387f416a62b561f8d57b4fd`
- WAW raw SHA `ef35301a78f6e47ac0574e5727aa7c5015b2f30cd73e1c8c45487fa8591dde91`
- WAW postop raw SHA `72269a5a872389db9d6dd316d7d46b614bad8d81796b0bb72625814907d98a98`

## V4补记（2026-09-06 Fable session）

加入第六个cohort UCSD-BraTS2024test56：Flip8−identity=`+0.002971`、CI `[-0.002740,+0.008750]`跨0、Point差0（Full identity与Flip8均56/56全中）。六cohort Scan点估计全部为正；CI全正仍是internal270/CPTAC33/WAW-pre152三处，Africa95/WAW-post31/UCSD-BraTS2024test56跨0；Point差均不为负。保存计数的事后分解在新cohort再次显示precision `+0.01111`（CI全正）、预测体素显著减少、recall近零——保守去假阳性机制在五个有保存计数的外部cohort中5/5复现。部署仍8×；不pool、不授权视图/权重/阈值搜索。V4 JSON/PNG/PDF SHA=`55aa9784…/b1871cf8…/e8c8aef7…`；见[RESULT_V4.json](RESULT_V4.json)与[COMPARISON_V4.png](COMPARISON_V4.png)。
