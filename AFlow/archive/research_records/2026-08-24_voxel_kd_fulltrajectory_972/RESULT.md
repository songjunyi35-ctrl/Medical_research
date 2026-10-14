# 972病例随机初始化完整voxel-KD轨迹结果

## 结论

核心直觉得到强支持：让新增216例从随机初始化第一轮就参与，而不是在history168之后补32轮，产生了当前最高的fixed validation108单模型点估计。

Full972 candidate epoch168 Scan/Subject/Point=`0.740013/0.731266/103/108`，同时超过same-seed、same-compute full756 control的`0.729070/0.723284/104/108`和archived KD200的`0.734384/0.724434/101/108`。

但冻结promotion仍失败一门：candidate-vs-KD200 Subject paired 95% CI下界为`-0.004484<-0.003`。因此candidate是新的development champion，但`candidate_authorized_for_test=false`，没有访问reused internal test270。

## 方法与唯一变化

阶段A使用train-only公共inner：control fit648、candidate fit864，共用108 scans / 49 subjects inner tune。两臂随机初始state、968 effective draws/121 steps、teacher、loss和200轮schedule完全相同。Candidate在post100选择epoch168；相对同epochcontrol Scan/Subject `+0.010026/+0.009460`，inner门通过。

阶段B用新seed20260932从随机初始化分别训练full756/full972到同一epoch168。两臂均每轮请求972、消费968、121步；唯一变化是eligible病例池`756→972`。模型、初始state、history/EMA teacher、DiceCE+voxel-KL、LR、增强、阈值0、无TTA/SAM/postprocess和1×推理固定。

## Fixed validation108 / 50 subjects

| 方法 | Scan Dice | Subject Dice | Point |
|---|---:|---:|---:|
| **Full972 random-init KD epoch168** | **`0.740012535`** | **`0.731266217`** | **103/108** |
| Full756 matched control epoch168 | `0.729069746` | `0.723284016` | 104/108 |
| Archived continuation KD200 | `0.734384308` | `0.724434143` | 101/108 |

相对full756 control：Scan `+0.010942789`，95% CI `[+0.002929710,+0.020136154]`；Subject `+0.007982201`，95% CI `[-0.002698040,+0.019661437]`；Point `-1/108`。62/108 scans改善。

相对archived KD200：Scan `+0.005628227`，95% CI `[-0.002265812,+0.014124843]`；Subject `+0.006832074`，95% CI `[-0.004484495,+0.020256842]`；Point `+2/108`。60/108 scans改善。

CI为冻结seed下10,000次paired percentile bootstrap；点估计领先不宣称统计显著。

## Promotion

通过：超过full756 control与KD200的Scan/Subject；Point≥101；相对control两项CI安全门；相对KD200 Scan CI安全门；全部工程门。

失败：相对KD200 Subject CI下界`-0.004484`低于冻结`-0.003`门。故不访问test，不因结果漂亮放宽门。

## 直觉是否支持

强支持。新增病例从训练早期参与比late continuation更有效：full972相对matched full756的Scan CI整体为正，且同时提高Scan/Subject并保住103 Point。当前失败的是subject层不确定性安全门，不是点估计或机制失败。

## 成本与访问

Inner约4.42h、final约3.61h，两卡总墙钟约8h；每卡peak20.29GiB。推理仍4,700,897参数、1×forward。Train-only inner与fixed validation108已访问；outer/confirmation/test270未访问。

Raw validation SHA `30cd65c78ad1ed2a2b5748652f48ec05cde3b54468b44a66eba77d062d3ce129`；pre-access receipt SHA `b06724284b7ccad49fe759844fa4d787c0b22e788460c697bdf21bbaf5d77c67`；selected checkpoint SHA `f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b`。

## 下一决定

保留为development champion。不在当前validation调CI门、epoch、loss或threshold；若要test授权，需要独立subject-level确认或显式改变科研目标。

## 用户授权failed-gate test270

用户在得知validation Subject CI门失败后，明确授权对冻结checkpoint运行一次探索性reused internal test270。该结果不是pristine blind、外部确认或正式promotion，且禁止触发调参。

Full972 Scan/Subject/Point=`0.751760/0.737327/262/270`，成为当前内部test总体和单模型最高点估计。相对旧ensemble为`+0.006705/+0.009982/+4 points`，Scan/Subject 95% CI分别`[+0.000784,+0.013006]`/`[+0.001524,+0.019404]`；相对KD200为`+0.008783/+0.012149/+4 points`，CI分别`[+0.002994,+0.015118]`/`[+0.003561,+0.021742]`。

Test报告SHA `9827a75b88dc9239902c7e473a7d75e2d6058f4f515bfd2b4d93003ba7bd34cd`。Test已访问；仍标记failed-gate exploratory，不据此调epoch/loss/threshold/postprocess。
