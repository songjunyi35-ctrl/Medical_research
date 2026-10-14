# 六cohort冻结证据矩阵

状态：七份冻结raw报告只读汇总；覆盖656 scans，0新模型forward、0cohort重访、0选择/调参。不同cohort不合并排名。

## 一眼看清“最好”的作用域

| cohort | 证据身份 | Dice点估计最高 | 最高Point | GCSAM−Full identity Scan（95% CI） | Flip8−Full identity Scan（95% CI） |
|---|---|---|---|---|---|
| internal270/119 subjects | reused internal，非盲/非外部 | Full Flip8 `0.757270`（8×） | Full Flip8 265 | `+0.000244` `[-0.005655,+0.005740]` | `+0.005510` `[+0.002751,+0.008022]` |
| UCSD75/53 | 同已见cohort official-float来源纠正 | GCSAM `0.755890`（1×） | Full/GCSAM同70 | `+0.009634` `[-0.002163,+0.025053]` | 未运行 |
| Africa95 | public post-challenge四中心 | GCSAM `0.591619`（1×） | Full Flip8 88 | `+0.068617` `[+0.044472,+0.092498]` | `+0.003435` `[-0.010536,+0.016815]` |
| CPTAC33 | public Training external-to-project | Full Flip8 `0.844577`（8×） | 三者同33 | `+0.001954` `[-0.006646,+0.012384]` | `+0.003246` `[+0.000828,+0.005855]` |
| WAW152 | public Warsaw preop | GCSAM `0.693601`（1×，仅高Flip8 `0.000107`） | Full Flip8 143 | `+0.010223` `[-0.003005,+0.023481]` | `+0.010116` `[+0.004472,+0.016274]` |
| WAW early-postop31 | preop患者零重叠、day0–22 | Full Flip8 `0.598959`（8×） | Full/Flip8同15 | `+0.001657` `[-0.012498,+0.016205]` | `+0.007312` `[-0.001925,+0.016626]` |

## 稳定结论

- 没有全域、Dice/Point/成本统一冠军。内部与CPTAC总体Dice由Full Flip8领先；Africa与WAW由GCSAM点估计领先；UCSD也偏GCSAM，但不同cohort绝不能横向按绝对数值排名。
- GCSAM相对Full identity只有Africa95的Scan配对区间整体为正；internal/UCSD/CPTAC/WAW-pre/WAW-post全跨0。它的Africa优势是真实队列内证据，但不是单调规律或训练配方因果。
- 固定Flip8在五个已运行cohort的Scan点估计均正；internal/CPTAC/WAW-pre区间全正，Africa/WAW-post跨0，Point差均不负。方向更一致，但每例8次locator forward。
- WAW-postop的Point约45–48%，16/17个Full/GCSAM miss中13/14落GT切除腔；这是早期术后特异失败，不授权GT-RC避让。
- 证据等级仍有限：internal已复用；UCSD official是同75例来源纠正；Africa/CPTAC/WAW均public非hidden blind且无patient-ID桥。训练checkpoint还继承BraTS2024 Kaggle mirror与官方Synapse parity未证明的共同限制。
- 656只是各cohort scans数之和，不是可pool样本量。阶段、标签、医院、scanner、grid与预处理共同变化，不能做总体显著性或“域距离”回归。

## 当前部署选择如何说

- 若只允许1×：内部Scan点估计最高是GCSAM，内部Subject/Point是Full identity；UCSD/Africa/WAW Dice也偏GCSAM，但除Africa外差值不确定。
- 若允许8×且目标重视区域Dice/Point：Full Flip8是internal/CPTAC总体点估计最高，并在WAW最高Point；Africa最高Point也是Flip8但Dice仍远低于GCSAM。必须按目标/cohort说明，不能称通用最佳。
- 任何新选择都需要真正未见的新队列；已揭示六cohort不得用于挑view、权重、阈值、checkpoint或router。

## 证据

- [V2机器矩阵](EVIDENCE_MATRIX_V2.json)，SHA-256 `01912a2ce6cd816a8243f9b96873d4e8b03963ef9eb03eafe6d2cfbac4760816`
- [V1（postop访问前）](EVIDENCE_MATRIX.json)，SHA-256 `b6f5bc185f2e18fe6ba2165ece5733f185e226d9d5baf94df25862dc787c9097`
- [GCSAM六cohort图](../2026-09-06_cross_cohort_checkpoint_synthesis/COMPARISON_V4.png)
- [Flip8五cohort图](../2026-09-06_flip8_cross_cohort_synthesis/COMPARISON_V3.png)
- Canonical [Leaderboard](../../../AFLOW_BRATS_LEADERBOARD.md)

## V3补记（2026-09-06 Fable session）

加入UCSD-BraTS2024test56后为七cohort/712 scans：[EVIDENCE_MATRIX_V3.json](EVIDENCE_MATRIX_V3.json)。GCSAM−Full唯一CI全正仍是Africa95，**新增唯一CI全负的UCSD-BraTS2024test56（`−0.012312`）**，其余五处跨0——方向依cohort可逆。Flip8六处点估计全正、三处CI全正、Point差均不负、8×成本。仍无全域统一冠军；不pool、不回流选择。V2保留。
