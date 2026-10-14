# VM Dice fixed-validation负结果

## Outcome

2025/2026 Voxel-wise Modulated Dice完成了validation108错误分层、一手公式核验、七问预注册、19项合成测试、双卡paired smoke、gamma0/gamma1双臂32轮formal训练、完整配对审计和一次fixed validation108决策。

全部implementation、pairing、24GiB显存、8个checkpoint reload/deploy门通过。gamma1确实改善了预指定的高边界27例，也把Point保持到103/108；但selected模型没有超过同轮gamma0或KD200的Scan/Subject，KD200-bottom27反而变差。因此不授权、不访问test270。

## Method and only changed factor

两臂都显式构造foreground/background两类，使用论文平方分母、epsilon`1e-6`并保留`0.1×BCEWithLogits`：

- gamma0：modulation显式为1，即paper-Dice causal control；
- gamma1：`m=|y-stopgrad(p)|`，即论文默认VM Dice；
- 唯一变化为`gamma 0→1`；起点、seed20260870、batch8、sampler/增强、fresh AdamW、LR`1e-5`、32轮、grid、validation和推理全部相同；
- 无teacher，部署仍plain final raw logit、4,700,897参数、1×forward。

## Fixed validation108 / 50 subjects

gamma1仅按Scan Dice选择epoch224。

| Epoch | gamma0 Scan / Subject / Point | gamma1 Scan / Subject / Point |
|---|---|---|
| 208 | `0.731299 / 0.720756 / 0.953704` | `0.730685 / 0.720458 / 0.953704` |
| 216 | `0.731724 / 0.721232 / 0.953704` | `0.731059 / 0.720830 / 0.953704` |
| **224 selected** | `0.732818 / 0.722534 / 0.953704` | **`0.732178 / 0.721938 / 0.953704`** |
| 232 | `0.732428 / 0.721731 / 0.953704` | `0.732019 / 0.721597 / 0.953704` |

Selected gamma1相对same-epoch gamma0：

- Scan `-0.000640561`，95% CI `[-0.001979059,+0.000667640]`；45/108改善、63变差。
- Subject `-0.000596242`，95% CI `[-0.002381061,+0.001346782]`；22/50改善、28变差。
- Point相同，均103/108。

相对historical KD200：

- Scan `-0.002206624`，95% CI `[-0.004480934,+0.000249591]`；36/108改善、72变差。
- Subject `-0.002496407`，95% CI `[-0.005444867,+0.000668027]`；14/50改善、36变差。
- Point `+2/108`。

## Hard-tail mechanism

- Boundary-proxy top27：gamma1−gamma0 mean `+0.000902898`，12改善、15变差，95% CI `[-0.002152712,+0.004341220]`。
- KD200 Dice bottom27：mean `-0.001494994`，10改善、17变差，95% CI `[-0.005889199,+0.002764157]`。
- 两组来自同一个reused validation且交集13例，不是独立confirmation。

因此“对高边界组可能有帮助”得到弱支持，但“修复整体困难尾部”不支持；机制门只过一半。

## Promotion decision

通过：全部前置门、Point≥101/108、相对gamma0的Scan/Subject CI安全、boundary-top27正delta。

失败：

- gamma1未超过same-epoch gamma0的Scan或Subject；
- gamma1未超过KD200的Scan或Subject；
- 相对KD200的Scan/Subject CI下界低于`-0.003`；
- KD200-bottom27 delta为负。

所以`all_gates_passed=false`、`vm_dice_candidate_authorized_for_test=false`，Leaderboard不更新。

## Mechanism interpretation

VM loss从epoch201的`0.885725`降至epoch232的`0.297783`，约下降66.4%，不是loss没工作。它对高边界组产生了方向正确但不稳定的小增益，却牺牲更多普通及bottom-tail病例。

gamma1初期loss约为gamma0的8.5倍，formal首轮最大grad norm `6.53`并触发clip5，而gamma0最大约`0.63`。这支持“默认gamma1对强KD basin施加了很强的新梯度方向”的解释，但不能据此事后降低gamma。另一个事实是gamma0 source-Dice本身也低于KD200，说明foreground+background平方分母continuation不如历史MONAI Dice轨迹。

## Smoke implementation failure

首次gamma1 smoke在反传/optimizer前因smoke-only `torch.quantile`无法处理33,554,432元素而停止。诊断改为数学目的等价的exact min/max后，两臂全部重跑并通过；训练loss、公式和正式协议未改，未访问validation/test。修订见`AMENDMENT_003_SMOKE_DIAGNOSTIC_FIX.json`。

## Cost and deployment

- gamma0/gamma1 formal `1121.52/1123.44s`；两卡并行墙钟18.72min。
- 两臂peak allocated均约`20.48GiB`，低于24GiB门。
- workspace约145M，无新cache。
- 8个checkpoint reload/deploy差均0。
- 推理仍4.70M参数、1×forward，无loss/teacher/辅助头。

## Evidence and access

- Raw validation SHA-256 `3750b46cd23aa07942ff0e7420f1a7f41cc91f114140c9e663fcc1915fa060c0`。
- Validation started SHA-256 `047d2d3cb53e5db7315820e65df76226a4a3a2a74212a293d7579c4dd975de48`。
- Pairing receipt SHA-256 `42490c63a365e952729328f57e50c10bcffc5c83170799939e09380fbb62d4b4`。
- gamma0/gamma1 TRAIN_DONE SHA-256 `e0162726...0cc21` / `36ab2ba1...6b30`。
- Selected gamma1 epoch224 checkpoint SHA-256 `15b9c20b8239b82c32180ae4e2cd97fe36bd051711191ff86730d6907d11a017`。

本方法的新fixed validation108已访问；起点KD200具有历史validation及随后test访问。本方法未访问outer selector、confirmation或新test，也未使用test选择gamma、epoch、LR或阈值。

当前VM Dice适配停止，不调参。若继续tail方向，下一候选为case-level FairDRO；实施前必须重新核验一手论文/官方实现和七问预注册。

Primary source: [PM/VM Dice paper](https://arxiv.org/pdf/2506.15744).
