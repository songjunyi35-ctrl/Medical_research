# 覆盖优先的小病灶回放蒸馏结果

## 结论

“先增加病例覆盖，再保留少量小病灶回放”的局部机制得到支持，但完整晋级目标失败。

候选只按冻结的fixed validation108 Scan Dice选择global epoch200，Scan/Subject/Point为`0.734158/0.724950/98/108`。它相对same-epoch原生体积加权replacement control提高Scan/Subject `+0.001481/+0.001890`，说明在相同972病例、752有效draw和94步预算下，减少重复、强制相邻两轮覆盖全部病例确实改善了Dice。

但候选相对archived KD200的Scan仍低`0.000226`，Point少3例；candidate-vs-KD200 Scan 95% CI下界`-0.003098`也略低于冻结安全门`-0.003`。因此`candidate_authorized_for_test=false`，没有访问reused internal test270，排行榜不变。

## 方法与唯一变化

- 输入：972 scans / 444 subjects的单通道T1ce，冻结N4、image-only brain crop和`128^3` cache。
- Student/teacher/loss：history168 student；`0.75 history168 + 0.25 EMA168` raw-logit teacher；Dice+`0.1`交叉熵+`0.3*T^2`逐体素KL，`T=2`。
- Control：每轮按原生WT体积权重带放回请求756次，batch8/drop-last实际752 draws / 94 steps。
- Candidate：首轮随机覆盖680例；以后每轮纳入上轮遗漏292例并从上轮已见病例保留388例，再消费72个原生体积加权小病灶replay，末4个replay由drop-last丢弃。
- 两臂唯一差异是sampler policy。seed20260884、数据、cache、模型、teacher、loss、增强、AdamW、late-cosine、32轮、checkpoint grid、threshold0、无TTA/SAM/postprocess和`1×`部署全部固定。

## 工程过程与失败边界

所有问题都发生在正式validation前并透明保留：

1. 首次合成测试12/14：原滚动队列不能保证任意相邻两轮全覆盖，一对只有880/972；Amendment001改为上轮互补覆盖，随后通过。
2. 首次remote smoke在模型构建前被历史runner写死的seed20260846拦截；Amendment002增加只读校验兼容层，实际训练始终用20260884。
3. V2/V3 smoke计算门均通过，但证据审计发现顶层sampling contract缺失及旧writer覆盖新文档哈希；Amendment003/004只修报告合同，V4重跑通过。
4. 首次validation evaluator CPU合同测试1/4通过；fresh-import旧常量和猜错cache目录导致3项接线失败。Amendment005固定formal文档表并按index SHA定位权威validation cache，最终5/5通过。

上述失败均未使用新validation/outer/confirmation/test指标。最终18项sampler测试、5项validation合同测试、双卡V4 smoke、双卡formal、32轮candidate sampler审计、8 checkpoint文件/SHA/model-state和reload门全部通过。

## Fixed validation108 / 50 subjects

| global epoch | Candidate Scan | Candidate Subject | Candidate Point | Control Scan | Control Subject | Control Point |
|---:|---:|---:|---:|---:|---:|---:|
| 176 | `0.725897` | `0.717699` | 99/108 | `0.731176` | `0.721248` | 100/108 |
| 184 | `0.731671` | `0.723693` | 99/108 | `0.732927` | `0.724079` | 100/108 |
| 192 | `0.733052` | `0.723727` | 99/108 | `0.733350` | `0.723729` | 101/108 |
| **200 selected** | **`0.734158`** | **`0.724950`** | **98/108** | `0.732678` | `0.723060` | 99/108 |

选择严格使用candidate Scan→Subject exact tie→早epoch，不用Point或control选epoch。

Selected candidate相对same-epoch control：

- Scan `+0.001480713`，paired 95% CI `[-0.000208634,+0.003246221]`；65/108 scans改善。
- Subject `+0.001890207`，paired 95% CI `[-0.000525115,+0.004555629]`；29/50 subjects改善。
- Point `-1/108`。

Selected candidate相对archived KD200 `0.734384/0.724434/101/108`：

- Scan `-0.000226018`，paired 95% CI `[-0.003098169,+0.002469755]`；68/108 scans改善。
- Subject `+0.000515730`，paired 95% CI `[-0.002651706,+0.003490821]`；29/50 subjects改善。
- Point `-3/108`。

置信区间均为冻结seed下10,000次paired percentile bootstrap；点估计领先不宣称显著性。

## Promotion gates

通过：超过same-epoch control Scan/Subject；超过KD200 Subject；相对control的两项CI安全门；相对KD200 Subject CI安全门；所有实现、formal、hash、reload、memory、time和access门。

失败：没有超过KD200 Scan；Point仅98/108、低于101；相对KD200 Scan CI下界`-0.003098<-0.003`。

因此不访问test，不调680/72比例、seed、epoch、LR、weight、threshold或postprocess。

## 直觉是否支持

部分支持。Coverage-first相对同计算量replacement control在epoch200带来Dice双提升，说明此前972池的带放回重复确实浪费了一部分多样性；但候选早期三个epoch都低于control，收益出现得晚且Point持续更差。原生体积回放只占9.57%有效draw，没能保住小病灶自动point。这个结果否定当前`680 coverage + 72 replay`、seed20260884、32轮实现的完整晋级主张，不证明所有覆盖采样或更多训练预算无效。

## 成本

- Formal：control/candidate `1935.0s/1938.5s`，两卡并行墙钟约32.3min。
- 峰值allocated：两臂均`21,788,257,792 B`，约20.29GiB/A100。
- Workspace：约253MB；不建新大cache，复用9.5GB train cache和validation108 cache。
- 推理：4,700,897参数、`1×` locator forward；无新增部署复杂度。

## Evidence与访问

- Raw validation SHA-256 `9672f8e25b443ecc7ba88895f96c3f2f51b1d31d2cf8486fa94cf47d4602058e`。
- Validation started SHA-256 `c4d3aab74ad03d42cfb6c8f2ef4b8be8974862e24af553ab41fa50c3478ac62c`。
- Pre-access receipt SHA-256 `a45cf8021db6cd97dca423418bef9b7051e0ae3b797a69bf71d454fd897b1ed3`。
- Implementation receipt SHA-256 `17535ed9fef216ad2fc15357b4b7f027cc558363e380665700cf5ef766ec26ae`。
- Control/candidate TRAIN_DONE SHA-256 `ffc51f7288defccfcb7c09213fc41b70e54e75662c6b13edffac65f6e9da0c8a` / `0887859fe99aa77b4b2740c621cdc30b444576ee9de94803b599a73efe735579`。
- Selected candidate checkpoint SHA-256 `4a02e9ba0beb704a3b1a6429394df39bff5a706f418c12c10436230f7f658ae4`。
- Validation cache index SHA-256 `9339e4575561447dd5bd99bc29e909ae1e9f566721d351a244d087a40622f322`。

本方法的新fixed validation已访问；outer/confirmation未访问且已作为train角色回收；test270未访问。

## 下一决定

停止当前coverage/replay配方，不搜索比例。若继续数据轴，仍可用独立train-only动机预注册“972 effective draws/epoch”的数据+计算预算实验，并用两臂matched compute隔离预算；不能用本次validation选择其sampler、比例、epoch或阈值。
