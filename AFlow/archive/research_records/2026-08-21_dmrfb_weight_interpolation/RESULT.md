# KD200–DMRFB216 weight interpolation validation结论

## Outcome

同架构同盆地weight interpolation完成可行性几何、预注册、10项合成测试、3个interior checkpoint生成/reload和fixed validation108。

Alpha0.5同时超过KD200和DMRFB216的Scan/Subject，并改善boundary/bottom困难分层，是近期第一条超过KD200 Dice点估计的单模型开发结果。但Point仍只有100/108，低于冻结101门，所以不访问test270。

## Method

- Alpha0：KD200装入`dmrfb_dilation2`架构，adapter保持确定性初始权重与zero projection，函数精确等于KD200。
- Alpha1：validation-selected DMRFB216。
- Interior：逐浮点state tensor做`(1-alpha)theta0+alpha theta1`，固定alpha0.25/0.5/0.75。
- 无训练、teacher、ensemble、阈值或后处理变化；部署仍一个DMRFB单模型、1×forward。

## Fixed validation108

| Alpha | Scan | Subject | Point |
|---:|---:|---:|---:|
| 0.25 | `0.734639407` | `0.724746832` | `100/108` |
| **0.50 selected** | **`0.734683010`** | **`0.724934341`** | **`100/108`** |
| 0.75 | `0.734528962` | `0.724892522` | `100/108` |
| KD200 endpoint | `0.734384308` | `0.724434143` | `101/108` |
| DMRFB216 endpoint | `0.734242477` | `0.724742362` | `100/108` |

Selected alpha0.5相对KD200：

- Scan `+0.000298702`，95% CI `[-0.000845339,+0.001413506]`；63/108改善。
- Subject `+0.000500197`，95% CI `[-0.000718836,+0.001523545]`；29/50改善。
- Point `-1/108`。

相对DMRFB216：

- Scan `+0.000440533`，95% CI `[-0.000525627,+0.001509138]`。
- Subject `+0.000191979`，95% CI `[-0.000961739,+0.001571383]`。
- Point相同100/108。

相对KD200，selected boundary-top27 mean `+0.001053475`（19/27改善），bottom27 `+0.002011388`（17/27改善）。这些是描述性证据，不参与alpha selection/promotion。

## Promotion

通过：同时超过两个端点的Scan/Subject；相对两端四个CI安全门；generation/reload/access门。

失败：Point仅100/108，低于101/108。

因此`candidate_authorized_for_test=false`，无Leaderboard更新。不能在结果后改选alpha0.25/0.75或借KD200单独出Point。

## Implementation preflight

首次evaluator preflight因checkpoint公式metadata字符串`then cast`与验证器预期长字符串不一致而在started receipt前停止；未读取validation。修复只对齐metadata合同，checkpoint/state/alpha/selection/门槛均未改变。

## Cost and deployment

- 无训练；生成3个checkpoint约55M。
- 生成器做CPU完整128³ strict-forward，耗时约数分钟；validation GPU约19秒。
- Selected checkpoint约19.1MB。
- 部署4,757,985参数、1×forward，无ensemble。

## Evidence and access

- Raw validation SHA-256 `19f65f5ae71ca21760a1bc6c2cbfe76952fcb21f86df82f4efbb0bb692e0f4ab`。
- Started receipt SHA-256 `6a2eb02d1d8f8a0eb5c741c4d062dae31f4c0a5957b6ce2541261e1fabe81feb`。
- Generation SHA-256 `45d02460a47f468373b8d15826c30b02d31a093d8cdbcacb5c9ad5ada39cc3aa`。
- Selected alpha0.5 checkpoint SHA-256 `0db67cb81ae3b8aa76449918910b110207b25233dc5b962070caab6f3939195a`。

本方法新fixed validation已访问；outer selector、confirmation和test未访问。两个endpoint具有历史validation，KD200还具有随后历史test访问；没有使用test选择alpha。

## Next decision

当前插值协议停止，不改alpha或Point来源。在当前目标下不进test。如果未来用户明确改变目标为“Dice优先、允许Point少1例”，alpha0.5可以作为明确标注的development-selected候选，但那是新的目标与证据协议。
