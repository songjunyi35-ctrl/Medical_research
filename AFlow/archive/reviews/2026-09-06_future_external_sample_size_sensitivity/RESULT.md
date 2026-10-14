# 未来外部队列样本量敏感性（规划，不是功效保证）

状态：只读Africa95/CPTAC33/WAW152逐例配对差；0模型/cohort重访/选择。

四个队列都是每人一scan，可用逐例差的sample SD做粗略独立配对设计敏感性。Africa/CPTAC/WAW-pre/WAW-post的GCSAM−Full SD=`0.1210/0.0289/0.0845/0.0417`，Flip8−Full=`0.0670/0.00742/0.0364/0.0268`；队列间变化本身就说明不能搬用单一方差。

使用双侧α=0.05、名义80% power的正态近似：

| 比较 | 预先关心的平均Dice差 | 中位SD场景所需独立subjects | 最大SD场景 |
|---|---:|---:|---:|
| GCSAM−Full | `0.005` | 1,249 | 4,595 |
| GCSAM−Full | `0.010` | 313 | 1,149 |
| GCSAM−Full | `0.020` | 79 | 288 |
| Flip8−Full | `0.005` | 314 | 1,411 |
| Flip8−Full | `0.010` | 79 | 353 |
| Flip8−Full | `0.020` | 20 | 89 |

这说明CPTAC33的小N仍能给Flip8正区间，部分因为其逐例差方差异常小；WAW-pre152的GCSAM `+0.0102`跨0与较大SD相容，WAW-post31两差均跨0则同时受小N影响。EGD374若解锁，规模可能覆盖部分`+0.01` Flip8或`+0.02` GCSAM场景，但不能保证。

限制：公式忽略非正态尾部、多重比较、中心/纵向聚类、失访、来源失败和效果跨域异质性。Internal270因reused且多scan/subject、UCSD75因75 scans/53 subjects，均不进入独立subject SD场景。未来必须在新数据前冻结临床/工程上有意义的delta，不能按本表挑最容易显著的目标。

证据：V2 [机器结果](RESULT_V2.json)，SHA-256 `0b4205edccce6b0b2c65bf7efd84a279b4713b793ee00c8808290bae02bafc70`；V1（postop访问前）[机器结果](RESULT.json)，SHA-256 `d80407983a73d7e720d0e88af45550ee097a60fbcc2e38df22e5c090201eb028`。

## V3补记（2026-09-06 Fable session）

加入UCSD-BraTS2024test56的subject级配对差SD（41独立患者；scan级无效因12人多时点）：Flip8−identity subject SD=`0.01915`（该近域下检出`+0.005/+0.01`约需116/29人）；GCSAM−Full subject SD=`0.05231`（检出`+0.01/+0.02`约需215/54人）。近域SD明显小于Africa等远域场景，但单cohort不可移植；新数据前仍须先冻结最小有意义delta。V2保留；见[RESULT_V3.json](RESULT_V3.json)。
