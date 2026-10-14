# 972病例配额一致voxel-KD结果

## 结论

用户提出的核心判断得到明确但不完整的支持：972病例确实需要相称训练预算。只把每轮有效draw从752提高到968、optimizer steps从94提高到121，selected candidate200相对same-epoch budget752 control提高Scan/Subject `+0.000779/+0.000954`，Point持平100/108。

但candidate Scan `0.734259`仍比archived KD200 `0.734384`低`0.000125`，Point也少1例；因此冻结promotion失败，不访问test270。不能因为只差极小数值而事后延长epoch、改draw或放宽Point门。

## 唯一变化与固定项

- 两臂均为972 scans / 444 subjects、同一cache、同一原生WT体积权重、同一history168 student和history/EMA voxel-KD teacher。
- Control：756 requested / 752 consumed / 94 steps per epoch，32轮共3008步。
- Candidate：972 requested / 968 consumed / 121 steps per epoch，32轮共3872步。
- 唯一变化是每轮训练预算`+28.72%`。seed20260924、batch8、增强、AdamW/LR、loss、checkpoint grid、threshold0和1×推理全部固定。

## Fixed validation108 / 50 subjects

| epoch | Candidate Scan | Candidate Subject | Candidate Point | Control Scan | Control Subject | Control Point |
|---:|---:|---:|---:|---:|---:|---:|
| 176 | `0.733130` | `0.723170` | 101/108 | `0.734142` | `0.725468` | 100/108 |
| 184 | `0.732592` | `0.722970` | 100/108 | `0.732517` | `0.723932` | 101/108 |
| 192 | `0.733065` | `0.723078` | 99/108 | `0.732220` | `0.722257` | 101/108 |
| **200 selected** | **`0.734259`** | **`0.724956`** | **100/108** | `0.733480` | `0.724003` | 100/108 |

Candidate只按Scan→Subject exact tie→早epoch选择200。

相对same-epoch control：Scan `+0.000779220`，95% CI `[-0.000818007,+0.002474315]`；Subject `+0.000953659`，95% CI `[-0.000944511,+0.002890412]`；Point持平。54/108 scans改善、54下降。

相对archived KD200：Scan `-0.000125370`，95% CI `[-0.002025240,+0.001765068]`；Subject `+0.000522177`，95% CI `[-0.001710577,+0.002939398]`；Point `-1/108`。

## Promotion与解释

通过：超过same-epoch control Scan/Subject；超过KD200 Subject；四项CI安全门；所有工程/hash/reload/time/memory/access门。

失败：没有严格超过KD200 Scan；Point只有100/108，低于101门。因此`candidate_authorized_for_test=false`。

直觉部分支持：更多训练预算能把972病例的多样性转化成更高Dice，并把Subject推到`0.724956`，是当前development单模型最高Subject点估计；但提升很小且未恢复KD200的Point。当前结果只否定`968 effective draws × 32 epochs`完整晋级，不证明972病例无法超过KD200。

## 成本

- Control/candidate formal `1923.3s/2477.5s`；双卡并行墙钟约41.3min。
- 每卡peak allocated `21,788,257,792 B`，约20.29GiB。
- Workspace约181MB；无新cache。
- 推理仍4,700,897参数、1×forward。

## Evidence与访问

- Raw validation SHA `d880644cb7e537be96be092a755e978e2159c479b1efe9305d647929475090c2`。
- Pre-access receipt SHA `4b3fc5e55f6243b146052ceeabe0fd10dd7eaad18700cf16ab846d296c2603e0`。
- Implementation receipt SHA `efa55f68b53b4ba9dc6c1af62336c5c7bca46931f9942b025176fa39ab149416`。
- Control/candidate TRAIN_DONE SHA `b45cba3cf5ac8c4d9eb394cf34f6718abfb34cae3bd2fdc94924643f2d58d103` / `1e66ca4fa428684b004bddb422c2642fcc3f5a09e1a0d7488db44377ef7c8c2a`。
- Selected checkpoint SHA `543e8cfbd80c63fec011453bd91a26beb20fc5a0486640e7846cf877de3a5fcb`。

本方法的新fixed validation已访问；outer/confirmation未访问且已回收为train；test270未访问。

## 下一决定

停止当前预算协议，不事后增加轮数。结果把问题收窄到Point稳定性与很小的Scan差距；任何进一步长时程、不同draw或checkpoint选择必须是独立新预注册，不能用本validation直接调。
