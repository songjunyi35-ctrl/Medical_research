# 固定计算量972例voxel-KD结果

## 结论

数据扩容的直觉得到了明确但不完整的支持。只把可抽样训练池从756 scans / 346 subjects扩大到972 scans / 444 subjects，在相同3008个optimizer steps下，selected epoch184相对新cache-matched 756 control提高Scan/Subject `+0.003297/+0.004287`，而且两项paired 95% CI均高于0。这是近期第一条对完全匹配control有清晰正增益的数据机制证据。

但候选Scan仍比历史KD200低`0.000335`，Point为99/108、比KD200少2例，因此冻结promotion失败，不访问test270。不能在结果后增加draw数、延长epoch、改sampling weight或借KD200的Point。

## 唯一变化与固定项

- `data756_control`和`data972_candidate`都从同一个旧972缓存读取，重叠756例image/label逐体素完全相同。
- 唯一arm差异是可抽样病例池756→972及其由该population自然诱导的权重向量。
- 两臂都从historical168初始化，teacher仍是`0.75 history168 + 0.25 EMA168`，保留原DiceCE、`0.3` voxel-KL、`T=2`、seed20260846、batch8、每轮756 sampler draws、94步、32轮late-cosine、shift8和clip5。
- 输出仍是4,700,897参数plain单模型，raw-logit阈值0，1×forward，无TTA/SAM/后处理。

## Fixed validation108

| 方法 | Scan Dice | Subject Dice | Point |
|---|---:|---:|---:|
| **972 candidate epoch184** | **`0.734049264`** | **`0.724694392`** | `99/108` |
| cache-matched 756 control epoch184 | `0.730752078` | `0.720407590` | `100/108` |
| archived KD200 | `0.734384308` | `0.724434143` | `101/108` |

相对新cache-matched control：

- Scan `+0.003297185`，95% CI `[+0.001027061,+0.005548872]`，68/108 scans改善、40下降。
- Subject `+0.004286801`，95% CI `[+0.001545535,+0.007158382]`，33/50 subjects改善、17下降。
- Point `-1/108`。

相对历史KD200：

- Scan `-0.000335044`，95% CI `[-0.002641552,+0.001956023]`。
- Subject `+0.000260248`，95% CI `[-0.002678973,+0.003242736]`。
- Point `-2/108`。

四个candidate epoch的Scan依次为176 `0.732818`、184 `0.734049`、192 `0.733770`、200 `0.733600`；按预注册只以Scan、Subject、早epoch选择184。

## Promotion与科学解释

通过：同时超过same-epoch control的Scan/Subject；超过KD200 Subject；四项CI安全门；cache、loss、GPU smoke、formal、8 checkpoint reload和access门。

失败：没有超过KD200 Scan；Point只有99/108，低于101/108。

因此正确结论不是“更多数据没用”，而是：在相同计算量和同一旧cache权重口径下，扩大病例池产生了真实、广泛且CI为正的泛化收益；不过历史KD200使用原生标签体积做weighted sampling，仍有略高Scan和更稳定Point。当前实验没有隔离“更多病例”和“每病例重复次数减少”各自贡献，也不能事后把每轮draw从756改到972。

## 成本

- 两臂各约1939.5秒；两卡并行墙钟约32.3分钟。
- 每臂peak allocated `21,788,257,792 B`，约20.29GiB。
- 复用已有9.5GB cache，没有新建cache；本实验workspace约181MB。
- 部署仍4,700,897参数、1×forward，selected checkpoint约18.8MB。

## Evidence与访问

- Raw validation SHA-256 `8661509aa1dfe4e6c1eaf06ef94980863796ba1ebb12fbbf85091f8070d7b530`。
- Selected checkpoint SHA-256 `0f69e049aba20e734474be0e9675b0af374c9a50ee662eb61eb7f4f3a97a34c3`。
- Control/Candidate TRAIN_DONE SHA-256 `de56cb67149882424851a5f289fd0f92b4a24213989fef8fe7f437002b1ea3ec` / `d52a3217a23f7249e32372e4a97e8118648a138fc48b6ca940c92dca65289f6a`。
- Implementation receipt SHA-256 `212e76d211c91254963a627282e294d9d1ccc711d73453d8c39519bd635d0c48`。

本方法的新fixed validation已访问；outer selector和confirmation病例已被明确回收为训练角色，不再可用作独立评估；test270未访问。

## 下一决定

当前固定计算量协议停止，不增加draw或改采样。结果把下一条数据研究问题收窄为：原生体积感知sampling与Point稳定性可能比单纯增加病例更关键。若继续，应在train-only证据上预注册一个同时使用972病例和原生标签体积权重的独立方法，并重新跑匹配control；不能直接用本次validation选择权重。

