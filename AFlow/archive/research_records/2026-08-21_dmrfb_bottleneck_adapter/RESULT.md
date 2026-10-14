# Single-bottleneck DMRFB dilation fixed-validation结论

## Outcome

MICCAI 2025 DMRFB的single-bottleneck、same-parameter dilation实验完成了NMSW/native-patch可行性淘汰、一手论文/官方block审计、七问预注册、11项合成测试、双卡identity/two-step-gradient smoke、32轮paired formal和fixed validation108决策。

Candidate超过same-parameter control的Scan/Subject，并改善预指定boundary-top27和KD200-bottom27；这是近期第一条四项机制方向都为正的方法。但candidate Scan仍略低于KD200，Point少1例，所以完整promotion失败，不访问test270。

## Method and only changed factor

两臂都在KD200最深`128@16³`特征后增加相同57,088参数DMRFB residual adapter：三条depthwise branch→concat384→GN128组→GELU→zero-init 1³ projection→identity residual。

- Control有效感受野`[1,3,3]`。
- Candidate有效感受野`[1,3,5]`，唯一变化为第三支`dilation/padding 1/1→2/2`。
- 两臂adapter tensor初值完全相同；epoch0 FP32/CUDA-BF16 logits与plain KD200精确相同。
- Base group LR/wd=`1e-5/1e-5`；adapter group=`1e-3/0.08`；两臂相同。
- Loss、seed20260900、sampler、增强、32轮、grid和推理均相同；无teacher/辅助loss/第二模型。

## Fixed validation108 / 50 subjects

Candidate仅按Scan选择epoch216。

| Epoch | Same-RF control Scan / Subject / Point | DMRFB candidate Scan / Subject / Point |
|---|---|---|
| 208 | `0.732962 / 0.723187 / 0.935185` | `0.733007 / 0.723184 / 0.935185` |
| **216 selected** | `0.733997 / 0.724596 / 0.935185` | **`0.734242 / 0.724742 / 0.925926`** |
| 224 | `0.733355 / 0.723528 / 0.944444` | `0.733652 / 0.723908 / 0.935185` |
| 232 | `0.733389 / 0.723322 / 0.935185` | `0.733541 / 0.723613 / 0.925926` |

Selected candidate相对same-epoch control：

- Scan `+0.000245164`，95% CI `[-0.000146184,+0.000668318]`；62/108改善、46变差。
- Subject `+0.000146359`，95% CI `[-0.000376769,+0.000693317]`；26/50改善、24变差。
- Point `-1/108`。

相对KD200：

- Scan `-0.000141831`，95% CI `[-0.002369948,+0.001881431]`；未严格超过KD200。
- Subject `+0.000308219`，95% CI `[-0.002291890,+0.002539568]`。
- Point `-1/108`，100/108，低于101门。

## Hard-stratum mechanism

- Boundary-top27 candidate-control mean `+0.000107092`，12改善、15变差，95% CI `[-0.000694455,+0.001093573]`。
- KD200-bottom27 mean `+0.001184257`，19改善、8变差，95% CI `[-0.000149361,+0.002622229]`。

两项方向均为正，尤其bottom27有19/27改善，支持“扩大最深感受野帮助困难尾部”的机制，但区间仍跨0，证据强度有限。

## Promotion decision

通过：超过same-parameter control的Scan/Subject；超过KD200 Subject；四个CI安全；boundary/bottom机制门；全部implementation/identity/two-step-gradient/pairing/reload/memory门。

失败：

- Scan `0.734242477`未严格超过KD200 `0.734384308`；
- Point仅100/108，低于101/108。

因此`all_gates_passed=false`、`candidate_authorized_for_test=false`；无Leaderboard变更。

## Partial validation interface failure

首次validation在第一批forward后因output缺少可选`center_logits=None`属性停止，没有写raw report、没有汇总指标或选epoch。修复只增加只读兼容property，state_dict/logit/checkpoint/算法不变；2项interface/roundtrip回归通过后，以新V2文件重新执行同一冻结评估。partial access与修订均已归档。

## Cost and deployment

- Control/candidate formal `1113.12/1114.83s`，双卡墙钟18.58min。
- Peak allocated每臂`21,904,692,224` bytes，约20.40GiB。
- Workspace约146M；无新cache。
- 每模型4,757,985参数，比plain多1.2%；单模型1×forward，保留adapter。
- 8个modified checkpoint reload/deploy差均0。

## Evidence and access

- Raw V2 validation SHA-256 `d164c6818fcd74fefc7d542c33edd9f0883b59281ac5ec192cac3b00ef516a88`。
- Started receipts SHA-256 `8bb058be0d58f8e847f28bea153936e92c08106eca1d0069809222775b955cf6`。
- Pairing receipt SHA-256 `e0670edaa788b88ab5abe943e398e52a02c2e0512ca2f58cbee580ad09f61c00`。
- Control/candidate TRAIN_DONE SHA-256 `f9a8d198...7d60a2` / `67bfbc40...952fff`。
- Selected candidate216 checkpoint SHA-256 `df53d9bd61f1489746ae1240fffba1f8a3897d0de1416122a389e1d420def684`。

本方法新fixed validation已访问；partial失败和V2重跑均明确记录。起点KD200有历史validation及随后test访问。本方法未访问outer selector、confirmation或新test。

当前single-block dilation协议停止，不调参。下一条若继续，应先审同架构同盆地weight interpolation能否在单模型内折中KD200的Scan/Point与DMRFB的Subject/tail收益；必须重新预注册alpha grid且不看test。

Primary sources: [MICCAI 2025 paper](https://papers.miccai.org/miccai-2025/paper/4895_paper.pdf), [official code](https://github.com/SLDGroup/EfficientMedNeXt).
