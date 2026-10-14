# Hard mL1-ACE fixed-validation 负结果

## Outcome

2025 Average Calibration Losses 的20-bin、foreground-only hard marginal L1 Average Calibration Error（hard mL1-ACE）完成了一手论文/代码审计、七问预注册、13项合成测试、A100 GPU smoke、32轮formal训练、完整配对审计和一次fixed validation108决策。

所有implementation、随机流配对、显存、checkpoint reload和deploy门均通过。hACE把校准误差大幅降低，Point hit也提高，但selected模型的Scan和Subject Dice显著低于matched control及KD200，两个Dice安全CI门也失败。因此**不授权、不访问本方法的新test270**。

结论是：当前`Dice:CE:ACE=1:0.1:1`、T1ce-only二元WT、KD200后32轮的适配能够让概率更可信，却不能同时守住分割重叠质量。它不否定论文在四模态多类、从头长训练、`1:1:1`设置中的总体结论。

## Method and only changed factor

两臂都从相同KD200开始，以相同seed20260860、batch8、sampler/增强、fresh AdamW、LR`1e-5`继续32轮：

- control：历史DiceCE，Dice系数1、交叉熵系数0.1；
- hACE：只在上述loss后增加`1.0×`前景hard mL1-ACE；
- 每个volume把float32 sigmoid概率分进20个hard bins，比较每个occupied bin的平均预测概率与真实前景频率；bin assignment detach，bin内概率均值保留梯度；空GT病例贡献0；
- 无teacher、无新增参数，部署仍为原始4,700,897参数locator的最终raw logit。

唯一变化是训练期hACE。起点、756例训练数据、随机流、optimizer、epoch grid、validation108、threshold0、无SAM/TTA/postprocessing和推理路径全部固定。完整201–232 case-order与CUDA RNG hashes和control逐项相同。

## Fixed validation108 / 50 subjects

hACE按预注册在208/216/224/232中只以Scan Dice最大、Subject exact tie-break选择epoch224；校准指标没有参与选点。

| Epoch | Control Scan / Subject / Point | hACE Scan / Subject / Point | Control macro / official-micro ACE | hACE macro / official-micro ACE |
|---|---|---|---|---|
| 208 | `0.733130 / 0.723298 / 0.935185` | `0.718967 / 0.708906 / 0.935185` | `0.227226 / 0.193833` | `0.123644 / 0.066548` |
| 216 | `0.732926 / 0.723036 / 0.935185` | `0.723104 / 0.712469 / 0.953704` | `0.229513 / 0.193934` | `0.124639 / 0.065905` |
| **224 selected** | `0.734413 / 0.724670 / 0.925926` | **`0.723870 / 0.713278 / 0.953704`** | `0.230865 / 0.197534` | **`0.125927 / 0.071770`** |
| 232 | `0.734134 / 0.724618 / 0.916667` | `0.722810 / 0.712917 / 0.953704` | `0.232302 / 0.199433` | `0.120889 / 0.062443` |

Selected hACE224为103/108 Point hits；same-epoch control为100/108，historical KD200为101/108。

### Selected hACE224 comparisons

| Comparison | Scan delta / paired 95% CI | Subject delta / paired 95% CI | Point delta |
|---|---|---|---:|
| hACE − same-epoch control | `-0.010543607` / `[-0.014594454,-0.006692960]` | `-0.011391898` / `[-0.017179631,-0.006042880]` | `+0.027778` |
| hACE − historical KD200 | `-0.010514543` / `[-0.014859691,-0.006402609]` | `-0.011156431` / `[-0.017323244,-0.005343740]` | `+0.018519` |

相对same-epoch control，30/108 scans和12/50 subjects改善，78 scans和38 subjects变差。相对KD200，33 scans和13 subjects改善，75 scans和37 subjects变差。

Selected macro ACE从control的`0.230865180`降到`0.125926688`，delta `-0.104938487`，paired 95% CI `[-0.112169538,-0.097648101]`；107/108病例改善，1例变差。official-micro ACE从`0.197533637`降到`0.071769588`，pooled-voxel ACE从`0.197531864`降到`0.071816333`。

## Promotion decision

通过：全部implementation/synthetic/smoke、32轮配对、reload/deploy、Point≥100/108、macro ACE至少降0.005且paired CI上界<0、official-micro ACE降低。

失败：

- selected hACE的Scan/Subject均未超过same-epoch control；
- selected hACE的Scan/Subject均未超过KD200；
- 相对KD200的Scan/Subject paired CI下界分别`-0.01486/-0.01732`，远低于`-0.003`安全门。

因此`all_gates_passed=false`、`hace_candidate_authorized_for_test=false`；没有test evaluator、test receipt或Leaderboard变更。

## Mechanism interpretation

机制确实工作：train hACE从epoch201的`0.201153`降到epoch232的`0.070369`，约下降65%；validation macro ACE约下降45%，Point增加2–3例。因此结果不是“loss没有梯度”或实现错误。

但Dice下降约0.0105，而且置信区间完全低于0。最可信的解释是当前weight1校准压力与主分割目标同量级：smoke中ACE/监督loss比为`1.095`，它强力调整全体素概率分布，却不直接奖励边界与区域重叠，最终用更可靠的置信度换走了Dice。这个解释是基于loss尺度与观察结果的推断，不证明更小权重一定成功；预注册禁止在看到validation后降权重重跑。

## Cost and deployment

- Formal训练`1354.37s`，即22.57min；峰值allocated `20.29GiB`、reserved `21.53GiB`，一张A100-SXM4-40GB。
- Smoke峰值allocated `20.23GiB`，4-step全部门通过。
- 新workspace约76M；不建新cache。
- 四个正式checkpoint各为plain deploy state；selected epoch224 SHA-256 `e773e50f6f5941cdfcf37a60e9915c76b0a86acaec29f3afa4dff9e1789b71ab`。
- 部署仍4,700,897参数、1×forward，无teacher、辅助头、SAM、TTA或后处理。

## Source/implementation boundary

官方实验是四模态、4类softmax、含background、从头长训练，Dice:CE:ACE=`1:1:1`；项目为了保持历史control只增加ACE，形成单通道sigmoid WT的`1:0.1:1` continuation。论文BraTS平均Dice保持，但官方raw WT Dice也有`0.9212→0.9190`下降，因此本次WT负结果与来源风险边界一致。

## Evidence hashes and access

- Raw validation：`workspace/BraTS/sammed3d_t1ce_wt/plain_hard_ml1ace_20260820/HARD_ML1ACE_VALIDATION.json`，SHA-256 `b40b688c5372825894280e45f2392212e5c8f011afc5591b51fbd3d77ebdd477`。
- Validation started：SHA-256 `be5abed21d60276d6241cb8e3beb41532872562964d54002c6dce9efc8a23312`。
- Archived validation pre-access receipt：SHA-256 `f030f69cfe18f48ece6dd722f4335af647e5535c76654459889c7fbc896bf3cc`。
- Formal pairing receipt：SHA-256 `1495b77597052b2552c925d87c8c391c2d1fd89a049deaed1d6b0693c956acf7`。
- hACE TRAIN_DONE：SHA-256 `1030efed43146960ca9412250f37785977aa4af2bd9d70ea6007a42ae7d4922f`。
- Implementation receipt：SHA-256 `0b6ba7b9de53e00a16116474cbd0f5610d0c863ecc6a799a893e86f9e2bf0b6a`。
- Model/trainer/evaluator SHA-256：`94ef859a...cf70ac` / `27c83b84...19e51` / `8195a3ed...7c044`。

本方法的新fixed validation108已访问；起点KD200在此前已有validation选择及随后test访问，但那些test数值没有用于选择hACE的bins、weight、epoch、threshold或后处理。本方法没有访问outer selector、confirmation或新test。

当前适配停止，不搜索bin数、right、weight、起点、LR、epoch、threshold或后处理。下一单因素候选为PM Dice难体素调制；实施前必须重新核验一手论文/实现并完成新的七问预注册。

Primary sources: [paper v4](https://arxiv.org/html/2506.03942v4), [official repository](https://github.com/cai4cai/Average-Calibration-Losses).
