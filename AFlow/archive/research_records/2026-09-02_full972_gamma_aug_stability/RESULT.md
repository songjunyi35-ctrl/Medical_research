# Full972 保统计Gamma强度增强结果

## 结论

保均值/总体标准差的随机Gamma在fresh train-only inner上相对配对Control取得`+0.006151/+0.005570/+1 Point`并通过六门，但在fresh-seed Full972 final的fixed validation上反向为`-0.003848/-0.002312/0`，且低于archived Full972 seed32`0.011011/0.010695/2 Point`。最终11门只通过工程配对与Point不低于Control两项，`candidate_authorized_for_test=false`；没有访问test270或调Gamma。

通俗地说：在第一套新题上，随机改变MRI明暗曲线看起来很有帮助；换一个随机初始模型并用全部972例重训后，优势消失且反向。它不是稳定可复现的提升。

## 方法与唯一变化

Control使用原flip/shift/scale/offset/Gaussian-noise增强。Candidate在完全相同原增强后，用独立CUDA随机流对约30%病例做非反相Gamma`0.7–1.5`，再匹配Gamma前均值与总体标准差。标签、global RNG、模型、teacher、DiceCE+voxel-KL、sampler、AdamW、late-cosine、epoch168、阈值0和1×部署固定。

Stage A为864 fit/392 subjects与108 inner/52 subjects；Stage B为full972/444 subjects，fresh seed20261082。每阶段两臂初始state、全部168轮sampler和首个pre-Gamma原增强batch exact。

## Stage A fresh train-only inner

| 方法 | Scan Dice | Subject Dice | Point |
|---|---:|---:|---:|
| Control | `0.702204724` | `0.704525663` | 100/108 |
| Gamma | **`0.708356196`** | **`0.710095986`** | **101/108** |
| Gamma−Control | `+0.006151471` | `+0.005570323` | `+1` |

Scan/Subject 95% CI为`[-0.001553,+0.014807]`/`[-0.001596,+0.013453]`；48/108 scans改善、60变差；25/52 subjects改善、27变差。点估计三项同向且CI下界过`-0.003`安全门，Stage A六门全过。

## Stage B fresh Full972 fixed validation

| 方法 | Scan Dice | Subject Dice | Point |
|---|---:|---:|---:|
| fresh matched Control | `0.732849598` | `0.722882354` | 101/108 |
| fresh Gamma | `0.729001313` | `0.720570825` | 101/108 |
| Gamma−Control | `-0.003848284` | `-0.002311528` | `0` |
| archived Full972 seed32 | **`0.740012535`** | **`0.731266217`** | **103/108** |
| Gamma−archived | `-0.011011222` | `-0.010695392` | `-2` |

- 相对Control Scan/Subject CI：`[-0.010318,+0.002356]`/`[-0.012517,+0.008107]`；49/108 scans改善、59变差；23/50 subjects改善、27变差。
- 相对archived CI：`[-0.019562,-0.003676]`/`[-0.020360,-0.002526]`，两项整体低于0；45/108 scans改善、63变差。
- 最终11门过2失9：只过工程配对与Point不低于Control；不进test。

## 工程、成本与解释

Stage A Candidate 49151/162624 draws应用Gamma，比例0.302237；Stage B为48751/162624、比例0.299777。两个阶段Gamma范围均覆盖约0.7000–1.5000，均值/总体标准差最大误差<`4e-7`，label/global RNG不变。所有训练/reload/deploy/finite/memory/time门通过。

Stage A Control/Candidate约`12212.47/12176.91s`；Stage B约`12223.19/12189.59s`。Formal总约13.56 GPU-hours，两阶段墙钟约6.79h；peak均约20.30GiB；workspace289M，无新cache；推理仍4.70M、1×。

原直觉只在Stage A得到split/seed特异的支持，fresh final明确不支持稳定泛化。Stage A虽然均值提高，但改善病例少于变差病例，正向由少数大收益驱动；跨seed后这些收益没有重现。当前只否定固定`p0.3/range0.7–1.5/retain-stats/after-existing-augmentation`实现，不证明所有MRI强度增强不可能；但不得在已揭示cohort搜索设置。

## Evidence与访问

- IDEA/PREREG SHA：`a0935b570590c14211b44457d3b41664c21dd45b14732b861f57b84d9dd0f9fb`/`061238cfc064a9ff57c9b4134e68ebf3849ca9ba9a8b392a014c0cc5e4c67b13`。
- Stage A raw/decision SHA：`14aee9fd3bf9d9e0d610ca2f2bc1d3da1dc9104b0720b9be439b856b75daca0d`/`97bfd6b405737168debf327bbba576a29886600e224e23a7cba1b35df9fe210b`。
- Stage B validation/preaccess SHA：`5551adf2b1ede8572b4bd7db0d67e504926b35d5e0c5179bcc73de18155a1485`/`f6c1f6f0aec01d596068076f05f3fcc5e10fe53bd6f3f64d743d1fa260fe11cd`。
- Candidate checkpoint SHA：`529281c1f1ae4720773a3fe9c2e989d0f2b8a6feba06f820ddf929998b11f818`。
- Train-only inner与fixed validation已访问；outer、confirmation、test270未访问。

## 下一决定

关闭Gamma分支，不调概率/范围/顺序/retain-stats/seed/epoch/阈值，不进test。后续不能由Stage A正点估计声称有效；它被fresh final反证。真正提高证据等级仍需要独立外部cohort。

Evidence links: [七问](IDEA.md)、[预注册](PREREG.md)、[Stage A决定](STAGE_A_DECISION.json)、[raw inner](../../../workspace/BraTS/sammed3d_t1ce_wt/full972_gamma_aug_stability_20260902/STAGE_A_INNER.json)、[raw validation](../../../workspace/BraTS/sammed3d_t1ce_wt/full972_gamma_aug_stability_20260902/STAGE_B_VALIDATION.json)、[checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/full972_gamma_aug_stability_20260902/formal_stageb_gamma_candidate/raw_gamma_candidate_epoch168.pth)。
