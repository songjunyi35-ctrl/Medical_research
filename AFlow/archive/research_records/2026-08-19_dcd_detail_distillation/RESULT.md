# DCD方向细节蒸馏 fixed-validation 负结果

## Outcome

MICCAI 2026 DCD的项目适配完成了一手论文/官方代码审计、七问与预注册、两项执行前amendment、独立新入口、11项合成测试、真实batch8 H100 smoke、global epoch169–200 formal训练，以及一次fixed validation108决策。

所有implementation、filter reconstruction、四stage gradient、teacher no-grad、显存、wall-time、checkpoint reload和deploy-equivalence门都通过；但selected DCD没有超过base voxel-KD的Scan或Subject点估计，因此promotion失败，**不访问 reused internal test270**。

结论不支持“在现有voxel-KD上增加KD200→student的db4方向detail distillation能改善最终WT mask”这一具体同容量、32轮continuation直觉。它不证明论文的大teacher→压缩student场景无效，也不授权调整teacher、wavelet、band、stage、mu或schedule。

## Idea and only changed factor

现有student仍从history168开始，原`0.75 history + 0.25 EMA` logit teacher、温度2、voxel-KD权重0.3全部保留。唯一增加：用same-basin KD200作为detail teacher，对四个SegResNet encoder stage `16@128³/32@64³/64@32³/128@16³`做：

- canonical `db4`三层递归3D DWT；
- 保留directional bands index1–6，排除LLL和HHH；
- 使用同一decomposition bank的PyTorch adjoint ConvTranspose3d重建detail-only feature；
- 每stage三level MSE取mean，四stage直接sum；
- 以官方BraTS YAML的max`mu=0.01` cosine-healing按32轮比例映射。

总损失为`DiceCE + 0.3 voxel-KD + mu(k) DCD`。same-channel不加projection，不加官方student EMA。训练后只部署原4.70M student final logit。

## Fixed validation108 / 50 subjects

按预注册只比较global epochs176/184/192/200，最大Scan Dice、Subject exact tie-break选择epoch200。

| DCD checkpoint | mu | Scan Dice | Subject Dice | Point hit |
|---|---:|---:|---:|---:|
| epoch176 | `0.000145291` | `0.731501079` | `0.721018331` | `0.925926` |
| epoch184 | `0.003226976` | `0.731575937` | `0.721023793` | `0.925926` |
| epoch192 | `0.007840324` | `0.732105775` | `0.722114761` | `0.925926` |
| **epoch200 selected** | **`0.010000000`** | **`0.733603344`** | **`0.723798160`** | **`0.935185`** |

### Selected epoch200 versus base voxel-KD200

| Metric | DCD | voxel-KD200 | DCD − KD | paired 95% CI |
|---|---:|---:|---:|---:|
| Scan Dice | `0.733603344` | `0.734384308` | `-0.000780964` | `[-0.001888339,+0.000057705]` |
| Subject Dice | `0.723798160` | `0.724434143` | `-0.000635983` | `[-0.001581392,+0.000123148]` |
| Point hit | `0.935185` | `0.935185` | `0` | — |

DCD相对KD200有49/108 scans改善、59变差；20/50 subjects改善、30变差。与matched supervised continuation epoch200相比，Scan只差`-0.000002316`，Subject高`+0.000126260`，几乎回到supervised control，未保留base voxel-KD相对control的微小增益。

## Promotion decision

通过：

- 11项合成测试与真实batch8 smoke；
- 四stage DCD专属gradient finite/nonzero，三位teacher无gradient；
- paired Scan/Subject CI下界安全门；
- Point within one case；
- 四checkpoint strict reload与stripped deploy logit差均为0。

失败：

- selected DCD Scan/Subject未超过同epoch base voxel-KD；
- selected DCD Scan/Subject未超过当前KD200。

因此`all_gates_passed=false`、`dcd_candidate_authorized_for_test=false`；无test evaluator运行、无test receipt、无Leaderboard变更。

## Mechanism interpretation

raw DCD loss从epoch169的`3.918064`降到epoch200的`0.548860`，下降约`86.0%`，说明student确实学会接近KD200的directional-detail feature；这不是loss没工作。epoch200四stage raw losses为`0.02649/0.04721/0.11637/0.35879`，深层16³stage贡献最大。

但detail matching没有转化成更高最终Dice。可能边界detail与当前二值WT的主要误差不完全一致，或同容量KD200 teacher没有提供base voxel-KD之外的新信息。这个解释只来自当前结果；不能据此在同一实验中换teacher/band/mu。

## Cost and deployment

- Formal训练：`5954.21s`（99.24min），约为base voxel-KD `1180.85s`的`5.04×`。
- Peak allocated/reserved GPU：`32.85/43.58 GB`，约`30.59/40.59 GiB`；H100 80GB。
- 四个checkpoint每个`18,835,600` bytes；DCD workspace约72M；不建新图像cache。
- 部署：4,700,897参数plain locator、1×forward；无teacher、DWT、projection、student EMA、SAM、TTA或postprocessing。
- 四个checkpoint reload max abs difference与stripped deploy difference：全部`0`。

## Implementation/source boundary

论文报告大teacher→r=4压缩student、四模态多类别nnU-Net、1000轮；这里是同容量T1ce/WT SegResNet的32轮增量。论文正文写`mu=0.05`，公开BraTS YAML使用max`0.01`；项目预先选择YAML并按比例映射。

官方`dec`→PyTorch Conv3d、`rec`→ConvTranspose3d组合不能通过全频重建审计；项目在执行前固定使用同一db4 decomposition bank做analysis与adjoint synthesis，全8-band max error门`<1e-5`。没有在validation后比较两种方向。

## Evidence and hashes

- Raw validation report: `workspace/BraTS/sammed3d_t1ce_wt/plain_dcd_20260819/DCD_VALIDATION.json`, SHA-256 `0bb0da55ec9f753378feada2f4b6cda6c81eb128ed8efe5f948cff2e404d6357`.
- Validation started receipt: SHA-256 `445b57cc3c5e9dab80e17a45ead0ae98f6978eb376b697042b7ad02fc3f7258a`.
- Validation pre-access receipt: SHA-256 `2688b87434d8353ef1c329b5f5095161ec7e263a37aaa23354aa654e1525a967`.
- Formal training report: SHA-256 `d1869fc9de40da6a8e4e1848579f2ff4d5abcf049b960913a6544277891361ac`.
- Implementation receipt: SHA-256 `725069c39d2e83231b3ed596a3100c662a28560b22244c70132bd64e8d7ec2e6`.
- Selected epoch200 checkpoint: SHA-256 `9354a0f6ef9e6747729ca5be61e3faa357b76f558898940e52e8f7b274f543dd`.
- Model/trainer/validation evaluator SHA-256: `0ac6c3...34a77` / `daea72...7ad2` / `fce0a8...63aa`.

## Access scope and next decision

本方法的新fixed validation108已访问。detail teacher KD200在本实验前已有历史validation选择及随后test访问；本次teacher选择只依据既有validation与same-basin证据。outer selector、confirmation和本方法的新test270均未访问。

当前DCD适配停止，不调参。下一handoff候选为CVPR 2026 IAD infusion-to-assimilation distillation，必须重新走七问和预注册。

Primary sources: [MICCAI 2026 paper](https://arxiv.org/html/2605.26382v1), [official repository](https://github.com/ClinicaAlpha/DCD-3D-MedSeg).
