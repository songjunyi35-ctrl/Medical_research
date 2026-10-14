# KAS-on-voxel-KD fixed-validation 负结果

## Outcome

CVPR 2026 IAD的teacher-free Knowledge Assimilation Stage（KAS）完成了论文/官方代码审计、七问预注册、执行前amendment、独立模型/paired trainer/evaluator、12项合成测试、2×A100 paired smoke、control/KAS双臂32轮formal训练和一次fixed validation108决策。

所有implementation、双卡batch/RNG配对、projector gradient、显存、checkpoint reload和deploy-equivalence门通过；但selected KAS在Scan、Subject、Point和Subject CI门上均失败。因此**本方法的新test270不访问**。

本结论不支持“现有voxel-KD完成知识注入后，accepted-paper等权KAS能通过student自我同化提高WT分割”这一具体3D适配。它不是full IAD，因为没有重跑论文专有KIS prototype alignment；也不授权根据结果把MSE/Dice改成官方注释的`1e-4`权重。

## Method and only changed factor

两臂都从同一voxel-KD200开始，fresh AdamW、LR`1e-5`、teacher-free继续32轮：

- control：DiceCE only；
- KAS：最深`128@16³` encoder feature经registered `Conv3d(128→1,1×1×1)`、trilinear到128³，增加raw MSE、sigmoid Dice consistency和per-volume cosine InfoNCE；
- `L_KAS=DiceCE+0.01*(MSE+DiceConsistency+Contrastive)`；无detach；
- 训练后删除projector，只部署原student final logit。

唯一变化是KAS训练期projector/loss。起点、756例数据、seed20260860、batch8、sampler/增强、fresh optimizer、32轮、checkpoint grid、validation、threshold0和推理协议全部固定。control/KAS分别在两张A100-40GB独立进程运行；完整201–232 case-order和CUDA RNG hashes逐项相同。

## Fixed validation108 / 50 subjects

KAS按预注册在208/216/224/232中以Scan最大、Subject exact tie-break选择epoch232。

| Epoch | Control Scan/Subject/Point | KAS Scan/Subject/Point |
|---|---|---|
| 208 | `0.733130 / 0.723298 / 0.935185` | `0.732614 / 0.723029 / 0.925926` |
| 216 | `0.732926 / 0.723036 / 0.935185` | `0.731854 / 0.721956 / 0.916667` |
| 224 | `0.734413 / 0.724670 / 0.925926` | `0.732807 / 0.722862 / 0.916667` |
| **232 selected KAS** | `0.734134 / 0.724618 / 0.916667` | **`0.732915 / 0.723199 / 0.916667`** |

### Selected KAS232 comparisons

| Comparison | Scan delta / 95% CI | Subject delta / 95% CI | Point delta |
|---|---|---|---:|
| KAS − same-epoch control | `-0.001218828` / `[-0.003020162,+0.000461335]` | `-0.001418980` / `[-0.003375498,+0.000411416]` | `0` |
| KAS − historical KD200 | `-0.001468952` / `[-0.002870374,-0.000151189]` | `-0.001235103` / `[-0.003061812,+0.000419564]` | `-0.018519` |

相对same-epoch control，46/108 scans改善、62变差；19/50 subjects改善、31变差。相对KD200，50 scans改善、58变差；21 subjects改善、29变差；Point从101/108降到99/108。

## Promotion decision

通过：implementation/synthetic/paired smoke、两臂完整配对、所有reload/deploy、Scan CI安全下界。

失败：

- selected KAS未超过same-epoch control的Scan或Subject；
- selected KAS未超过historical KD200的Scan或Subject；
- Point仅99/108，低于冻结的至少100/108；
- Subject CI下界`-0.0030618`低于`-0.003`安全线。

因此`all_gates_passed=false`、`kas_candidate_authorized_for_test=false`；无test evaluator、无test receipt、无Leaderboard变更。

## Mechanism interpretation

KAS raw loss从epoch201的`48.0620`降到epoch232的`1.5396`，weighted项从`0.4806`降到`0.0154`；projector和student确实学会了encoder→logit一致性，不是模块没工作。

但accepted-paper等权gamma在初期让KAS约为supervised loss的2倍（smoke单batch甚至12.6倍），导致student主任务轨迹长期低于matched control。到后期KAS已很小，性能仍未恢复。这支持“当前loss尺度/约束方向压制主任务”的解释，但不能在本实验中事后改权重。

一个额外但非预注册test候选的观察：teacher-free control epoch224 validation Scan/Subject为`0.734413/0.724670`，点估计略高于原KD200的`0.734384/0.724434`，Point少1例。它没有经过本实验为control预注册的test promotion，不能据此访问test；只说明简单teacher-free低LR continuation比KAS更安全。

## Cost and deployment

- Control/KAS formal：`1098.52/1159.98s`；并行墙钟约`19.33min`，KAS单臂约control的`1.06×`。
- Peak allocated：control/KAS `21.78/21.91 GB`，约`20.29/20.41 GiB`；两张A100-SXM4-40GB。
- Workspace约217M；control checkpoint约18.84MB，KAS checkpoint约37.68MB（含full training projector state和stripped deploy state）。
- 部署：projector删除，仍4,700,897参数plain locator、1×forward，无teacher/SAM/TTA/postprocessing。
- 8个checkpoint reload与stripped deploy max abs difference全部0。

## Source/implementation boundary

论文支持KAS接在已有KD后，但未规定两阶段epoch。本项目把KD200视作infusion代理，并固定额外32轮。官方release当前KIS/KAS块大多被注释、loss未定义、projector在batch内临时创建且不进optimizer；因此项目按accepted paper公式重建registered projector。binary官方注释又把MSE/Dice额外降为`1e-4`，与paper冲突；项目预先选择paper等权gamma0.01，没有在smoke/validation后切换。

## Evidence hashes

- Raw validation: `workspace/BraTS/sammed3d_t1ce_wt/plain_iad_kas_20260820/KAS_VALIDATION.json`, SHA-256 `c727472954103b1f21b7e2e1bac02ea67a0c85837bde3b359fc13db70e6a6c0c`.
- Validation started: SHA-256 `41f9976d98c3d579af8627aadb7f8a40d95c5a28a3edabb5a082b85fde800916`.
- Validation pre-access receipt: SHA-256 `328568f0d1f7d44a8c11ae99689efe7f9d012ed08676e1abd77b9f91739ba989`.
- Formal pairing receipt: SHA-256 `cc8e21276f3e5a29ff472aa68e2ddb0e11b8b56e497a4f1db6a084806cd83856`.
- Control/KAS TRAIN_DONE: `ce987642...c6392` / `dbcab505...c8855e`.
- Selected KAS232 checkpoint: SHA-256 `845a6448a5d472c643146ecd11b43e03f440a29eae34ec029344f47d7acda409`.
- Model/trainer/evaluator SHA-256: `3a0f38...1ef33` / `289168...2ea8` / `342a42...c29b`.

## Access scope and next decision

本方法的新fixed validation108已访问；start KD200在此前已有validation选择及后续历史test访问，但本方法没有使用test数值设计或选择。outer selector、confirmation和本方法的新test均未访问。

当前KAS适配停止，不调参。下一优先候选转为2025/2026校准/难例loss，其中hard-binned mL1-ACE与当前MONAI SegResNet/BraTS最接近；仍需新七问和预注册。

Primary sources: [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Hong_From_Infusion_to_Assimilation_Distillation_for_Medical_Image_Segmentation_CVPR_2026_paper.pdf), [official repository](https://github.com/hjklearn/IAD).
