# AFlow BraTS continuation handoff 5 — 2026-08-19

本文件承接`AFLOW_BRATS_HANDOFF_20260819_CONTINUATION_4.md`。

## 33. 严格2025–2026文献重检与voxel-logit KD单模型新高

用户要求idea必须来自2025年以后。三个agent重新严格检索2025–2026一手论文/官方代码，完整排序见
`AFLOW_BRATS_FRONTIER_IDEAS_2025PLUS_20260819.md`。首选为把已成功固定ensemble teacher
`0.75 history168 + 0.25 EMA168`蒸馏回单个同架构128³ SegResNet，依据2025 ensemble
segmentation distillation与2026 3D脑MRI/CT voxel-KL工作。

预注册单因素：student从历史epoch168初始化，续训global epoch169–200；matched control与KD使用
同seed`20260846`、同sampler/augmentation/LR。KD唯一变化为binary voxel KL，temperature`2`、
weight`0.3`。固定保存176/184/192/200，validation选择epoch200。全部8项gate通过：KD在
validation为`0.734384/0.724434/0.935185`，历史为`0.731601/0.722121/0.925926`，固定
ensemble teacher为`0.734171/0.724490/0.944444`。

复用internal test270结果：

| metric | historical single | KD single | KD-历史 | paired 95% CI |
|---|---:|---:|---:|---:|
| scan Dice | `0.741203949` | **`0.742976353`** | **`+0.001772404`** | `[-0.000489,+0.003902]` |
| subject Dice | `0.723179486` | **`0.725178533`** | **`+0.001999047`** | `[-0.001734,+0.005344]` |
| point hit | `0.929630` | **`0.955556`** | **`+0.025926`** | — |

这是当前最高单模型internal-test点估计；Dice CI跨0，不能称统计显著。相对双模型ensemble，KD
scan/subject低`0.002078/0.002166`但point完全相同，推理从2×恢复1×。KD训练耗时1,181秒，
matched control 661秒，训练约1.79×；推理仍单个4.70M locator。selected checkpoint SHA-256为
`56f54898ded11a0208e75f8000d57d4d053e80d6dedd760402730ad345c111b7`，test report SHA-256为
`36a72112595c8606537d2182d31a2558ea982e2f327a82c7a8cc6d4b079e7b0e`。完整归档见
`archive/research_records/2026-08-19_plain_voxel_kd/RESULT.md`。

实现审计：V1遗漏checkpoint保存所需架构元数据，首个checkpoint前失败；清理后V2 control干净重跑。
V2完整训练/保存/reload/写报告后仅最终print漏`json`依赖；V3只补打印依赖用于尚未运行的KD臂。
两次均在validation/test前，协议和科研结果未改变。

当前部署选择：若追求最高分，使用固定w=0.25双模型ensemble（test scan`0.745054`）；若追求单模型
效率，使用KD epoch200（test scan`0.742976`、point同ensemble）。下一条2025+候选按优先级为
ReCo-KD区域蒸馏或LoMix 3D，不应在已揭示test上调T/lambda/epoch。
