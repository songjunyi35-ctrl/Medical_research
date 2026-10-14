# 2025/2026 voxel-logit ensemble distillation result

## Outcome

A single true-128³ SegResNet student distilled from the fixed teacher

`z_teacher = 0.75 z_history168 + 0.25 z_EMA168`

completed a matched 32-epoch continuation experiment. The KD-only factor was binary per-voxel KL with temperature `2` and weight `0.3`, following the selected 2025/2026 literature. The student starts from historical epoch168 and deploys as one ordinary 4.70M-parameter locator.

All validation gates passed. The validation-selected KD global epoch200 checkpoint set a new single-model point estimate on reused internal test270, though paired Dice intervals versus the historical model cross zero. The two-model logit ensemble remains the overall project high.

## Matched validation at selected global epoch200

| model | scan Dice | subject Dice | point hit |
|---|---:|---:|---:|
| historical epoch168 | `0.731601219` | `0.722120565` | `0.925926` |
| matched supervised continuation | `0.733605660` | `0.723671900` | `0.935185` |
| **voxel-KD continuation** | **`0.734384308`** | **`0.724434143`** | **`0.935185`** |
| fixed two-model teacher | `0.734171097` | `0.724489716` | `0.944444` |

KD minus matched control was scan `+0.000779` and subject `+0.000762`; both matched CIs cross zero. Most of the validation improvement over historical comes from the controlled epoch169–200 continuation, with a smaller positive KD contribution. KD nevertheless met the preregistered teacher-gain absorption and point gates.

## Reused internal test270

| metric | historical single | KD single | KD - historical | paired 95% CI |
|---|---:|---:|---:|---:|
| scan Dice | `0.741203949` | **`0.742976353`** | **`+0.001772404`** | `[-0.000489,+0.003902]` |
| subject Dice | `0.723179486` | **`0.725178533`** | **`+0.001999047`** | `[-0.001734,+0.005344]` |
| point hit | `0.929630` | **`0.955556`** | **`+0.025926`** | — |

KD improved 175/270 cases, tied two, and worsened 93; 80/119 subjects improved and 39 worsened. It is the current highest single-model internal-test point estimate. The Dice intervals cross zero, so it is not a statistically established improvement.

Compared with the fixed two-model ensemble, KD has identical point hit but scan/subject are lower by `0.002078/0.002166`; the scan paired CI is `[-0.004183,-0.000032]`, while subject CI crosses zero. This is the measured cost of compressing the two forward passes into one.

## Efficiency

- Matched control 32-epoch continuation: `661.0s`
- KD continuation: `1180.8s` (`1.79×` training wall time)
- Control/KD peak allocated GPU memory: approximately `20.07/20.29 GiB`
- Final inference: one 4.70M-parameter locator (`1×`), no teacher, TTA, postprocessing, or SAM
- Final KD checkpoint reload maximum logit difference: `0`

## Artifacts and implementation audit

- Selected KD checkpoint SHA-256: `56f54898ded11a0208e75f8000d57d4d053e80d6dedd760402730ad345c111b7`
- Matched control epoch200 SHA-256: `0d4246a700e7f85b501faea96dc2f5cc348bfebed42a46bbd0fc76c5963f9bdb`
- Validation report SHA-256: `db7c3bc6b5ff9e1076827d25c9f4de0dea88781bfc5de45da0fb2f79a09bfb57`
- Test report SHA-256: `36a72112595c8606537d2182d31a2558ea982e2f327a82c7a8cc6d4b079e7b0e`
- Test pre-access receipt SHA-256: `232cbac6d273fa566ea8d6b55d8f4ef470492bb50f511fd647cb68820e6928db`
- Synthetic KL/blend/soup regression: 9 passed

V1 omitted frozen architecture metadata in the CLI and failed before the first checkpoint. A clean V2 control rerun completed; its final report was written successfully but the process then missed a final `json` print dependency. V3 added only that print dependency for the not-yet-run KD arm. Both incidents occurred without validation/test access and are documented in implementation amendments.

## Evidence scope

The method was selected from 2025/2026 primary literature, but test270 is reused and previously revealed. This is not pristine blind test evidence, external confirmation, or a public BraTS SOTA claim. Test does not trigger tuning of temperature, KD weight, teacher ratio, or epoch.

Primary method sources: [2025 ensemble segmentation distillation](https://arxiv.org/abs/2509.11689) and [2026 3D brain MRI/CT KD](https://www.nature.com/articles/s44401-026-00108-w).
