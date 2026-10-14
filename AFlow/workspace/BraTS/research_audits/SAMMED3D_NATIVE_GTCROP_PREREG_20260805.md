# SAM-Med3D native-geometry arm preregistration (2026-08-05 UTC)

Run identity: `sammed3d_native_gtcrop_fullunfreeze_ddp2_train972_val108_e100_s20260805`

Status: **PRE-AUTHORIZED by the operator's standing instruction** ("如果测评结果不行的话
去调研 medsam3d 是怎么训练的 重新训再测评"), issued 2026-08-05 before going to sleep.

## 1. Why this arm exists

The full-unfreeze arm reached test scan-macro C3 `0.571109` -- a new project best,
but `0.170` short of the `0.741` reference. Investigation
(`SAMMED3D_NATIVE_PROTOCOL_INVESTIGATION_20260805.md`) found that the shortfall is
dominated by **spatial protocol**, not by training quality:

- SAM-Med3D's own `train.py:78` and `validation.py:294` both use
  `tio.CropOrPad(mask_name='label', target_shape=(128,128,128))` -- a label-centred
  crop at **native resolution**, commented "crop only object region". They never
  resize the whole volume.
- This project's own ledger records that an **untouched, un-finetuned** SAM-Med3D
  checkpoint already scores WT `0.716053` (and a Phase-0 re-run `0.7194`) under
  that native protocol -- i.e. essentially the reference number, with zero training.
- The arm just completed used whole-volume `182x218x182 -> 128^3` resize, which
  shrinks the lesion to ~0.29x its native voxel count and coarsens exactly the
  boundary voxels that dominate Dice.

## 2. Single-factor design

This arm changes **exactly one thing** versus the completed full-unfreeze arm:

| | completed arm | this arm |
|---|---|---|
| **spatial** | whole-volume resize to `128^3` | **label-centred `CropOrPad(128^3)` at native resolution** |
| N4 bias correction | on, shrink 2 | on, shrink 2 (identical) |
| trainable | full unfreeze, 100,509,232 | identical |
| encoder / head LR | `1e-4` / `1e-5` | identical |
| weight decay | `0.1` | identical |
| effective batch | 12 (2 x 2 GPUs x 3 accum) | identical |
| precision | BF16 | identical |
| clicks | 3 accumulated + recurrent mask | identical |
| train / validation seed | `20260805` / `20260723` | identical |
| split | frozen 972 / 108 | identical |

So the validation delta between the two arms is attributable to the spatial
protocol alone. That is the point of the arm.

## 3. Two deliberate schedule corrections

The completed arm exposed a design flaw of mine: `patience=40` was smaller than the
first LR milestone at epoch 120, so early stop fired at epoch 64 and
**MultiStepLR never decayed even once**.

Corrected here, keeping the official *relative* schedule:

- `max_epochs = 100` (the completed arm peaked at epoch 24 and plateaued by 40, so
  200 was never the binding constraint);
- `lr_milestones = [60, 90]` -- the same 60% / 90% fractions of the budget that the
  official `[120, 180]` of 200 represents;
- `patience = 101`, i.e. **no early stop**: the full budget runs so the decay
  actually happens and its effect is observable. Overfitting is handled by the
  selection rule, not by stopping.

## 4. Selection and readout

Selected checkpoint = strict argmax of validation final-click scan-macro Dice over
completed epochs, ties keeping the earlier epoch. Then exactly one independent
strict-reload evaluation on the 108-case / 50-subject clean validation, seed
`20260723`, sourced from the matching gt_crop N4 cache so train/eval preprocessing
agree by construction.

Secondary, evaluation-side only and free: the same selected checkpoint is also
scored at **5 clicks**, the official `--num_clicks` default, to quantify the second
protocol lever without retraining.

## 5. The honest caveat that must travel with any number from this arm

A label-centred crop uses the ground-truth mask to place the field of view at
inference. That is **oracle localization**. Any Dice obtained this way is an
interactive upper bound and **must not** be placed in the same column as automatic
methods (nnU-Net `0.728`, U-Mamba `0.815`), which localize the lesion themselves.

This is precisely the ambiguity in DPD-SAM's table, which lists SAM-Med3D `74.1`
among automatic methods without disclosing its prompt protocol, and which
BCGFMamba then transcribes as `0.731` -- a value that does not match DPD-SAM's own
table.

If this arm reaches ~0.72-0.74, the correct claim is *"we reproduce SAM-Med3D's
reported operating point under its native oracle-interactive protocol"*, never
*"we match the automatic baselines"*.

## 6. Prohibitions in force

- the 270-case test is NOT read by this arm; it has already been consumed 5 times
  and any further use requires a fresh operator decision and a fresh freeze;
- no threshold / TTA / morphology / connected-component / postprocess search;
- no mid-run change of any value above;
- the completed arm's test result `0.571109` must NOT be used to select or tune
  anything here; selection is validation-only.
