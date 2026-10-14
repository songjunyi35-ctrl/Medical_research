# FV-LoRA selected checkpoint: 108-case clean-validation reload audit

Date: 2026-07-24 UTC

## Scope and frozen inputs

- Scope was limited to the frozen 972-case train / 108-case subject-clean
  validation split. No test evaluation was run.
- Validation manifest:
  `data/datasets/brats2024_sammed3d_t1ce_wt_inner90_10_seed20260723_v1/validation_t1ce_wt.jsonl`
- Validation manifest SHA-256:
  `fc750ab15266b8a53c6d9a7ff62bd043b10cc21733e9b68354a2a989fd1f488c`
- Manifest audit: 108 unique scans, 50 subjects, all rows declare
  `split=validation`, target `WT`, foreground labels `{1,2,3}`, excluded label
  `{4}`.
- Selected checkpoint:
  `workspace/BraTS/sammed3d_t1ce_wt/fv_lora_r4_aug_train972_val108_e50_s20260725/sam_model_val_dice_best.pth`
- Selected checkpoint SHA-256:
  `c1d1bd79791aa7beeffc541095472dec968630cc70ea7967128c86552bd7f1ec`
- Training metadata binds this checkpoint to epoch 33 and trainer validation
  C3 `0.532816762311591`.

## Independent case-seeded reload

Frozen evaluator SHA-256:
`183886e13a7a6aacbe01f9c27f7cc9860edcb081bb7b84ec24ec98fba034c02e`.

The evaluation command was:

```bash
bash scripts/run_worker_research.sh eval-sammed3d-t1ce-wt \
  --manifest data/datasets/brats2024_sammed3d_t1ce_wt_inner90_10_seed20260723_v1/validation_t1ce_wt.jsonl \
  --expected-manifest-sha256 fc750ab15266b8a53c6d9a7ff62bd043b10cc21733e9b68354a2a989fd1f488c \
  --expected-split validation \
  --checkpoint workspace/BraTS/sammed3d_t1ce_wt/fv_lora_r4_aug_train972_val108_e50_s20260725/sam_model_val_dice_best.pth \
  --expected-checkpoint-sha256 c1d1bd79791aa7beeffc541095472dec968630cc70ea7967128c86552bd7f1ec \
  --spatial-preprocessing whole_resize \
  --device cuda:0 \
  --seed 20260723 \
  --output workspace/BraTS/sammed3d_t1ce_wt/fv_lora_r4_aug_train972_val108_e50_s20260725/INDEPENDENT_RELOAD_VAL108_E33_S20260723.json
```

| Aggregation | C1 | C2 | C3 |
|---|---:|---:|---:|
| scan macro | 0.5043246616 | 0.5114485611 | 0.5214277774 |
| subject balanced | 0.5035608318 | 0.5096221842 | 0.5182634436 |

Report SHA-256:
`43127d86b15b824dfe9c737234e0149e13068f1f8a4982b8c2369f26e4ccbf79`.
The report contains 108 unique cases, 50 subjects, exactly three click records
and three metric records per case; all DSC values are finite and in `[0,1]`.

The independent C3 differs from trainer C3 by `-0.0113889849`. This is not a
reload failure: the frozen independent evaluator derives a separate PCG64
stream from `(seed, case_id)` for every case, while trainer validation resets
one legacy NumPy MT19937 stream and consumes it over the sorted validation
rows.

## Trainer-RNG diagnostic replay

To isolate the RNG difference without changing the frozen evaluator, a
validation-only wrapper was added:
`scripts/replay_sammed3d_t1ce_wt_val_trainer_rng.py`, SHA-256
`278effb8b732d3cfcae9a685bb5119e88fc24f4a756b9a1ad469d623869b3920`.
It hard-binds the validation manifest and SHA, forces
`expected_split=validation` and `whole_resize`, and exposes the trainer's
legacy MT19937 stream through the evaluator's generator interface.

The replay command was:

```bash
PYTHONDONTWRITEBYTECODE=1 ../sammed3d-venv/bin/python \
  scripts/replay_sammed3d_t1ce_wt_val_trainer_rng.py \
  --device cuda:0 \
  --seed 20260723 \
  --output workspace/BraTS/sammed3d_t1ce_wt/fv_lora_r4_aug_train972_val108_e50_s20260725/TRAINER_RNG_REPLAY_VAL108_E33_S20260723.json
```

| Aggregation | C1 | C2 | C3 |
|---|---:|---:|---:|
| scan macro | 0.5125897108 | 0.5270505142 | 0.5328167620 |
| subject balanced | 0.5149064962 | 0.5261093594 | 0.5283455755 |

Replay report SHA-256:
`d545f580a3b2cae76a5d05df02a43f150aa9c0a34837b5f35870c46fb42e1d63`.
Replay C3 differs from the trainer epoch-33 value by only
`-3.4744807e-10`, i.e. it agrees to numerical precision. This validates
checkpoint reload, LoRA adapter restoration, whole-volume preprocessing and
the three-click recurrent-mask inference interface.

## Resource closure

Both evaluations ran on worker `1009989`, GPU 0 (NVIDIA A10), and exited with
code 0. Final `nvidia-smi` showed 0 MiB used, 0% utilization and no compute
applications.
