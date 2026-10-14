# Local storage cleanup — 2026-08-10

User-authorized cleanup removed regenerable IDE artifacts and superseded T1ce
checkpoint payloads. Source code, manifests, training configs, histories, metrics,
evaluation JSON, NAS data, caches, the SAM-Med3D base checkpoint, and the current
T1ce mainline checkpoints were not removed.

## Removed checkpoint payloads

- 31 `.pth`/`.pt` files (12.907 GiB).
- Scope: July DPD-SAM and early SAM-Med3D T1ce weights, `INVALID` and smoke
  weights, superseded partial/LoRA weights, and duplicate `latest` weights.
- Experiment directories and their non-checkpoint provenance files remain.

## Retained checkpoint payloads

- `sammed3d_braincrop_fullunfreeze_ddp2_train972_val108_e200_s20260806/sam_model_val_dice_best.pth`
  — `87dad82c67151128c7e99d64ae87f37c68547c769d2e610a56d432b554e0732a`
- `sammed3d_native_gtcrop_fullunfreeze_ddp2_train972_val108_e100_s20260805/sam_model_val_dice_best.pth`
  — `42f96a0a2ade9d98d9db897a11a5a285f7a6351a7693c6884d6ac97a29dfde52`
- `sammed3d_official_fullunfreeze_ddp2_train972_val108_e200_s20260805/sam_model_val_dice_best.pth`
  — `75dd0b55f9fb29f558b97f9bd8f306b57832c244e49252f2f3095a82dd464e82`
- `sammed3d_stage2_gtcrop_last4_h100_train972_val108_e200_s20260807/sam_model_val_dice_best.pth`
  — `f999e12432901e9f1ce21f2c087b6fcfd123abf1210de9b8f883bb1a4776422f`
- `residual_refiner_stage2_gtcrop_h100_train972_val108_e60_s20260810_v2/refiner_val_dice_best.pth`
  — `4513ea4544ae153064575ec2ed5c56e1b78adf5e89e9d094754227b3bd429f21`

The upstream base checkpoint `../SAM-Med3D-main/ckpt/sam_med3d_turbo.pth`
also remains, SHA-256
`899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c`.

## Removed IDE artifacts

- VS Code cached VSIX installers.
- One inactive old VS Code server release.
- One superseded Claude extension release.
- Inactive Trae and MarsCode server installations.
- Four superseded CloudIDE Merlin extension releases.

Active CloudIDE/VS Code server releases and the active ChatGPT/Codex extension
were retained.

## Result

- System disk changed from approximately 87 GiB used / 39 GiB available to
  71 GiB used / 55 GiB available (`57%` utilization).
- Five T1ce checkpoint payloads remain, totaling 1.551 GiB.
