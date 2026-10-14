# 2026 ReCo-inspired region-normalized KD result

The ReCo-inspired extension completed all 32 continuation epochs with exact final reload. It added fixed foreground/boundary/background-normalized SmoothL1 teacher-logit loss (`weight=0.1`, boundary multiplier two) on top of the successful voxel-KD recipe.

The validation-selected ReCo checkpoint was global epoch192:

- scan Dice `0.732959670`
- subject Dice `0.723071822`
- point hit `0.944444`

At the same epoch192, ReCo minus base voxel-KD was scan `+0.000441`, subject `+0.000688`, and exactly one additional point hit. Paired scan/subject CIs crossed zero. Although the point sub-gate was conceptually met by one case, ReCo's best absolute Dice remained lower than the already selected base-KD epoch200 (`0.734384/0.724434/0.935185`). ReCo is therefore negative for the current benchmark objective and does not enter test270.

- Selected ReCo checkpoint SHA-256: `974ef3009b6f59f8eae78b19cf1b4ad84f364476f71455739aadd8c376167224`
- Training report SHA-256: `31fa50290ca17470508fa8a29dece6496b9e0fb27c3c15f9793a8af3b3daf74b`
- Validation report SHA-256: `0c1e690f6c928b770cb4b1a17322cb4c821a0d599da01f46dfa581da0e5eebf1`
- Training time: `1202.41s`
- Peak allocated/reserved GPU memory: approximately `20.35/23.03 GiB`
- Final reload max difference: `0`
- Test access: false

No region weight, boundary width, temperature, epoch, or threshold tuning follows this result.
