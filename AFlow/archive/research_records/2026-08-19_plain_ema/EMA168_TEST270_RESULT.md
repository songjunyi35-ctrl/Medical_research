# EMA168 single-model test270 result

At the user's explicit request, the fixed EMA168 checkpoint was evaluated by itself on the reused internal test270. EMA200 was not tested. Threshold remained raw logit `>0`; there was no TTA, postprocessing, SAM, EMA retuning, or checkpoint selection from test.

| metric | historical epoch168 | EMA168 single | EMA - historical | paired 95% CI |
|---|---:|---:|---:|---:|
| scan Dice | `0.741203949` | `0.740696468` | `-0.000507482` | `[-0.005142,+0.003969]` |
| subject Dice | `0.723179486` | `0.724904822` | `+0.001725336` | `[-0.005896,+0.009275]` |
| point hit | `0.929630` | `0.948148` | `+0.018519` | — |

EMA168 improved 128/270 cases and worsened 140, with two ties; 66/119 subjects improved and 53 worsened. Both Dice paired intervals cross zero. The single EMA model is therefore statistically compatible with the historical model on Dice and better on point hit, but it does not set a new scan-Dice high.

This explains why the fixed `0.75 historical + 0.25 EMA168` ensemble works: EMA contributes complementary point/error information, while the stronger historical model preserves headline Dice. The ensemble remains the recommended benchmark method.

- EMA168 checkpoint SHA-256: `4ea97dacb9a538993e1c16593bc56b03ce6dc3df2f457c1f0426439c6252197e`
- EMA168 test report SHA-256: `6e4515dc098143016b02b7c7121970c4bc4fcf5a61e1164b64d6863858c6e739`
- Test pre-access receipt SHA-256: `a6e148d92f61837294cbb2a4c7aba9b273359ffd22a2adbb814e1bbef9a1540a`

This reused internal test is not pristine blind or external confirmation. Result will not trigger EMA or threshold tuning.
