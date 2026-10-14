# True-128³ late-cosine EMA result

## Outcome

EMA completed exactly as preregistered: all 756 train cases, seed `20260833`, 200 epochs, EMA initialized after epoch100 and updated with decay `0.999` for exactly 9,400 optimizer steps. Raw and EMA checkpoints were saved at epochs168 and200; both epoch200 checkpoint reloads had maximum logit difference zero.

EMA improved the same-seed raw trajectory, especially at epoch168, but failed to beat the existing historical-best true-128³ epoch168 checkpoint on the same fixed validation108. The EMA direction is therefore negative for the benchmark objective and will not access test270.

## Same-trajectory EMA effect

| fixed endpoint | comparison | scan delta | subject delta | point-hit delta | paired subject 95% CI |
|---|---|---:|---:|---:|---:|
| epoch168 | EMA - raw | `+0.012271` | `+0.011547` | `+0.009259` | `[+0.005038,+0.020030]` |
| epoch200 | EMA - raw | `+0.000608` | `+0.000689` | `0.000000` | `[-0.001318,+0.002699]` |

This supports the mechanism that EMA can rescue a noisy trajectory. It does not establish a new project best.

## Comparison with the historical best

Historical true-128³ epoch168 on the fixed validation108/50 subjects:

- scan Dice `0.731601219`
- subject Dice `0.722120565`
- point hit `0.925926`

| candidate | scan | subject | point hit | scan delta vs historical | subject delta vs historical |
|---|---:|---:|---:|---:|---:|
| EMA168 | `0.718824903` | `0.710217848` | `0.944444` | `-0.012776` | `-0.011903` |
| EMA200 | `0.716017457` | `0.706696547` | `0.935185` | `-0.015584` | `-0.015424` |

EMA168 paired scan CI versus the historical best was `[-0.023367,-0.004621]`; subject CI was `[-0.022707,-0.003218]`. EMA200 was also significantly lower on both Dice means. Higher point hit did not compensate for the Dice loss.

## Artifacts

- raw168: `06103977212f62ad9b70a98700057cc27baef1e3e11a712732b2e080c839ed1a`
- EMA168: `4ea97dacb9a538993e1c16593bc56b03ce6dc3df2f457c1f0426439c6252197e`
- raw200: `30df173b051f6d8b21b23755fafcfcdffd3bd104a3dafae01548dc577340e878`
- EMA200: `b106740a8d255c30304958dfd3468c7742ac545db85b92d2e4a30c18fddcf0a9`
- training report SHA-256: `87d4a770a9ca9cdb098dd483bda9c8175d36e8a482f00b89fe42aef18981cd73`
- EMA validation report SHA-256: `a97d91a113e565a4ac2e5304feeedca4b56ea77c7a8b3d1c329fec41e0347242`
- historical comparison report SHA-256: `3404d598ade63ec6d2b7ccd44f67e284ca7f2c7c7a904546fdfcd1ed8fb95f12`

Training took `4117.05s`; peak allocated/reserved GPU memory was approximately `20.09/22.45 GiB`. The validation v1 metadata assertion incorrectly expected 45 subjects; it failed before inference. V2 corrected the SHA-bound manifest count to 50 subjects without changing any metric or decision rule.

Access: fixed validation true; outer selector false; confirmation false; test false.
