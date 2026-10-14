# Amendment 005: 固定formal文档表并纠正validation cache路径

Status: frozen after the first CPU-only evaluator contract test and before validation model/data access.

## Failures observed

The first evaluator contract suite passed the candidate 32-epoch adjacent-coverage audit but failed three evidence-wiring tests:

1. A fresh Python process imported the current runner without calling its runtime configuration. `train.method_documents()` therefore inherited one stale historical native-module constant for the key `amendment_001_sha256`, while both formal reports and all eight checkpoints consistently contain the training-time frozen sampler-spec hash.
2. The evaluator test guessed a validation cache directory that does not exist. A read-only SHA search located the canonical index at `workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_ema_20260819/validation108/index.json`, whose SHA is the already frozen `9339e457...f322`.

No validation image/label was loaded, no model was constructed by the evaluator, and no validation, outer, confirmation, or test metric was computed.

## Frozen correction

- Define the exact training-time method-document dictionary explicitly in the new evaluator, byte-for-byte equal to both formal `TRAIN_DONE.json` files and all formal checkpoints. Do not mutate or rewrite any training artifact.
- Use the canonical validation108 cache directory identified by its already frozen index SHA. The cohort manifest, count108/50, threshold, checkpoint grid, bootstrap seeds, selection rule, and promotion gates remain unchanged.
- The validation pre-access receipt must bind this amendment SHA, evaluator SHA, implementation receipt SHA, both formal report SHAs, all eight checkpoint SHAs, validation manifest/index SHAs, KD200 report SHA, grid, statistics, selection, and access flags before the evaluator starts.

This amendment fixes evidence lookup only; it changes no model, sample, output, metric, or decision rule.

Access flags at amendment freeze: validation=false, outer=false, confirmation=false, test270=false.
