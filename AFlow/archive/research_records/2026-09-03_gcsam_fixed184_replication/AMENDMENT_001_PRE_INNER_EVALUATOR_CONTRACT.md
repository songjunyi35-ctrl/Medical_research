# Amendment 001: pre-inner evaluator and freeze contract

Status: frozen on 2026-09-03 after formal controls started, before any formal inner metric access and before any formal candidate training result exists

## Why this amendment exists

`IMPLEMENTATION_RECEIPT.json` froze the split/training runner and GPU smoke before formal training. The independent inner evaluator and four-artifact preaccess freezer were intentionally completed while train-only controls were running, so their hashes cannot be retroactively inserted into that pre-formal receipt. This amendment freezes them prospectively before inner access and does not change any training data, optimizer, epoch, seed, metric, or promotion gate.

## Frozen entry points

- Evaluator: `scripts/evaluate_t1ce_wt_gcsam_fixed184_replication_inner.py`, SHA-256 `d0aaa02d7feb124ad2a97cf742213288212fdf84096f1f251419e71eeef6d55c`.
- Preaccess freezer: `scripts/freeze_t1ce_wt_gcsam_fixed184_inner_preaccess.py`, SHA-256 `81af898dccfc18f732a1a787bcb20e91c81e1279842a39e94c26d00571bd925f`.
- Focused tests: `tests/test_gcsam_fixed184_replication.py`, SHA-256 `73e3a7848a58c9811e41a1e64d29a44d2d4f19e2a348c461b51a76304d5830ba`.
- Test result with historical optimizer contracts: 17 passed; one environment-only CUDA availability warning in the non-GPU sandbox.

## Access and sequencing contract

1. Complete all four formal epoch184 training reports/checkpoints first.
2. The freezer validates every report/checkpoint, access flags, model-state hash, and both pairs' exact initial state, sampling weight, first augmented batch/loss, and all184 epoch sampler hashes.
3. Only then write `INNER_PREACCESS.json`, with `inner_metrics=false`, binding all four artifact hashes, both inner manifest hashes, this evaluator hash, 10,000 bootstrap samples, and frozen gates.
4. The evaluator refuses to construct either inner dataset unless the receipt file hash is explicitly supplied and every artifact still matches it.
5. Each inner is opened once; threshold0 and epoch184 are fixed. No checkpoint selection exists.
6. Any gate failure stops before Full972/fixed validation. Test270 and UCSD75 remain forbidden regardless of outcome.

## No changes

- No new data or metric access occurred while writing this amendment.
- Formal controls use the already-frozen training runner SHA from `IMPLEMENTATION_RECEIPT.json`.
- GCSAM `rho=0.05`, optimizer semantics, teachers, loss, splits, seeds, fixed epoch184, bootstrap seeds, and promotion gates are unchanged.
