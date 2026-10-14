# Amendment 001: Stage A inner evaluator contract

Status: frozen on 2026-09-02 before completion of Stage A training, creation of a pre-access receipt, or any train-only inner/fixed-validation/outer/confirmation/test access.

This amendment does not change the method, `k=6`, capture epochs, seeds, metrics, promotion gates, threshold, or access rules. It freezes the standalone evaluator implementation required by the preregistration.

## Frozen evaluator

- Entry point: `scripts/evaluate_t1ce_wt_lawa6_stage_a_inner.py`
- SHA-256: `96c9d26d0f181f71237efc849aab833187ca8f7614f94ef0e8ed4f91e47df8db`
- Gate-test file: `tests/test_lawa6_stage_a_inner.py`
- Gate-test SHA-256: `198ad362ca1eed08e5708329b1df8cdb5cdf80f868d8aab77292011bd3b6c73e`
- Combined command: `PYTHONDONTWRITEBYTECODE=1 ../sammed3d-venv/bin/python -m pytest -q tests/test_lawa6_stage_a_inner.py tests/test_lawa6_utils.py`
- Result: `9 passed`.

## Evaluator input and access contract

- Requires both formal Stage A `TRAIN_DONE.json` files with `all_gates_passed=true`, the exact frozen trainer SHA, `smoke=false`, capture epochs`163..168`, and all access flags false.
- Requires each report-bound raw and LAWA checkpoint SHA, checkpoint/state hash, arm, kind, epoch168, access seal, and LAWA capture metadata to match.
- Requires fit`864/393 subjects` and inner`108/51 subjects`, with case and subject overlap both zero.
- Requires a separate pre-access receipt binding evaluator SHA, trainer SHA, report SHA values, checkpoint SHA values, and all access flags false.
- Refuses repeated evaluation if either the final output or `.started.json` exists.
- Writes `.started.json` before constructing the inner dataset/loader and marks only `train_only_inner_access=true`.

## Frozen promotion calculation

For each of seed20261061 and seed20261062 independently:

- LAWA6 Scan delta over same-run raw168 must be strictly positive.
- LAWA6 Subject delta over same-run raw168 must be strictly positive.
- LAWA6 Point hit rate must not be lower.
- Paired Scan and Subject 95% CI lower bounds must each be`>=-0.003`.
- Bootstrap count remains10,000 with seed`20261064`.

All ten booleans are conjunctive. Any failure writes`full972_final_authorized=false`; it does not authorize seed/epoch/window/weight/threshold changes.

## Access at freeze time

- Stage A formal training was still running at epoch31/168 for both arms.
- Train-only inner: not accessed.
- Fixed validation108: not accessed.
- Outer: not accessed.
- Confirmation: not accessed.
- Test270: not accessed.
