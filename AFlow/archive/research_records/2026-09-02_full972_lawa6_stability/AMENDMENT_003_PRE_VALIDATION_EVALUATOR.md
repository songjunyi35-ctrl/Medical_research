# Amendment 003: Stage B fixed-validation evaluator contract

Status: frozen while Stage B formal training was at epoch1/168, before its completion, validation pre-access receipt, or any fixed-validation/outer/confirmation/test access.

This amendment does not change training, LAWA, data, seed, metric, comparator, promotion gate, threshold, or access settings.

## Frozen evaluator

- Entry point: `scripts/evaluate_t1ce_wt_lawa6_stage_b_validation.py`
- SHA-256: `8462b81592c60b72e118081e9c9eca74a283039b788af8ae758e3dbe68754987`
- Test: `tests/test_lawa6_stage_b_validation.py`
- Test SHA-256: `f571ac59b87222e4d1240412b890376371460b7539562ed3fd7f68ec8ae6ff52`
- Combined LAWA utility/Stage A/Stage B suite: `14 passed`.

## Frozen input contract

- Stage B runner/checkpoint schema, arm`stageb_full972_seed20261063`, seed20261063,168 history rows,972 scans,972 requested/968 consumed/121 steps, captures`163..168`, all engineering gates and access seal must match.
- Fixed validation manifest SHA-256: `f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb`.
- Fixed validation cache index SHA-256: `9339e4575561447dd5bd99bc29e909ae1e9f566721d351a244d087a40622f322`.
- Archived Full972 seed32 validation report SHA-256: `30cd65c78ad1ed2a2b5748652f48ec05cde3b54468b44a66eba77d062d3ce129`.
- A separate pre-access receipt must bind evaluator, runner, train report, raw checkpoint, LAWA checkpoint, archived report, and all pre-access flags false.
- Evaluator refuses repeated access and writes `.started.json` before dataset/loader construction.

## Eleven conjunctive gates

1. LAWA Scan strictly exceeds matched raw.
2. LAWA Subject strictly exceeds matched raw.
3. LAWA Scan strictly exceeds archived Full972 seed32`0.740012535`.
4. LAWA Subject strictly exceeds archived Full972 seed32`0.731266217`.
5. LAWA Point is at least matched raw.
6. LAWA Point is at least103/108.
7–8. LAWA-vs-raw Scan/Subject paired CI lower bounds are`>=-0.003`.
9–10. LAWA-vs-archived Scan/Subject paired CI lower bounds are`>=-0.003`.
11. All Stage B training engineering gates passed.

Bootstrap remains10,000 draws with seed20261065. All gates must pass to authorize the single frozen reused-test report; otherwise stop without changing`k`, epochs, weights, seed, threshold or postprocessing.

## Access at freeze time

- Fresh train-only inner: accessed once in Stage A.
- Fixed validation108: not accessed.
- Outer: not accessed.
- Confirmation: not accessed.
- Test270: not accessed.
