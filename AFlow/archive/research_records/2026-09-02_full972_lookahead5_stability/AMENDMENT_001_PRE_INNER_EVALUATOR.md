# Amendment 001: Stage A pairing and inner evaluator freeze

Status: frozen while paired formal training was at epoch2/168, before completion, pre-access receipt, or any train-only inner/fixed-validation/outer/confirmation/test access.

This amendment does not change Lookahead, AdamW, data, seed, epoch, loss, metric, gate, threshold, or access rules.

## Frozen evaluator

- Entry point: `scripts/evaluate_t1ce_wt_lookahead5_stage_a_inner.py`
- SHA-256: `9fbd17c64d93f61fdb03521d48041681c3448a3c89e7d5f9c18514b3b3797f26`
- Tests: `tests/test_lookahead_stage_a_inner.py`
- Test SHA-256: `737abd8f4aafc816c9966681a5c43b000ed2b08c52a747dc6c19ed41cf811232`
- Combined optimizer/pairing/promotion tests: `8 passed`.

## Mandatory pre-inner engineering gates

- Control/Candidate initial model-state SHA exact.
- Native sampling-weight SHA exact.
- All168 requested sampler-index sequence SHA values exact.
- First augmented image and label tensor SHA exact.
- Candidate fast steps=`20328`,slow updates=`4065`,trailing fast steps=`3`,final slow loaded=true,all count gates true.
- Both training reports/checkpoints/pairing reports bind the frozen trainer/evaluator and all access flags false.
- Split remains864 fit/392 subjects and108 inner/52 subjects with zero case/subject overlap.

## Frozen promotion gates

- Candidate slow168 Scan and Subject must each strictly exceed matched AdamW Control168.
- Candidate Point must not decrease.
- Paired Scan and Subject 95% CI lower bounds must each be`>=-0.003`.
- Engineering/pairing gates must all pass.
- 10,000 bootstrap draws, seed20261073.
- Any failure sets`full972_final_authorized=false` and prohibits hyperparameter or threshold changes.

## Access at freeze time

- Train-only inner: not accessed.
- Fixed validation108: not accessed.
- Outer: not accessed.
- Confirmation: not accessed.
- Test270: not accessed.
