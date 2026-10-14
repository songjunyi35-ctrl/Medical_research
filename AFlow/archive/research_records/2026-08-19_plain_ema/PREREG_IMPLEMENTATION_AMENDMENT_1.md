# Implementation amendment 1

Frozen before any checkpoint inference on validation.

The validation v1 entry incorrectly reused the outer-selector subject count of 45. The SHA-bound historical fixed validation manifest actually contains 108 cases / 50 subjects. V1 failed at this metadata assertion before creating an access receipt, loading checkpoints, running model inference, or writing metrics.

V2 changes only the asserted and reported subject count from 45 to 50. Manifest, cases, cache, checkpoints, EMA definition, threshold, metrics, bootstrap seed/sample count, primary comparison, and gates are unchanged.
