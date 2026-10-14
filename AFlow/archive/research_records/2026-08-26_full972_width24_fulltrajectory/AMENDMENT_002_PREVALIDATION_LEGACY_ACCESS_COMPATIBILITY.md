# Pre-validation legacy access-field compatibility

Status: frozen before any fixed validation108 dataset construction or inference.

The historical implementation receipt was created before access tracking was split into
`fixed_validation108_access`, `outer_selector_access`, and `confirmation_access`. It records
the complete legacy seals `validation_access=false` and `test_access=false`; the underlying
inner/final reports and checkpoints carry every newer split field as `false`.

Two attempted evaluator launches stopped while parsing this implementation receipt, before
creating `WIDTH_VALIDATION.started.json`. The evaluator therefore accepts the two legacy
fields only for that historical receipt. It continues to require every split field on the
pre-access receipt, both final training reports, and both final checkpoints.

No model, epoch, threshold, metric, bootstrap seed, gate, validation case, outer selector,
confirmation set, or test set changed.
