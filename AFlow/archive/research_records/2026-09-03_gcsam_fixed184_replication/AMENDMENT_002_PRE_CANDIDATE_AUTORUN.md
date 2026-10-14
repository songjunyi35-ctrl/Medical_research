# Amendment 002: candidate autorun and one-time inner sequencing

Status: frozen while controls are running, before candidate construction and before inner access

The long Stage A sequence is automated only to avoid idle GPUs or accidental reordering. This does not change the method, data, seeds, epoch, evaluator, or gates.

- Entry point: `scripts/run_t1ce_wt_gcsam_fixed184_stage_a_after_controls.sh`.
- SHA-256: `f7b416e27c642eb2924429a56b8021a6910af796f77190e9553d76b0c70b2e0e`.
- Syntax check: `bash -n` passed.

Frozen sequence:

1. Poll only for both control `TRAIN_DONE.json` files.
2. Validate both as complete epoch184 train-only reports with the frozen runner hash and access flags.
3. Refuse if either formal candidate directory already exists.
4. Launch replicate A/B GCSAM candidates in parallel on cuda:0/cuda:1 with the frozen fit manifests and seeds.
5. If either candidate exits nonzero, stop without freezing or opening inner.
6. If both succeed, run the frozen four-artifact preaccess freezer.
7. Compute the receipt SHA-256 and supply it explicitly to the frozen evaluator.
8. Open each inner once and write `FIXED184_REPLICATION_INNER.json`.

The script contains no fixed-validation, outer, confirmation, test270, UCSD75, threshold-search, or postprocessing command. It does not update documentation or authorize Stage B; the main agent must inspect and archive the Stage A decision after completion.

