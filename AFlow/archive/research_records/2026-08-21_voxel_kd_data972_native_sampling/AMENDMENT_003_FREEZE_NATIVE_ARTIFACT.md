# Pre-implementation amendment 003: freeze passed native-volume artifact

Status: frozen after train-only feasibility generation and before model implementation, synthetic model tests, GPU, formal, or validation access.

## Frozen implementation evidence

- Builder: `scripts/build_t1ce_wt_native_volume_artifact.py`.
- Builder SHA-256: `80874d4745d7ae5b0ced90471f0fa59d9c6345a597b27ec02a253eb6a19c010a`.
- Artifact: `archive/research_records/2026-08-21_voxel_kd_data972_native_sampling/NATIVE_WT_VOLUMES.json`.
- Artifact SHA-256: `32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31`.
- Artifact schema/status: `aflow.brats.t1ce_wt_native_volume_artifact`, `passed`.
- Candidate/control counts:972/756; every crop-native WT count is positive.
- Overlapping source-volume mismatches:0/756.
- Control native-weight tensor exact equality: true, max-abs difference`0`.
- Generated/historical control weight SHA-256: `757e463d8e60f96ef6429613f6783fb3aa1014b580932959f9ed354d37072124`.
- Candidate native-weight SHA-256: `81d3fa6a907f791c4d0b97efe95ece5e057d15b85673463cadbeecc2b3f148ec`.
- Weight range for both arms: `[0.5,4.0]`.

The trainer must load this exact artifact, reconstruct weights from the selected case IDs and `native_wt_voxels`, and reject any tensor SHA mismatch. The control arm must equal the historical control weight SHA above; the candidate must equal its frozen candidate SHA.

## Failed pre-output attempts

For transparency, three builder attempts preceded the final artifact and none created an output or accessed model/validation/test data:

1. Full-FOV WT definition: exact gate found one mismatch and stopped.
2. Image-crop definition without RAS canonicalization: exact gate found23 mismatches and stopped.
3. Correct canonical definition passed parity, then a Python `false`/`False` serialization typo stopped before output creation.

Amendments001/002 froze the two semantic fixes before the successful artifact. The final boolean fix changed no method/data value. No result metric informed these corrections.

