# 972-case native-volume-weighted voxel-KD preregistration

Status: frozen before data-artifact generation, implementation, GPU smoke, formal, and new validation access.

## Frozen causal design

Run a cache-matched two-arm experiment. Both arms use the same existing 972 cache and native-source WT-volume weight implementation; the only arm contrast is eligible population 756 versus972. Both use exactly756 sampler draws/epoch, batch8/drop-last, 94 optimizer steps/epoch, and32 history168-to-200 voxel-KD continuation epochs.

The native WT-volume artifact is generated only from the frozen 972 development-train manifest labels. Its exact path/SHA, builder SHA, counts, overlap audit and weight-vector audit must be frozen in a pre-implementation amendment after generation and before any model code/smoke/formal.

## Frozen inputs

- Control manifest SHA `a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16`, 756 scans / 346 subjects.
- Candidate manifest SHA `6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65`, 972 scans / 444 subjects.
- Shared cache index SHA `7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd`, `test_access=false`.
- Historical formal756 index SHA `d26b823a4715d9f3676d75cddabcf630180d48807352c23042be7faf240b622d`; only its frozen `per_case_source_positive_voxels` is used for the exact overlap audit.
- Historical168 init SHA `f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1`.
- EMA168 teacher SHA `4ea97dacb9a538993e1c16593bc56b03ce6dc3df2f457c1f0426439c6252197e`.
- Archived KD validation SHA `db7c3bc6b5ff9e1076827d25c9f4de0dea88781bfc5de45da0fb2f79a09bfb57`.
- Fixed validation manifest/index SHA `f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb` / `9339e4575561447dd5bd99bc29e909ae1e9f566721d351a244d087a40622f322`.

## Native-volume artifact

- For each manifest row, load only `label_path`; require a finite 3D integer-valued label image.
- WT is labels `{1,2,3}`; label4 is excluded. Store count of WT voxels in the source NIfTI grid.
- Require 972 unique case IDs, all volumes positive, no missing files, and no test paths/roles.
- For the overlapping frozen756 IDs, require exact integer equality to the historical formal index source-volume mapping.
- Weight formula per selected arm population: `clip(sqrt(median(selected_native_volumes)/native_volume_i),0.5,4.0)`, float64 CPU tensor. Median is recomputed within the selected arm population, matching the historical dataset behavior.
- For the 756 arm, require the full weight tensor SHA and values to equal `VariableResolutionBrainCropDataset(...).sampling_weights()` from the historical formal756 cache exactly.

## Frozen training

- Arms: `native756_control`, `native972_candidate`.
- Student/teachers/loss/autocast/RNG/optimizer/augmentation are byte-for-byte semantically identical to the prior fixed-compute data experiment and original successful voxel-KD protocol.
- `WeightedRandomSampler(native_weights, num_samples=756, replacement=True, Generator(20260846))`; 752 consumed, 4 dropped, 94 steps/epoch.
- AdamW LR uses archived late-cosine global169-200, base`2e-4`, min`1e-5`, wd`1e-5`; clip norm5 with nonfinite error.
- Save global176/184/192/200. Training/checkpoint access flags remain false for validation/outer/confirmation/test.

## Technical gates

1. Native-volume artifact and 756 overlap/weight exact audits pass.
2. Synthetic tests lock the volume formula, median-per-arm behavior, weight SHA, sampler draws, voxel-KL value/gradient, optimizer and LR grid.
3. Each real-cache batch8 GPU smoke performs two teacher forwards, student forward, DiceCE+KL, backward, clip, step and strict reload; peak<=24GiB and synchronized extrapolation<=45min.
4. Formal has32 epochs,94 steps/752 consumed draws each, four checkpoints, exact reload, peak<=24GiB and elapsed<=45min per arm.

## Fixed validation

- Evaluate only candidate/control global176/184/192/200 on canonical validation108/50 subjects.
- Selection `(candidate Scan, candidate Subject, -epoch)`; no Point/tail selection.
- Candidate-vs-control paired bootstrap seeds `20261060..63`; candidate-vs-archived-KD200 `20261064..67`; 10,000 percentile draws, Scan by108 cases and Subject after within-subject averaging across50 subjects.
- Promotion: selected candidate Scan/Subject strictly exceed same-epoch control and KD200; Point>=101/108; all four CI lower bounds>=-0.003; every provenance/technical/access gate passes.

## Test governance

Only all-gates-passed validation authorizes one frozen reused-internal-test270 evaluation. Otherwise stop and archive. Outer/confirmation have been reclassified into candidate train and cannot be used as independent evidence. No test metric may change sampling, draw count, epoch, LR, threshold or postprocessing.

