# Pre-implementation amendment 001: run a new cache-matched 756 control

Status: frozen before code, synthetic tests, GPU smoke, formal, and validation.

## Reason

The archived successful 756-case voxel-KD used the newer variable-resolution cache index and computed sampling weights from source-grid WT volumes. The existing 972 cache is voxel-identical for all overlapping images/labels but its older index exposes only resized-128³ positive volumes. Reusing archived KD as the only causal control would therefore change both eligible case population and the sampling-weight volume definition.

## Superseding decision

Run two new arms from the same old 972 cache and the same new runner:

- `data756_control`: restrict the 972 cache to the frozen historical 756-case manifest.
- `data972_candidate`: expose the full frozen 972-case manifest.

Both arms compute weights by `BrainCropDataset.sampling_weights()` from their selected cached 128³ labels, use `WeightedRandomSampler(..., num_samples=756, replacement=True, Generator(20260846))`, and consume 752 draws / 94 optimizer steps per epoch. Thus the only arm difference is the eligible case pool and its induced population-normalized weight vector. The weight changes are part of the data-population intervention, not a separately tuned hyperparameter.

Both arms independently repeat the exact same history168 initialization, two frozen teachers, KD formula, RNG reset, optimizer/LR, augmentation, epoch grid, threshold, and deployment path. Run on separate GPUs concurrently when available. Each arm must satisfy peak allocated `<=24GiB` and elapsed `<=45min`.

Primary causal promotion compares selected `data972_candidate` to same-epoch `data756_control`. The archived KD200 remains a required secondary benchmark. Candidate must strictly exceed both comparators in Scan and Subject, retain Point at least101/108, and pass the four paired CI safety gates. Selection still uses candidate Scan, then Subject, then earlier epoch only.

Formal output now contains eight checkpoints and two TRAIN_DONE reports; expected incremental disk is below200MB. A pairing/provenance finalizer must verify identical initial model state, optimizer contract, teacher hashes, fixed draw/step counts, access flags and strict reload for both arms. Exact batch case identity is neither expected nor required because the arm intervention is the eligible data population.

