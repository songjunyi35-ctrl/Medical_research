# Amendment 001: 用互补覆盖替代跨排列滚动队列

Status: frozen after the first synthetic test failure and before GPU access, model construction, training, or validation/test access.

## Failure observed

The first synthetic suite completed 12/14 tests. The deterministic, allocation, range, control-equivalence, drop-last, input-rejection, and contract tests passed. Two tests failed because the original persistent queue did not satisfy the preregistered claim that every adjacent pair of epochs covers all972 cases.

The first pairs that jointly completed one permutation passed, but a later pair straddled two different random permutations and its coverage union contained only880/972 unique indices. No image, model, GPU, validation, outer, confirmation, or test data were accessed. The failure is an implementation/design failure before formal execution, not a segmentation result.

## Frozen correction

Keep every allocation and training control unchanged:756 requested,752 consumed,680 coverage,72 consumed native-volume replay,4 dropped replay,94 steps, same three seed offsets and both arms.

Replace only the coverage-state transition:

1. Epoch1: generate one `randperm(972)` and select its first680 unique indices.
2. Every later epoch: take all292 population indices absent from the previous epoch's coverage set, then sample388 unique indices without replacement from the previous680 coverage indices using the coverage generator.
3. Deterministically shuffle these680 coverage indices before combining them with replay.

This construction gives exactly680 unique coverage cases per epoch and makes every adjacent two-epoch union exactly972 by construction. It is a one-step complement schedule, not a long FIFO queue and not a claim that all 3D data diversity is captured.

## Updated gates

- Every epoch coverage count and unique count are both680.
- Every adjacent two-epoch coverage union is972.
- The retained subset has388 unique indices and the complement has292.
- Same seed reconstructs the exact first32 epochs; a different seed changes them.
- All earlier replay, drop-last, control-equivalence, document, JSON/checkpoint, memory, runtime, reload, selection, confidence interval, access, and test gates remain unchanged.

The original `SAMPLER_SPEC.md` remains preserved as the failed design record. This amendment supersedes only its coverage queue steps 1–2 and the corresponding permutation gate. Validation metrics may not motivate another sampler change.

Access flags at amendment freeze: validation=false, outer=false, confirmation=false, test270=false.
