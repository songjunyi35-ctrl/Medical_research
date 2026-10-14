# Amendment 002: 新seed与历史校验器的兼容层

Status: frozen after the first remote smoke command stopped and before model construction, optimizer step, output creation, validation, or test access.

## Failure observed

Both remote smoke processes stopped inside the inherited `validate_frozen_inputs` function because the historical 972 runner hard-codes `args.seed == 20260846`. This method preregistered seed`20260884`. The stop happened before dataset/model construction and before either output directory was created. Both GPUs remained free of a training result.

This is a wrapper contract incompatibility. It does not justify changing the preregistered sampler seed and reveals no loss, memory, runtime, validation, outer, confirmation, or test metric.

## Frozen compatibility fix

The new entry point will wrap only the inherited read-only input validator:

1. Require incoming `args.seed == 20260884` and save it.
2. Temporarily present historical seed`20260846` while calling the inherited validator, allowing it to check every historical SHA, batch, draw, epoch, KD, manifest, cache, checkpoint, document, count, and subject gate unchanged.
3. Restore `args.seed` to`20260884` in a `finally` block before the validator returns or raises.
4. Assert after the call that `args.seed` is still`20260884`.

The model construction, DataLoader generator, candidate coverage/replay/order generators, Python/NumPy/PyTorch/CUDA RNG reset, augmentation, checkpoint metadata, and report must all see`20260884`. No historical script is edited.

## New synthetic gates

- The wrapper rejects an incoming seed other than`20260884`.
- A stub inherited validator observes exactly`20260846`.
- After success and after an injected exception, the caller observes exactly`20260884`.
- The real sampler deterministic hashes remain based on`20260884`.

All prior Amendment 001, sampler, training, promotion, access, and stop gates remain unchanged.

Access flags at amendment freeze: model_constructed=false, optimizer_step=false, validation=false, outer=false, confirmation=false, test270=false.
