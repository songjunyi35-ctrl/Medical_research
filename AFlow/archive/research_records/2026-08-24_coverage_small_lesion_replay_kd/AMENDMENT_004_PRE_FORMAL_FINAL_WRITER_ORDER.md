# Amendment 004: 最终JSON绕过旧wrapper的二次文档覆盖

Status: frozen after V3 smoke audit and before formal training or validation/test access.

## Audit finding

V3 added a complete top-level `sampling_contract`, and both compute/sampler gates passed. The contract contains the current arm strategy, native artifact/weight hashes, seed offsets, counts, and candidate epoch audit.

However, the final JSON writer was nested inside the inherited native wrapper. The current patch first installed the complete `method_documents`, then the older wrapper ran again and replaced that dictionary with its smaller historical document set. The V3 checkpoints, which are rewritten after the native checkpoint wrapper, contain all current hashes; only top-level JSON lost the current machine preregistration and Amendments001–003 hashes.

No loss, gradient, memory, runtime, sample index, checkpoint tensor, validation, outer, confirmation, or test metric was affected.

## Frozen fix

- Capture the original lowest-level atomic JSON writer at module import, before any native wrapper is installed.
- The current method's final writer calls that original writer directly on `patch_contract(value)`.
- `sampler_contract()` itself must include the native weight source/formula/dtype, artifact SHA, builder SHA, and native-weight SHA, so it does not rely on a later historical patch.
- Keep the checkpoint rewrite path unchanged because its final payload already contains the complete current contract and document set.

## Verification

Rerun synthetic tests and both V4 smokes. Require top-level and checkpoint evidence to include:

- current IDEA, Markdown preregistration, machine preregistration, sampler spec, Amendments001–004, native artifact and builder hashes;
- actual arm strategy/counts/seeds and candidate epoch audit;
- unchanged all-gates pass, memory, extrapolation, and strict reload.

Access flags at amendment freeze: smoke=true, formal=false, validation=false, outer=false, confirmation=false, test270=false.
