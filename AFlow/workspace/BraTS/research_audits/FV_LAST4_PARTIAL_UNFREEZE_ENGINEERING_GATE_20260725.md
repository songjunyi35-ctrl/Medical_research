# FV last-four partial-unfreeze engineering gate

Status: `ENGINEERING_PASS; GPU_SMOKE_BLOCKED_BY_CONTROL_PLANE; FORMAL_NOT_STARTED`

Date: 2026-07-25 CST

## Scope

This gate covers only the frozen 972-row training and 108-row / 50-subject
clean-validation development protocol. No test or sealed data, manifest, model
evaluation, or report was opened or enumerated. The 270-case test remains
forbidden.

## Scientific trigger

The selected LoRA and matched rank-0 checkpoints were independently reloaded
on the same CPU runtime. The preregistered subject-balanced click-3 contrast
was LoRA minus rank-0 `+0.0429051353`, with a 10,000-resample paired
50-subject bootstrap interval `[+0.0258629117, +0.0630796008]`.

This is a conditional, single-seed development contrast between two
validation-selected models. Click 1 has the same GT-defined sampling, while
clicks 2 and 3 follow each model's own XOR error. The final-click result is
therefore a closed-loop model-plus-corrective-trajectory contrast, not an
identical-prompt network-only effect.

## Frozen arm

- Initialization: untouched `sam_med3d_turbo.pth`, SHA-256
  `899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c`.
- LoRA and target token: disabled.
- Trainable image encoder: base-weight blocks 8--11 only,
  `28,367,616` parameters / 60 tensors, AdamW LR `1e-5`, WD `0.1`.
- Trainable heads: prompt encoder plus mask decoder,
  `7,584,304` parameters / 134 tensors, AdamW LR `1e-4`, WD `0.1`.
- Total trainable/model: `35,951,920 / 100,509,232` (`35.77%`).
- Training: batch 1, accumulation 2, BF16, workers 2, seed `20260725`;
  validation/click seed `20260723`.
- Selection: trainer validation scan-macro click-3 strict argmax, earliest tie.
- Budget: maximum 50 complete epochs; no stop before epoch 20; thereafter stop
  only after 20 consecutive complete epochs without a strict new high.
- Formal restart/resume, test access, full-encoder unfreezing, sweeps,
  thresholding, TTA, morphology, and postprocessing remain forbidden.

The policy fingerprints are:

- blocks 8--11:
  `7a4430f6a2138824f21f72d87b8b60615d2b1a17e36dfe9ddde4675cdf06c4e7`;
- prompt plus mask:
  `df886afa019777db6838916722cf1908d792a9137549eeb525c6190cfbc22afa`.

## Fail-closed implementation

The independent runner does not change the anchored generic trainer or the
evaluator. It verifies the machine preregistration, parent MD, manifests,
untouched checkpoint, adapter-disabled architecture, exact trainable-policy
record, optimizer-group union/disjointness/count/LR/WD, generic-trainer source
hash, empty output path, finite per-group gradients, and strict checkpoint
reload.

A final protocol audit found that the first runner version seeded only before
model construction, while the matched arms reseed after model/optimizer setup.
Before any GPU execution, this was corrected and bound into the machine
preregistration as `rng_reseed_after_model_setup=true`. This makes shuffling,
augmentation, and click RNG start from the same post-construction seed state
as the matched arms.

Final source/protocol SHA-256 values:

- human preregistration:
  `111968f92c09f35272a5ddd35741c3ad1a5b2803a09f307b2a6993473e156156`;
- machine preregistration:
  `b55e763c4ab422a8b38d49df7fe6dde9e95af8d5e154d485eaef5fdcc97a5f91`;
- partial-unfreeze helper:
  `dce2accc8d950ad171d4dc2417dead153ff2b74a78f007725dcb6ca5e5e32f07`;
- CPU preflight:
  `9b0af7a5cbddd99264d0037bee5e796c5b41254ba1ed1622053afa1dea6a1efa`;
- independent trainer:
  `643c1e9124986ac4d057be3779d7eea5ff7b0e99009cea1361feb496f90cc20e`;
- formal foreground launcher:
  `7034d7e978fdcf248d53de656482b9d96e78c9279bcf5c9a8d0f52847e507d33`;
- smoke foreground launcher:
  `5205a6b7e368777fe158fa2aa89453dc83af059599cc81f15263e16c8637ac24`;
- helper tests:
  `7b0913811c00dffd48245e16510d3c479b300f62df498d5d9996375032edc491`;
- runner tests:
  `14e1a708b1578166695f7f9c5184e2fa6c6afee2ba923932c97aed4c49ba9788`.

After the RNG correction, the focused regression set passed `24/24`. The
broader CPU/synthetic PEFT, T1ce-WT wrapper/preprocessing, and
partial-unfreeze contract suite then passed `41/41`. A full CPU preflight also passed and reported
`output_created=false`, the exact two optimizer groups, and
`train_image_encoder=true`.

## GPU smoke infrastructure block

The authorized smoke is exactly two training rows, two validation rows, and
one epoch. It is explicitly non-scientific, non-resumable, and separate from
the formal output.

On 2026-07-25 CST, two default-workspace A10 requests timed out in the MLX
control plane before returning a worker ID. A read-only worker listing after
the first request showed no worker. The control plane later also failed its
worker-list endpoint. One controlled Arnold-backend retry ended explicitly
with:

`launch worker error: failed to get current devbox volume usage`

caused by an HTTPS context deadline. No trainer output appeared. Both the
smoke and formal output paths remain absent. This is an infrastructure block,
not a model or training failure.

## Safe continuation

1. Once the MLX control plane recovers, list workers first and confirm there is
   no orphan allocation.
2. Run only
   `bash scripts/run_sammed3d_t1ce_wt_partial_unfreeze_smoke.sh` on one
   BF16-capable GPU worker.
3. Require one complete epoch, finite loss/gradients/metrics, resource
   measurements, checkpoint metadata, and strict save/reload success.
4. Only after the smoke passes, run the formal launcher in the worker
   foreground: first `preflight`, then `run`.
5. Do not bypass the smoke, do not resume or choose among formal attempts, and
   do not access the 270-case test.
