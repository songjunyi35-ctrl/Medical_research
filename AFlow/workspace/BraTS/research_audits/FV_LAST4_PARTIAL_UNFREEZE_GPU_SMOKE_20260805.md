# FV last-4 partial-unfreeze: GPU smoke + environment audit (2026-08-05 UTC)

## Scope

This record covers only: (a) re-verification of the frozen engineering gate on a
new host, (b) diagnosis of a torch-version deserialization block, (c) the single
authorized 2-train / 2-validation / 1-epoch GPU smoke, and (d) the formal
fail-closed preflight. It reports no scientific metric and performs no model
selection. The 270-case test was not read, listed, or evaluated.

## 1. Compute availability change

Prior sessions recorded an infrastructure block: MLX control-plane timeouts, no
worker ID, and no `/dev/nvidia0` in the execution environment. On this host the
block does not apply.

- 2x NVIDIA A10, 23,028 MiB each, driver `535.161.08`, CUDA `12.6`.
- Both GPUs idle at 0 MiB / 0% before launch.
- venv torch `2.7.1+cu126`, `torch.cuda.is_available()=True`, device count 2.
- No BraTS trainer or evaluator process was running at session start.

A10 is Ampere (sm_86) and BF16-capable, which is what the preregistration
requires. It is also the same GPU class that ran the FV-LoRA-Aug formal arm
(worker `1009989`), so the smoke and formal runs stay on precedent hardware.

## 2. Frozen-artifact re-verification (no drift)

All nine artifacts recomputed on this host match the engineering gate exactly:

| Artifact | SHA-256 | Match |
|---|---|---|
| human prereg MD | `111968f92c09f35272a5ddd35741c3ad1a5b2803a09f307b2a6993473e156156` | yes |
| machine prereg JSON | `b55e763c4ab422a8b38d49df7fe6dde9e95af8d5e154d485eaef5fdcc97a5f91` | yes |
| helper `sammed3d/partial_unfreeze.py` | `dce2accc8d950ad171d4dc2417dead153ff2b74a78f007725dcb6ca5e5e32f07` | yes |
| CPU preflight | `9b0af7a5cbddd99264d0037bee5e796c5b41254ba1ed1622053afa1dea6a1efa` | yes |
| trainer | `643c1e9124986ac4d057be3779d7eea5ff7b0e99009cea1361feb496f90cc20e` | yes |
| formal launcher | `7034d7e978fdcf248d53de656482b9d96e78c9279bcf5c9a8d0f52847e507d33` | yes |
| smoke launcher | `5205a6b7e368777fe158fa2aa89453dc83af059599cc81f15263e16c8637ac24` | yes |
| helper tests | `7b0913811c00dffd48245e16510d3c479b300f62df498d5d9996375032edc491` | yes |
| runner tests | `14e1a708b1578166695f7f9c5184e2fa6c6afee2ba923932c97aed4c49ba9788` | yes |

Untouched base checkpoint
`/mlx_devbox/users/songjunyi.666/playground/SAM-Med3D-main/ckpt/sam_med3d_turbo.pth`
= `899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c`, matching
the preregistration.

The gate's test counts were **not** inherited as a claim. They were re-run here:
the two focused partial-unfreeze files passed `11/11`, and the broader
PEFT / T1ce-WT / preflight contract set passed `45/45`. The CPU preflight was
re-run and reproduced the exact two optimizer groups with `training_started=false`.

## 3. Engineering block found and resolved

### Symptom

Smoke attempt 1 trained and validated normally, then raised
`_pickle.UnpicklingError` inside `strict_reload_checkpoint` ->
`sammed3d.peft.load_model_with_adapters`.

### Root cause

PyTorch 2.6 changed the default of `torch.load(..., weights_only)` from `False`
to `True`. The trainer stores `args.output_dir` as a `pathlib.PosixPath` in the
checkpoint payload (confirmed by direct inspection: the only such object is at
`/args/output_dir`). `sammed3d/peft.py:370` calls `torch.load` without an
explicit `weights_only`, so the safe unpickler rejects `pathlib.PosixPath` on
torch `2.7.1`. The original A10 worker ran torch < 2.6, where the permissive
default hid this.

This is a deserialization-permission difference only. It cannot alter any tensor
value, hyperparameter, seed, split, or architecture. The trainer's own loader
already passes `weights_only=False` at line 365; only the shared library omitted it.

### Why the obvious source edit was rejected

Adding `weights_only=False` to `sammed3d/peft.py` was implemented, tested, and
then **reverted**. The contract suite caught that `peft.py` is byte-pinned by the
already-completed rank-0 matched arm:
`scripts/preflight_sammed3d_t1ce_rank0_matched.py` requires peft_module SHA-256
`4844b1a086c74737f786399c4c67a2b3c56ae34f35b152e423f63591f226bc63`, and
`scripts/summarize_sammed3d_pair.py` repeats the same constant. Editing the file
broke that fail-closed guard, and rewriting the expected hash to let a run start
is explicitly forbidden. `sammed3d/peft.py` was restored byte-identical to the
pinned value and the rank-0 preflight test passes again.

### Resolution actually used

An explicit, opt-in environment shim,
`workspace/BraTS/env_shims/sitecustomize.py`, registers `pathlib.PosixPath` as a
torch safe global at interpreter startup. It is activated only by passing
`PYTHONPATH=workspace/BraTS/env_shims` to the frozen launchers, which are
themselves unmodified.

Properties:

- zero frozen or pinned files changed; every SHA-256 above still matches;
- strictly safer than `weights_only=False`, because the safe unpickler stays
  enabled and exactly one benign stdlib class is allowlisted;
- empirically minimal: an iterative probe showed `pathlib.PosixPath` is the only
  global required.

The strongest evidence that it is numerically inert: smoke attempt 2 reproduced
attempt 1 bit-for-bit on every metric and on peak memory (see below).

## 4. Authorized GPU smoke result

Command (single authorized smoke, 2 train / 2 validation / 1 epoch):

```
PYTHONPATH=workspace/BraTS/env_shims bash scripts/run_sammed3d_t1ce_wt_partial_unfreeze_smoke.sh
```

| Field | Value |
|---|---|
| completed epochs | 1 |
| train loss / dice | `0.8534855246543884` / `0.22207456082105637` |
| validation loss / dice | `0.9922760128974915` / `0.018977261090185493` |
| train throughput | `1.1658507392484694` case/s |
| validation throughput | `2.2503120262141856` case/s |
| CUDA peak allocated | `2,793,473,024` bytes (2.60 GiB) |
| CUDA peak reserved | `3,395,289,088` bytes (3.16 GiB) |
| selected checkpoint SHA-256 | `34e52fb81146259cc0110cd3b52b553d356dbbf00313af6b1aa05fd20d101fec` |
| `strict_state_load` | true |
| `optimizer_groups_exact` | true |
| `partial_unfreeze_policy_exact` | true |
| `test_access` | false |
| `scientific_result_eligible` | false |

Attempt 1 produced identical `train_dice`, `val_dice`, `train_loss`, `val_loss`,
and identical peak allocated/reserved memory, differing only in throughput
(wall-clock noise). Attempt 1 is retained unmodified at
`ATTEMPT1_FAILED_RELOAD_smoke_fv_partial_last4_disclr10_2train_2val_e1_s20260725`
with an `ATTEMPT_NOTE.json`; nothing was deleted.

Finite-value enforcement is a runtime property of the trainer, not an assumption:
per-optimizer-group finite-gradient checks run before every update
(`ensure_finite_gradients`, raising `"<group> has non-finite gradients"`), plus
non-finite loss and non-finite epoch-metric guards.

Peak usage of 2.60 GiB against 22.5 GiB of A10 memory leaves ample headroom for
the formal arm.

**The smoke metrics are engineering signals only.** Two training cases and two
validation cases carry no scientific meaning, and per the preregistration they
were not and must not be used to alter any configuration.

## 5. Formal fail-closed preflight

`bash scripts/run_sammed3d_t1ce_wt_partial_unfreeze_formal.sh preflight` exited
`0` with `preflight_passed: true` and `output_created: false`, reproducing the
frozen contract exactly:

- trainable / total = `35,951,920` / `100,509,232`;
- image encoder depth 12, trainable blocks `[8,9,10,11]`, frozen `[0..7]`,
  `non_block_parameters_trainable=false`, `update=base_weight_partial_unfreeze`;
- `lora_enabled=false`, `target_embedding_enabled=false`;
- group `image_encoder_blocks_8_11`: `28,367,616` params, 60 tensors, LR `1e-5`,
  WD `0.1`, fingerprint `7a4430f6a2138824f21f72d87b8b60615d2b1a17e36dfe9ddde4675cdf06c4e7`;
- group `prompt_encoder_and_mask_decoder`: `7,584,304` params, 134 tensors,
  LR `1e-4`, WD `0.1`, fingerprint `df886afa019777db6838916722cf1908d792a9137549eeb525c6190cfbc22afa`;
- `preregistration_sha256=b55e763c4ab422a8b38d49df7fe6dde9e95af8d5e154d485eaef5fdcc97a5f91`,
  `test_access=false`, `train_image_encoder=true`.

## 6. Documented deviation: process supervision

The handoff specifies running the formal arm in the foreground. The available
execution harness caps a foreground call at 10 minutes, while this arm is a
multi-hour run, so the launcher was started as a supervised background process
with active monitoring instead.

What the handoff was protecting against is preserved: **no watcher is attached**,
no automated downstream action is armed, and in particular nothing can trigger a
test evaluation. The `run` subcommand still re-ran the complete fail-closed
preflight in-process immediately before `exec`-ing the scientific process.
Monitoring is limited to process liveness, finite metrics, memory/throughput,
atomic epoch completion, and the frozen early-stop rule. No configuration will be
changed in response to the intermediate validation curve.

## 7. Boundaries

- No scientific metric is reported here; the partial-unfreeze arm has no result yet.
- The 270-case test remains permanently frozen and was not accessed in any form.
- No threshold, TTA, morphology, connected-component, or postprocess search was run.
- No hyperparameter, seed, unfreeze depth, initialization, augmentation, epoch cap,
  or patience was altered.
- No `git reset`, `git clean`, or bulk deletion was performed.
