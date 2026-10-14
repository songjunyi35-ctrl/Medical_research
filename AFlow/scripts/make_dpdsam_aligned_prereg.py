#!/usr/bin/env python
"""Emit the human + machine preregistration for the DPD-SAM-aligned arm.

Both documents are generated from one source so their bound hashes cannot drift
apart. Run only after both preprocessing caches are complete.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "workspace/BraTS/cache/t1ce_wt_n4_128"
OUT_MD = ROOT / "workspace/BraTS/research_audits/SAMMED3D_OFFICIAL_FULLUNFREEZE_PREREG_20260805.md"
OUT_JSON = ROOT / "workspace/BraTS/training_protocols/sammed3d_official_fullunfreeze_prereg_20260805.json"

RUN_IDENTITY = "sammed3d_official_fullunfreeze_ddp2_train972_val108_e200_s20260805"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cache(split: str) -> dict:
    index = json.loads((CACHE / split / "index.json").read_text())
    if index["test_access"] is not False:
        raise SystemExit(f"{split} cache does not declare test_access=false")
    return index


def main() -> None:
    train = load_cache("train")
    val = load_cache("validation")
    if train["count"] != 972 or val["count"] != 108:
        raise SystemExit(f"unexpected cache counts {train['count']}/{val['count']}")
    if not train["preprocessing"]["n4_bias_field_correction"]:
        raise SystemExit("train cache was built without N4")
    if not val["preprocessing"]["n4_bias_field_correction"]:
        raise SystemExit("validation cache was built without N4")

    runner = ROOT / "scripts/train_sammed3d_t1ce_wt_dpdsam_aligned.py"
    builder = ROOT / "scripts/build_t1ce_wt_cache.py"
    dataset = ROOT / "sammed3d/t1ce_wt_cached.py"
    launcher = ROOT / "scripts/run_sammed3d_t1ce_wt_dpdsam_aligned.sh"
    tests = ROOT / "tests/test_t1ce_wt_cached.py"

    machine = {
        "schema": "aflow.sammed3d.t1ce_wt_dpdsam_aligned_preregistration",
        "schema_version": 1,
        "run_identity": RUN_IDENTITY,
        "created_utc": "2026-08-05",
        "status": "AWAITING_USER_SIGNOFF",
        "objective": (
            "Follow SAM-Med3D's own published training recipe -- full image-encoder "
            "unfreeze with the encoder learning rate above the heads, plus its "
            "epoch budget and decay schedule -- on the frozen 972/108 "
            "clean-development split, to see how far the T1ce->WT arm moves toward "
            "the SAM-Med3D BraTS 2024 DSC of 74.1 +/- 1.5 that DPD-SAM reports. "
            "That baseline was most plausibly produced with this same official "
            "recipe, which is why it is the recipe being followed here rather than "
            "DPD-SAM's own frozen-backbone method."
        ),
        "design_type": "uncontrolled recipe-alignment arm",
        "matched_comparison_valid": False,
        "confounded_factors": [
            "FULL image-encoder unfreeze: 35,951,920 -> 100,509,232 trainable",
            "discriminative-LR direction INVERTED to SAM-Med3D's own (encoder above heads)",
            "N4 bias field correction added to preprocessing",
            "max epochs 50 -> 200 with official MultiStepLR([120,180], gamma 0.1)",
            "effective batch 2 -> 12 (per-GPU 2 x 2 GPUs x 3 accumulation)",
            "single-GPU -> 2-GPU data parallelism with explicit gradient all-reduce",
        ],
        "attribution_rule": (
            "Several factors move together by design. No outcome may be "
            "attributed to any single factor, and this arm may not be presented "
            "as a controlled contrast against the rank-0, LoRA, or last-4 "
            "partial-unfreeze arms."
        ),
        "task_definition": {
            "modality": "t1ce",
            "target": "official WT",
            "foreground_raw_labels": [1, 2, 3],
            "excluded_raw_labels": [4],
            "geometry": "whole-volume resize to 128^3",
            "clicks": "3 accumulated corrective clicks, recurrent low-resolution mask",
            "threshold": "raw logit > 0 (sigmoid 0.5), no postprocessing",
        },
        "data": {
            "train": {
                "manifest": train["manifest"],
                "manifest_sha256": train["manifest_sha256"],
                "rows": train["count"],
                "subjects": 444,
                "cache_images_sha256": train["images_sha256"],
                "cache_labels_sha256": train["labels_sha256"],
            },
            "validation": {
                "manifest": val["manifest"],
                "manifest_sha256": val["manifest_sha256"],
                "rows": val["count"],
                "subjects": 50,
                "cache_images_sha256": val["images_sha256"],
                "cache_labels_sha256": val["labels_sha256"],
            },
            "test": {
                "status": "PERMANENTLY FROZEN; not read, not evaluated, not used for selection",
            },
        },
        "preprocessing": train["preprocessing"],
        "initialization": {
            "checkpoint": "artifacts/checkpoints/sam_med3d_turbo.pth",
            "checkpoint_sha256": "899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c",
            "note": "untouched base; not initialized from any prior arm",
        },
        "model": {
            "model_type": "vit_b_ori",
            "lora_enabled": False,
            "target_token_enabled": False,
            "trainable_policy": "FULL unfreeze: entire image encoder + prompt encoder + mask decoder",
            "full_image_encoder_unfreeze": True,
            "prohibition_lifted": {
                "rule": "full image-encoder unfreeze",
                "lifted_by": "operator",
                "lifted_utc": "2026-08-05",
                "evidence": [
                    "SAM-Med3D train.py trains image_encoder at full LR, heads at 0.1x",
                    "project results monotone in encoder adaptation: 0.4785 frozen -> 0.5214 LoRA -> 0.5072 by epoch 7 for blocks 8-11",
                    "the prohibition carried no supporting experiment and was grouped with test-hacking techniques",
                ],
            },
            "trainable_parameters": 100509232,
            "total_parameters": 100509232,
        },
        "optimization": {
            "type": "AdamW",
            "groups": [
                {"name": "image_encoder_full", "lr": 1e-4,
                 "weight_decay": 0.1, "parameter_numel": 92924928},
                {"name": "prompt_encoder_and_mask_decoder", "lr": 1e-5,
                 "weight_decay": 0.1, "parameter_numel": 7584304},
            ],
            "precision": "bf16",
            "per_gpu_batch_size": 2,
            "world_size": 2,
            "gradient_accumulation_steps": 3,
            "effective_batch_size": 12,
            "lr_scheduler": {"type": "MultiStepLR", "milestones": [120, 180], "gamma": 0.1},
            "lr_rationale": (
                "SAM-Med3D's 8e-4 belongs to ITS effective batch of 240 "
                "(batch 12 x accumulation 20), which on 972 cases would give only "
                "~4 optimizer steps/epoch. Absolute LR is not transplantable; "
                "1e-4/1e-5 is a conservative sqrt-scaling to effective batch 12. "
                "Only the 10:1 direction and the decay schedule are inherited."
            ),
            "memory_measured": "11.35 GiB peak at per-GPU batch 2; batch 4 OOMs on a 23 GB A10",
            "max_epochs": 200,
            "min_epochs_before_stop": 40,
            "patience_complete_epochs": 40,
            "train_seed": 20260805,
            "validation_seed": 20260723,
        },
        "selection_rule": {
            "metric": "validation final-click scan-macro Dice (val_dice)",
            "rule": "strict argmax over completed epochs; ties keep the earlier epoch",
            "forbidden": "no selection on train metrics, no test access",
        },
        "readout": {
            "primary": "108-case clean-validation subject-balanced C3 via independent strict reload",
            "reference_lora_subject_balanced_c3": 0.5214101623252374,
            "reference_rank0_subject_balanced_c3": 0.4785050270,
            "interpretation_limit": (
                "Reaching any particular value on this validation set does NOT "
                "reproduce the DPD-SAM-reported 74.1: that number was measured by "
                "different authors, on their own split, under a prompt protocol "
                "their paper does not disclose."
            ),
            "evaluation_preprocessing_caveat": (
                "This arm is trained on N4-corrected inputs, so it must also be "
                "evaluated on N4-corrected inputs to avoid a train/eval "
                "preprocessing mismatch. The LoRA and rank-0 reference values were "
                "produced WITHOUT N4. The reference numbers therefore sit on "
                "different model inputs, which is an additional reason this is a "
                "descriptive context comparison and not a controlled contrast."
            ),
            "evaluation_protocol": (
                "Same 108 cases / 50 subjects, same case-seeded PCG64 click "
                "sampling and validation seed 20260723, same subject-balanced "
                "aggregation as the existing paired evaluator; only the image "
                "source changes (N4 cache instead of on-the-fly preprocessing)."
            ),
        },
        "prohibitions": [
            "no 270-case test access",
            "no threshold / TTA / morphology / connected-component / postprocess search",
            "no encoder change beyond the full unfreeze authorized for this arm",
            "no mid-run change of LR, unfreeze depth, augmentation, epoch cap or patience",
        ],
        "code_binding": {
            "runner_sha256": sha256_file(runner),
            "cache_builder_sha256": sha256_file(builder),
            "cached_dataset_sha256": sha256_file(dataset),
            "launcher_sha256": sha256_file(launcher),
            "tests_sha256": sha256_file(tests),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(machine, indent=2, sort_keys=True) + "\n")

    o = machine["optimization"]
    m = machine["model"]
    g0, g1 = o["groups"][0], o["groups"][1]
    lifted = m["prohibition_lifted"]
    rd = machine["readout"]
    md = f"""# SAM-Med3D official-recipe full-unfreeze arm preregistration (2026-08-05 UTC)

Run identity: `{RUN_IDENTITY}`

Status: **{machine["status"]}** -- no formal training may start until this is approved.

## 1. What this arm is, and is not

{machine["objective"]}

It is **not** a controlled experiment. The following change together, by intent:

{chr(10).join('- ' + f for f in machine['confounded_factors'])}

{machine["attribution_rule"]}

It is also **not** a reproduction of DPD-SAM. {rd["interpretation_limit"]}

## 2. Unchanged task definition

T1ce single channel, binary official `WT={{1,2,3}}` with RC `{{4}}` excluded,
whole-volume `128^3`, 3 accumulated corrective clicks with the recurrent
low-resolution mask, raw logit `> 0`, validation seed `{o["validation_seed"]}`.

## 3. Data

| Split | Rows | Subjects | Manifest SHA-256 |
|---|---:|---:|---|
| train | {train['count']} | 444 | `{train['manifest_sha256']}` |
| validation | {val['count']} | 50 | `{val['manifest_sha256']}` |

The 270-case test stays **permanently frozen**: not read, not evaluated, not
used for any selection.

## 4. Preprocessing and cache binding

Deterministic preprocessing is precomputed once and bound by content hash. The
cache boundary sits after the deterministic chain and before the stochastic
augmentation, so per-epoch randomness is preserved exactly. With N4 disabled the
cache was verified bit-identical to the live pipeline before the N4 caches were built.

| Cache | images.f32 SHA-256 | labels.u8 SHA-256 |
|---|---|---|
| train | `{train['images_sha256']}` | `{train['labels_sha256']}` |
| validation | `{val['images_sha256']}` | `{val['labels_sha256']}` |

N4 bias field correction: SimpleITK, 4x50 iterations, nonzero brain mask, shrink
factor {train['preprocessing']['n4_shrink_factor']}, applied to the raw canonical
volume before normalization. Skull stripping is not applied because BraTS 2024
volumes are already skull-stripped.

## 5. Model and optimization

From the untouched base checkpoint
`{machine["initialization"]["checkpoint_sha256"]}`; LoRA and target token disabled.

**Trainable policy: {m["trainable_policy"]}** --
`{m["trainable_parameters"]:,} / {m["total_parameters"]:,}`, i.e. every parameter.

| Group | Params | LR | WD |
|---|---:|---:|---:|
| `{g0["name"]}` | {g0["parameter_numel"]:,} | `{g0["lr"]}` | `{g0["weight_decay"]}` |
| `{g1["name"]}` | {g1["parameter_numel"]:,} | `{g1["lr"]}` | `{g1["weight_decay"]}` |

The encoder LR sits **above** the head LR, matching SAM-Med3D's own `train.py`
and inverting the direction used by the matched arms.

BF16, per-GPU batch {o["per_gpu_batch_size"]} x {o["world_size"]} GPUs x
{o["gradient_accumulation_steps"]} accumulation steps = **effective batch
{o["effective_batch_size"]}**. Max {o["max_epochs"]} epochs with
{o["lr_scheduler"]["type"]}(milestones={o["lr_scheduler"]["milestones"]},
gamma={o["lr_scheduler"]["gamma"]}). No performance early stop before epoch
{o["min_epochs_before_stop"]}, then stop only after
{o["patience_complete_epochs"]} consecutive complete epochs without a strict new
validation high. Train seed `{o["train_seed"]}`.

Measured resource envelope: {o["memory_measured"]}.

### 5.1 On the learning-rate magnitude

{o["lr_rationale"]}

### 5.2 A standing prohibition was lifted for this arm

The research protocol forbade **{lifted["rule"]}**. It was lifted by the
{lifted["lifted_by"]} on {lifted["lifted_utc"]}, on this evidence:

{chr(10).join('- ' + e for e in lifted["evidence"])}

Recorded here so the change of regime is explicit and auditable, not silent.

## 6. Selection and readout

Selected checkpoint = strict argmax of validation final-click scan-macro Dice
over completed epochs, ties keeping the earlier epoch. Afterwards exactly one
independent strict-reload evaluation on the same 108 cases / 50 subjects
produces the subject-balanced C3 readout.

{rd["evaluation_protocol"]}

{rd["evaluation_preprocessing_caveat"]}

Descriptive reference points on the same validation set: LoRA
`{rd["reference_lora_subject_balanced_c3"]}`, rank-0
`{rd["reference_rank0_subject_balanced_c3"]}`. These are context, not a
controlled contrast.

## 7. Prohibitions still in force

{chr(10).join('- ' + p for p in machine["prohibitions"])}

## 8. Code binding

| Artifact | SHA-256 |
|---|---|
| runner | `{machine['code_binding']['runner_sha256']}` |
| cache builder | `{machine['code_binding']['cache_builder_sha256']}` |
| cached dataset | `{machine['code_binding']['cached_dataset_sha256']}` |
| launcher | `{machine['code_binding']['launcher_sha256']}` |
| tests | `{machine['code_binding']['tests_sha256']}` |

Machine preregistration: `{OUT_JSON.relative_to(ROOT)}`
"""
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(md)

    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"  machine prereg SHA-256: {sha256_file(OUT_JSON)}")
    print(f"  human prereg   SHA-256: {sha256_file(OUT_MD)}")


if __name__ == "__main__":
    main()
