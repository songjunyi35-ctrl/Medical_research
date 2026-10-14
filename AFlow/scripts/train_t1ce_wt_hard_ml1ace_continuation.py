#!/usr/bin/env python3
"""Paired hard-mL1-ACE continuation from frozen voxel-KD200."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator  # noqa: E402
from sammed3d.t1ce_wt_hard_ml1ace import (  # noqa: E402
    BUCKETIZE_RIGHT,
    ML1ACE_WEIGHT,
    NUM_BINS,
    foreground_hard_ml1ace,
)
from sammed3d.t1ce_wt_variable_resolution import (  # noqa: E402
    VariableResolutionBrainCropDataset,
)
from scripts.train_t1ce_wt_capg3d import (  # noqa: E402
    augment_batch,
    case_view_metadata,
    dice_per_case,
    read_case_ids,
    sha256_file,
    subject_id,
    write_json_atomic,
)
from scripts.train_t1ce_wt_plain_iad_kas_continuation import (  # noqa: E402
    _architecture,
    _plain_reload_difference,
    _supervised_loss,
    cuda_rng_sha256,
    state_sha256,
    tensor_raw_sha256,
    text_lines_sha256,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_hard_ml1ace_continuation"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_hard_ml1ace_checkpoint"
SMOKE_SCHEMA = "aflow.sammed3d.t1ce_wt_hard_ml1ace_gpu_smoke"
IMPLEMENTATION_RECEIPT_SCHEMA = "aflow.sammed3d.t1ce_wt_hard_ml1ace_implementation_receipt"

KD200_SHA256 = "56f54898ded11a0208e75f8000d57d4d053e80d6dedd760402730ad345c111b7"
TRAIN_MANIFEST_SHA256 = "a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16"
TRAIN_CACHE_INDEX_SHA256 = "d26b823a4715d9f3676d75cddabcf630180d48807352c23042be7faf240b622d"
SAMPLING_WEIGHTS_SHA256 = "757e463d8e60f96ef6429613f6783fb3aa1014b580932959f9ed354d37072124"
CONTROL_TRAINING_SHA256 = "ce987642413fbf86ccbb45ec263851b8a12cfbec6f21d2325521529b583c6392"
CONTROL_PAIRING_SHA256 = "cc8e21276f3e5a29ff472aa68e2ddb0e11b8b56e497a4f1db6a084806cd83856"
PAIRING_DEPENDENCY_SHA256 = "289168d96d3494449d9b822513fd759c69aff1b4d1c502fc59b114c1cde62ea8"
PREREG_SHA256 = "c279d8f17aaa935a626addfe8970acd8c8388708b285d14bb279ddd25d5bed08"
AMENDMENT_SHA256 = "48293660994732ef88ddf8adc15e465f95689946a6f17f312b66eddc1de53540"

SAVE_GLOBAL_EPOCHS = (208, 216, 224, 232)
START_GLOBAL_EPOCH = 200
CONTINUATION_EPOCHS = 32
FIXED_LR = 1e-5
GPU_MEMORY_STOP_BYTES = 24 * 1024**3
EXPECTED_DEPLOY_PARAMETERS = 4_700_897


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _require_frozen_inputs(args: argparse.Namespace) -> None:
    for path, expected, label in (
        (args.preregistration, PREREG_SHA256, "preregistration"),
        (args.preregistration_amendment, AMENDMENT_SHA256, "amendment"),
        (args.initial_checkpoint, KD200_SHA256, "KD200 start"),
        (args.train_case_ids, TRAIN_MANIFEST_SHA256, "train manifest"),
        (args.cache / "index.json", TRAIN_CACHE_INDEX_SHA256, "train cache index"),
        (args.control_training_report, CONTROL_TRAINING_SHA256, "control TRAIN_DONE"),
        (args.control_pairing_receipt, CONTROL_PAIRING_SHA256, "control pairing receipt"),
        (ROOT / "scripts/train_t1ce_wt_plain_iad_kas_continuation.py", PAIRING_DEPENDENCY_SHA256, "paired trainer dependency"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"{label} SHA-256 mismatch")
    if args.seed != 20260860 or args.batch_size != 8 or args.num_workers != 4:
        raise ValueError("frozen hACE protocol requires seed20260860,batch8,workers4")
    if args.start_global_epoch != START_GLOBAL_EPOCH or args.continuation_epochs != CONTINUATION_EPOCHS:
        raise ValueError("frozen hACE continuation is global201-232")
    if args.lr != FIXED_LR or args.weight_decay != 1e-5 or args.max_shift != 8:
        raise ValueError("hACE optimizer/augmentation differs from preregistration")


def _validate_implementation_receipt(path: Path | None) -> str:
    if path is None:
        raise ValueError("formal hACE training requires implementation receipt")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != IMPLEMENTATION_RECEIPT_SCHEMA:
        raise ValueError("unexpected hACE implementation receipt schema")
    for field in ("synthetic_all_gates_passed", "gpu_smoke_all_gates_passed", "all_gates_passed"):
        if value.get(field) is not True:
            raise ValueError(f"implementation receipt {field} != true")
    for field in ("validation_access", "outer_selector_access", "confirmation_access", "test_access"):
        if value.get(field) is not False:
            raise ValueError(f"implementation receipt {field} must be false")
    if value.get("implementation", {}).get("trainer_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("implementation receipt trainer SHA mismatch")
    return sha256_file(path)


def _load_control_pairing(args: argparse.Namespace) -> dict[str, Any]:
    report = json.loads(args.control_training_report.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or report.get("arm") != "control":
        raise ValueError("matched control report is not complete")
    return report["pairing"]


def _load_model(args: argparse.Namespace, device: torch.device) -> AutomaticPromptLocator:
    checkpoint = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    model = AutomaticPromptLocator(
        center_aware=False,
        init_filters=int(architecture["init_filters"]),
        dropout=float(architecture["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if sum(value.numel() for value in model.state_dict().values()) != EXPECTED_DEPLOY_PARAMETERS:
        raise RuntimeError("unexpected hACE deploy parameter count")
    return model


def _make_optimizer(model: AutomaticPromptLocator, args: argparse.Namespace):
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
        maximize=False,
        foreach=False,
        fused=False,
        capturable=False,
    )
    return optimizer, parameters


def _make_data(args: argparse.Namespace):
    case_ids = read_case_ids(args.train_case_ids)
    assert case_ids is not None
    data = VariableResolutionBrainCropDataset(
        args.cache,
        expected_resolution=128,
        expected_manifest_sha256=TRAIN_MANIFEST_SHA256,
        case_ids=case_ids,
    )
    if len(data) != 756 or data.index.get("test_access") is not False:
        raise ValueError("unexpected hACE train cache")
    weights = data.sampling_weights()
    if tensor_raw_sha256(weights) != SAMPLING_WEIGHTS_SHA256:
        raise ValueError("sampling weights SHA mismatch")
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            weights, num_samples=len(data), replacement=True, generator=generator
        ),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    if len(loader) != 94:
        raise RuntimeError("hACE loader must have94 steps/epoch")
    return data, weights, loader


def _first_pairing(batch, raw_image, raw_label, image, label) -> dict[str, Any]:
    case_ids = [str(value) for value in batch["case_id"]]
    return {
        "initial_student_state_sha256": "",
        "first_batch_case_ids": case_ids,
        "first_batch_case_ids_sha256": text_lines_sha256(case_ids),
        "first_batch_raw_image_sha256": tensor_raw_sha256(raw_image),
        "first_batch_raw_label_sha256": tensor_raw_sha256(raw_label),
        "first_batch_augmented_image_sha256": tensor_raw_sha256(image),
        "first_batch_augmented_label_sha256": tensor_raw_sha256(label),
    }


def _forward_loss(model, image, label, supervised_loss):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(image).segmentation_logits
        supervised = supervised_loss(logits, label)
    statistics = foreground_hard_ml1ace(logits, label)
    total = supervised + ML1ACE_WEIGHT * statistics.loss
    if not all(bool(torch.isfinite(v)) for v in (supervised, statistics.loss, total)):
        raise RuntimeError("hACE produced non-finite loss")
    return logits, supervised, statistics, total


def _checkpoint(
    args, model, global_epoch, diagnostics, pairing, reload_difference, stripped
):
    return {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "arm": "hace",
        "global_epoch": global_epoch,
        "continuation_epoch": global_epoch - START_GLOBAL_EPOCH,
        "resolution": 128,
        "learning_rate": args.lr,
        "model_state_dict": OrderedDict(
            (name, tensor.detach().cpu().clone()) for name, tensor in model.state_dict().items()
        ),
        "architecture": _architecture(),
        "hard_ml1ace": {
            "enabled": True,
            "num_bins": NUM_BINS,
            "right": BUCKETIZE_RIGHT,
            "foreground_only": True,
            "weight": ML1ACE_WEIGHT,
            "precision": "float32",
        },
        "loss_diagnostics": diagnostics,
        "pairing": pairing,
        "checkpoint_reload_max_abs_difference": reload_difference,
        "stripped_deploy_max_abs_difference": stripped,
        "start_checkpoint_sha256": KD200_SHA256,
        "start_checkpoint_historical_validation_access": True,
        "start_checkpoint_historical_test_access_after_selection": True,
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }


def _smoke(args, model, optimizer, parameters, loader, initial_sha, device):
    write_json_atomic(args.output_dir / "SMOKE_STARTED.json", {
        "schema": SMOKE_SCHEMA + ".started", "status": "before_optimizer_step",
        "validation_access": False, "outer_selector_access": False,
        "confirmation_access": False, "test_access": False,
    })
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    model.train(); supervised_loss = _supervised_loss(); iterator = iter(loader)
    measured=[]; first=None; gradient_audit=None; latest={}; ace_rng_unchanged=None; started_all=time.time()
    for step in range(4):
        batch=next(iterator)
        raw_image=batch["image"].to(device=device,dtype=torch.float32,non_blocking=True)
        raw_label=batch["label"].to(device=device,dtype=torch.float32,non_blocking=True)
        image,label=augment_batch(raw_image.clone(),raw_label.clone(),max_shift=args.max_shift)
        if first is None:
            first=_first_pairing(batch,raw_image,raw_label,image,label); first["initial_student_state_sha256"]=initial_sha
        optimizer.zero_grad(set_to_none=True); torch.cuda.synchronize(device); started=time.time()
        logits, supervised, stats, total=_forward_loss(model,image,label,supervised_loss)
        if gradient_audit is None:
            grad=torch.autograd.grad(stats.loss,logits,retain_graph=True)[0]
            gradient_audit={"finite":bool(torch.isfinite(grad).all()),"nonzero":float(grad.float().norm())>0,"norm":float(grad.float().norm())}
            rng_before_loss=cuda_rng_sha256(device)
            _=foreground_hard_ml1ace(logits.detach(),label)
            ace_rng_unchanged=rng_before_loss==cuda_rng_sha256(device)
        total.backward(); grad_norm=nn.utils.clip_grad_norm_(parameters,max_norm=5.0); optimizer.step(); torch.cuda.synchronize(device)
        if step>0: measured.append(time.time()-started)
        latest={"supervised":float(supervised.detach()),"hard_ml1ace":float(stats.loss.detach()),"total":float(total.detach()),"ace_to_supervised_ratio":float(stats.loss.detach())/float(supervised.detach()),"grad_norm_before_clip":float(grad_norm.detach()),"train_dsc":float(dice_per_case(logits.detach().float(),label).mean()),"empty_target_cases":int((~stats.target_nonempty).sum())}
    peak=int(torch.cuda.max_memory_allocated(device)); reserved=int(torch.cuda.max_memory_reserved(device)); median=statistics.median(measured)
    gates={"losses_finite":all(np.isfinite(v) for v in latest.values()),"ace_gradient_finite_nonzero":gradient_audit["finite"] and gradient_audit["nonzero"],"peak_allocated_at_most_24_gib":peak<=GPU_MEMORY_STOP_BYTES,"rng_unchanged_by_loss":ace_rng_unchanged is True}
    report={"schema":SMOKE_SCHEMA,"schema_version":1,"status":"complete" if all(gates.values()) else "failed_gate","arm":"hace","initial_student_state_sha256":initial_sha,"pairing":first,"losses":latest,"gradient_audit":gradient_audit,"measured_step_seconds":measured,"median_step_seconds":median,"estimated_formal_seconds":median*94*32,"peak_gpu_memory_bytes":peak,"peak_gpu_reserved_bytes":reserved,"gates":gates,"all_gates_passed":all(gates.values()),"elapsed_seconds":time.time()-started_all,"runner_sha256":sha256_file(Path(__file__).resolve()),"validation_access":False,"outer_selector_access":False,"confirmation_access":False,"test_access":False}
    write_json_atomic(args.output_dir/"GPU_SMOKE.json",report)
    if not report["all_gates_passed"]: raise RuntimeError(f"hACE smoke failed:{gates}")
    return report


def _formal(args, model, optimizer, parameters, data, weights, loader, initial_sha, receipt_sha, control_pairing, device):
    write_json_atomic(args.output_dir/"STARTED.json",{"schema":SCHEMA,"schema_version":1,"status":"before_optimizer_step","arm":"hace","created_at_utc":datetime.now(timezone.utc).isoformat(),"runner_sha256":sha256_file(Path(__file__).resolve()),"implementation_receipt_sha256":receipt_sha,"validation_access":False,"outer_selector_access":False,"confirmation_access":False,"test_access":False})
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); torch.cuda.reset_peak_memory_stats(device)
    model.train(); supervised_loss=_supervised_loss(); history=[]; checkpoints={}; case_sha={}; rng_before={}; rng_after={}; first=None; started=time.time(); last_image=None
    for ce in range(1,args.continuation_epochs+1):
        ge=args.start_global_epoch+ce; cases=[]; totals=[]; sups=[]; aces=[]; dsc=[]; empty=0
        for step,batch in enumerate(loader):
            raw_image=batch["image"].to(device=device,dtype=torch.float32,non_blocking=True); raw_label=batch["label"].to(device=device,dtype=torch.float32,non_blocking=True)
            if step==0:rng_before[str(ge)]=cuda_rng_sha256(device)
            image,label=augment_batch(raw_image.clone(),raw_label.clone(),max_shift=args.max_shift)
            if first is None:first=_first_pairing(batch,raw_image,raw_label,image,label);first["initial_student_state_sha256"]=initial_sha
            cases.extend(str(x) for x in batch["case_id"]);optimizer.zero_grad(set_to_none=True)
            logits,sup,stats,total=_forward_loss(model,image,label,supervised_loss);total.backward();nn.utils.clip_grad_norm_(parameters,max_norm=5.0);optimizer.step()
            totals.append(float(total.detach()));sups.append(float(sup.detach()));aces.append(float(stats.loss.detach()));empty+=int((~stats.target_nonempty).sum());dsc.extend(dice_per_case(logits.detach().float(),label).cpu().tolist());last_image=image.detach()
        case_sha[str(ge)]=text_lines_sha256(cases);rng_after[str(ge)]=cuda_rng_sha256(device)
        diag={"train_total_loss":float(np.mean(totals)),"train_supervised_loss":float(np.mean(sups)),"train_hard_ml1ace":float(np.mean(aces)),"train_dsc":float(np.mean(dsc)),"empty_target_cases":empty}
        history.append({"global_epoch":ge,"continuation_epoch":ce,"learning_rate":args.lr,**diag,"optimizer_steps":len(loader),"elapsed_seconds":time.time()-started})
        pairing={"initial_student_state_sha256":initial_sha,**first,"epoch_case_order_sha256":dict(case_sha),"epoch_cuda_rng_before_sha256":dict(rng_before),"epoch_cuda_rng_after_sha256":dict(rng_after)}
        if ge in SAVE_GLOBAL_EPOCHS:
            model.eval()
            with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):ref=model(last_image).segmentation_logits.detach()
            model.train();reload_diff=_plain_reload_difference(OrderedDict((n,t.detach().cpu().clone()) for n,t in model.state_dict().items()),_architecture(),ref,last_image,device);stripped=0.0
            if reload_diff!=0.0:raise RuntimeError("hACE checkpoint reload mismatch")
            value=_checkpoint(args,model,ge,diag,pairing,reload_diff,stripped);path=args.output_dir/f"plain_128_hace_global_epoch{ge}.pth";torch.save(value,path)
            checkpoints[str(ge)]={"path":str(path.resolve()),"sha256":sha256_file(path),"checkpoint_reload_max_abs_difference":reload_diff,"stripped_deploy_max_abs_difference":stripped}
        write_json_atomic(args.output_dir/"progress.json",{"history":history,"checkpoints":checkpoints,"pairing":pairing})
        print(f"arm=hace global_epoch={ge}/232 loss={diag['train_total_loss']:.6f} sup={diag['train_supervised_loss']:.6f} ace={diag['train_hard_ml1ace']:.6f} dsc={diag['train_dsc']:.6f}",flush=True)
    if set(checkpoints)!={str(x) for x in SAVE_GLOBAL_EPOCHS}:raise RuntimeError("hACE checkpoint grid incomplete")
    pairing={"initial_student_state_sha256":initial_sha,**first,"epoch_case_order_sha256":case_sha,"epoch_cuda_rng_before_sha256":rng_before,"epoch_cuda_rng_after_sha256":rng_after}
    pair_fields=("initial_student_state_sha256","first_batch_case_ids","first_batch_case_ids_sha256","first_batch_raw_image_sha256","first_batch_raw_label_sha256","first_batch_augmented_image_sha256","first_batch_augmented_label_sha256","epoch_case_order_sha256","epoch_cuda_rng_before_sha256","epoch_cuda_rng_after_sha256")
    mismatched=[f for f in pair_fields if pairing[f]!=control_pairing[f]]
    report={"schema":SCHEMA,"schema_version":1,"status":"complete" if not mismatched else "failed_pairing","arm":"hace","history":history,"checkpoints":checkpoints,"pairing":pairing,"pairing_mismatched_fields":mismatched,"pairing_all_gates_passed":not mismatched,"reload_max_abs_difference":0.0,"stripped_deploy_max_abs_difference":0.0,"hard_ml1ace":{"enabled":True,"num_bins":NUM_BINS,"right":BUCKETIZE_RIGHT,"foreground_only":True,"weight":ML1ACE_WEIGHT,"precision":"float32"},"start_checkpoint_sha256":KD200_SHA256,"start_checkpoint_historical_validation_access":True,"start_checkpoint_historical_test_access_after_selection":True,"elapsed_seconds":time.time()-started,"peak_gpu_memory_bytes":int(torch.cuda.max_memory_allocated(device)),"peak_gpu_reserved_bytes":int(torch.cuda.max_memory_reserved(device)),"runner_sha256":sha256_file(Path(__file__).resolve()),"initial_student_state_sha256":initial_sha,"data_role":{"train":case_view_metadata(args.train_case_ids),"cases":len(data),"subjects":len({subject_id(x) for x in data.case_ids}),"cache_index_sha256":TRAIN_CACHE_INDEX_SHA256},"sampling_contract":{"weights_sha256":tensor_raw_sha256(weights),"generator_seed":args.seed,"steps_per_epoch":len(loader)},"validation_access":False,"outer_selector_access":False,"confirmation_access":False,"test_access":False}
    write_json_atomic(args.output_dir/"TRAIN_DONE.json",report)
    if mismatched:raise RuntimeError(f"hACE/control pairing mismatch:{mismatched}")
    return report


def run(args):
    _require_frozen_inputs(args)
    if args.output_dir.exists():raise FileExistsError("refusing hACE output reuse")
    args.output_dir.mkdir(parents=True,exist_ok=False);device=torch.device(args.device)
    if device.type!="cuda" or not torch.cuda.is_available():raise RuntimeError("hACE requires CUDA")
    torch.cuda.set_device(device);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model=_load_model(args,device);initial_sha=state_sha256(model.state_dict());optimizer,parameters=_make_optimizer(model,args);data,weights,loader=_make_data(args)
    if args.mode=="smoke":return _smoke(args,model,optimizer,parameters,loader,initial_sha,device)
    receipt_sha=_validate_implementation_receipt(args.implementation_receipt);control_pairing=_load_control_pairing(args)
    return _formal(args,model,optimizer,parameters,data,weights,loader,initial_sha,receipt_sha,control_pairing,device)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--mode",choices=("smoke","formal"),required=True);p.add_argument("--cache",type=Path,required=True);p.add_argument("--train-case-ids",type=Path,required=True);p.add_argument("--initial-checkpoint",type=Path,required=True);p.add_argument("--control-training-report",type=Path,required=True);p.add_argument("--control-pairing-receipt",type=Path,required=True);p.add_argument("--preregistration",type=Path,required=True);p.add_argument("--preregistration-amendment",type=Path,required=True);p.add_argument("--implementation-receipt",type=Path,default=None);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--seed",type=int,default=20260860);p.add_argument("--batch-size",type=int,default=8);p.add_argument("--num-workers",type=int,default=4);p.add_argument("--start-global-epoch",type=int,default=200);p.add_argument("--continuation-epochs",type=int,default=32);p.add_argument("--lr",type=float,default=1e-5);p.add_argument("--weight-decay",type=float,default=1e-5);p.add_argument("--max-shift",type=int,default=8);args=p.parse_args();r=run(args);print(json.dumps({"schema":r["schema"],"status":r["status"],"elapsed_seconds":r.get("elapsed_seconds"),"all_gates_passed":r.get("all_gates_passed")},indent=2),flush=True)


if __name__=="__main__":main()
