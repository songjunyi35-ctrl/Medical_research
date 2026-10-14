"""Student-only Otsu cache fork of the frozen no-N4 trainer; shared teachers/control."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss
from scripts import n4_otsu_protocol as p
from scripts.train_no_n4_clean import verify_teacher
from scripts import train_t1ce_wt_gcsam_inner as base

ROLES=("otsu",)
SCHEMA="aflow.n4_otsu_clean.train"


@torch.no_grad()
def update_ema(shadow,model):
    for key,value in model.state_dict().items():
        if value.is_floating_point():shadow[key].mul_(0.999).add_(value.detach(),alpha=0.001)
        else:shadow[key].copy_(value.detach())



def rng(device):
    return torch.random.get_rng_state().numpy().tobytes()+torch.cuda.get_rng_state(device).cpu().numpy().tobytes()


def run(args):
    p.docs();p.checked(p.SHARED_RECORD/"SPLIT.json",p.SPLIT_SHA);p.checked(p.NATIVE,p.NATIVE_SHA)
    from scripts.freeze_n4_otsu_clean import check_implementation
    check_implementation()
    if not args.smoke:
        smoke_receipt=json.loads((p.RECORD/"GPU_SMOKE_RECEIPT.json").read_text())
        if smoke_receipt["status"]!="passed_smoke_pairing" or not smoke_receipt["all_pairing_gates_passed"]:
            raise ValueError("formal launch requires all fold/role smoke gates")
    split=json.loads((p.SHARED_RECORD/"SPLIT.json").read_text());record=split["folds"][args.fold]
    fit_path=Path(record["fit"]["path"]);p.checked(fit_path,record["fit"]["sha256"])
    inner_path=Path(record["inner"]["path"]);p.checked(inner_path,record["inner"]["sha256"])
    p.own_fit(record,p.read_rows(fit_path),p.read_rows(inner_path))
    cache_receipt=json.loads((p.RECORD/"CACHE_RECEIPT.json").read_text())
    if cache_receipt["status"]!="passed" or not cache_receipt["all972_labels_exact"]:raise ValueError("cache not approved")
    cache=Path(cache_receipt["candidate_cache"]) if args.role=="otsu" else p.CONTROL_CACHE
    cache_sha=cache_receipt["candidate_index_sha256"] if args.role=="otsu" else p.INDEX_SHA
    p.checked(cache/"index.json",cache_sha)
    index=json.loads((cache/"index.json").read_text())
    if index["manifest_sha256"]!=p.SOURCE_SHA or index["count"]!=972 or index["split"]!="train" or index["test_access"] is not False:
        raise ValueError("cache role/count/source violation")
    if index["labels_sha256"]!=p.LABELS_SHA or index["preprocessing"]["n4_bias_field_correction"] is not True or index["preprocessing"].get("n4_fit_mask")!="OtsuThreshold(full_image,0,1,200)":
        raise ValueError("cache arm mismatch")
    teacher_role=args.role in ("history","ema")
    seed=p.SEEDS[args.fold][args.role if teacher_role else "student"]
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device(args.device)
    if not torch.cuda.is_available() or device.type!="cuda":raise RuntimeError("CUDA required")
    torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device);torch.set_num_threads(1)
    native=json.loads(p.NATIVE.read_text())
    volumes={str(k):int(v) for k,v in native["native_wt_voxels"].items()}
    ids=[row["case_id"] for row in p.read_rows(fit_path)]
    data=base.NativeFitDataset(cache,expected_split="train",case_ids=ids,limit=8 if args.smoke else None,native_volumes=volumes)
    if not args.smoke and len(data)!=930:raise ValueError("unexpected fit size")
    weights=data.sampling_weights();weight_sha=base.tensor_sha256(weights)
    sampler=base.AuditedWeightedSampler(weights,16 if args.smoke else p.DRAWS,seed)
    loader=DataLoader(data,batch_size=8,sampler=sampler,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0,drop_last=True)
    if len(loader)!=(2 if args.smoke else p.STEPS):raise ValueError("step count mismatch")
    teacher_info=[];teachers=[]
    if not teacher_role:
        for role in ("history","ema"):
            directory=p.PARENT_WORK/("smoke" if args.smoke else "formal")/f"{args.fold}_{role}"
            report=json.loads((directory/"TRAIN_DONE.json").read_text())
            path=Path(report["checkpoint"]["path"]);p.checked(path,report["checkpoint"]["sha256"])
            payload=torch.load(path,map_location="cpu",weights_only=False)
            verify_teacher(payload,report,args.fold,role,record["fit"]["sha256"],args.smoke)
            teacher=base.AutomaticPromptLocator(center_aware=False,init_filters=16,dropout=0.1).to(device)
            teacher.load_state_dict(payload["model_state_dict"],strict=True);teacher.eval().requires_grad_(False)
            teachers.append(teacher);teacher_info.append({"role":role,"path":str(path),"sha256":p.sha(path),"fit_sha256":payload["fit_sha256"]})
    model=base.AutomaticPromptLocator(center_aware=False,init_filters=16,dropout=0.1).to(device)
    if sum(t.numel() for t in model.parameters())!=p.PARAMETERS:raise ValueError("wrong parameter count")
    initial_sha=base.state_dict_sha256(model.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5,foreach=None,fused=None)
    loss_fn=DiceCELoss(sigmoid=True,squared_pred=False,lambda_dice=1.,lambda_ce=.1)
    output=p.WORK/("smoke" if args.smoke else "formal")/f"{args.fold}_{args.role}"
    output.mkdir(parents=True,exist_ok=False)
    access={"fit_train":True,"inner_ids_metadata_only":True,"inner_metrics":False,"fixed_validation":False,"outer":False,"confirmation":False,"test270":False,"ucsd75":False,"rhuh":False}
    p.write_new(output/"STARTED.json",{"schema":SCHEMA+".started","created_utc":datetime.now(timezone.utc).isoformat(),"role":args.role,"fold":args.fold,
        "smoke":args.smoke,"seed":seed,"fit_sha256":record["fit"]["sha256"],"cache_index_sha256":cache_sha,"initial_state_sha256":initial_sha,
        "teacher_checkpoints":teacher_info,"access":access,"runner_sha256":p.sha(Path(__file__)),"protocol_sha256":p.sha(Path(p.__file__))})
    history=[];shadow=None;ema_updates=0;start=time.monotonic()
    epochs=1 if args.smoke else p.EPOCHS
    for epoch in range(1,epochs+1):
        lr=base.learning_rate_for_epoch(epoch,2e-4,1e-5)
        for group in optimizer.param_groups:group["lr"]=lr
        model.train();losses=[];supervised_losses=[];kd_losses=[];dices=[];before=hashlib.sha256();after=hashlib.sha256();first=None
        for step,batch in enumerate(loader):
            image=batch["image"].to(device=device,dtype=torch.float32,non_blocking=True);label=batch["label"].to(device=device,dtype=torch.float32,non_blocking=True)
            before.update(rng(device));image,label=base.augment_batch(image,label,max_shift=8);after.update(rng(device))
            if epoch==1 and step==0:first={"image":base.tensor_sha256(image),"label":base.tensor_sha256(label)}
            optimizer.zero_grad(set_to_none=True)
            if teacher_role:
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                    logits=model(image).segmentation_logits;supervised=loss_fn(logits,label)
                kd=torch.zeros((),device=device);loss=supervised
            else:
                with torch.no_grad(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                    target=.75*teachers[0](image).segmentation_logits.float()+.25*teachers[1](image).segmentation_logits.float()
                loss,supervised,kd,logits=base.student_loss(model,image,label,target,loss_fn)
            if not bool(torch.isfinite(loss)):raise FloatingPointError("nonfinite loss")
            if first is not None and step==0:first["loss"]=float(loss.detach())
            loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            if float(norm)<=0:raise RuntimeError("zero gradient")
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            if args.role=="ema" and shadow is not None:update_ema(shadow,model);ema_updates+=1
            if args.smoke and args.role=="ema" and shadow is None:shadow={k:v.detach().clone() for k,v in model.state_dict().items()}
            if any(t.grad is not None for teacher in teachers for t in teacher.parameters()):raise RuntimeError("teacher gradient leak")
            losses.append(float(loss.detach()));supervised_losses.append(float(supervised.detach()));kd_losses.append(float(kd.detach()))
            dices.extend(base.dice_per_case(logits.detach().float(),label).cpu().tolist())
        if args.role=="ema" and epoch==100:shadow={k:v.detach().clone() for k,v in model.state_dict().items()}
        item={"epoch":epoch,"lr":lr,"loss":float(np.mean(losses)),"supervised":float(np.mean(supervised_losses)),"kd":float(np.mean(kd_losses)),"train_dice":float(np.mean(dices)),
            "steps":len(losses),"draws":len(losses)*8,"sampler_sha256":base.index_sha256(sampler.last_indices[:len(losses)*8]),
            "augmentation_rng_before_sha256":before.hexdigest(),"augmentation_rng_after_sha256":after.hexdigest(),"first_batch":first,"ema_updates":ema_updates,
            "elapsed_seconds":time.monotonic()-start,"inner_metrics":None}
        history.append(item);base.write_json_atomic(output/"progress.json",{"history":history,"inner_metrics_accessed":False})
        print(f"{args.fold}/{args.role} epoch={epoch} loss={item['loss']:.6f} fit_dice={item['train_dice']:.6f}",flush=True)
        if not args.smoke and time.monotonic()-start>6*3600:raise RuntimeError("six hour arm time gate")
        if torch.cuda.max_memory_allocated(device)>24*1024**3:raise RuntimeError("24GiB peak gate")
    selected_state=shadow if args.role=="ema" else model.state_dict()
    if selected_state is None:raise RuntimeError("EMA state was not initialized")
    state_sha=base.state_dict_sha256(selected_state)
    checkpoint={"schema":SCHEMA+".checkpoint","fold":args.fold,"role":args.role,"seed":seed,"epoch":epochs,"smoke":args.smoke,
        "fit_sha256":record["fit"]["sha256"],"input_cache_index_sha256":cache_sha,"model_state_dict":selected_state,"state_sha256":state_sha,
        "initial_state_sha256":initial_sha,"teacher_checkpoints":teacher_info,"inner_metrics_accessed":False,"ema_updates":ema_updates,
        "architecture":{"init_filters":16,"dropout":.1,"center_aware":False},"idea_sha256":p.IDEA_SHA,"prereg_sha256":p.PREREG_SHA,"split_sha256":p.SPLIT_SHA}
    ckpt=output/"fixed.pth";torch.save(checkpoint,ckpt)
    reloaded=torch.load(ckpt,map_location="cpu",weights_only=False)
    gates={"fixed_epochs":len(history)==epochs,"fixed_steps":all(h["steps"]==len(loader) for h in history),
        "reload_exact":base.state_dict_sha256(reloaded["model_state_dict"])==state_sha,
        "ema_updates_exact":ema_updates==(1 if args.smoke else 8228) if args.role=="ema" else ema_updates==0,
        "finite":all(np.isfinite(h[k]) for h in history for k in ("loss","supervised","kd","train_dice")),
        "no_inner":all(h["inner_metrics"] is None for h in history),"peak":torch.cuda.max_memory_allocated(device)<=24*1024**3}
    report={"schema":SCHEMA,"status":"complete","fold":args.fold,"role":args.role,"smoke":args.smoke,"seed":seed,"fit_sha256":record["fit"]["sha256"],
        "checkpoint":{"path":str(ckpt),"sha256":p.sha(ckpt),"state_sha256":state_sha},"initial_state_sha256":initial_sha,"sampling_weight_sha256":weight_sha,
        "history":history,"gates":gates,"all_gates_passed":all(gates.values()),"teacher_checkpoints":teacher_info,"cache_index_sha256":cache_sha,
        "elapsed_seconds":time.monotonic()-start,"peak_gpu_memory_bytes":torch.cuda.max_memory_allocated(device),"access":access,
        "runner_sha256":p.sha(Path(__file__)),"protocol_sha256":p.sha(Path(p.__file__))}
    p.write_new(output/"TRAIN_DONE.json",report)
    if not report["all_gates_passed"]:raise RuntimeError("training engineering gates failed")
    print(json.dumps({k:v for k,v in report.items() if k!="history"},indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--fold",choices=("e","f"),required=True);ap.add_argument("--role",choices=ROLES,required=True)
    ap.add_argument("--device",default="cuda:0");ap.add_argument("--workers",type=int,default=4);ap.add_argument("--smoke",action="store_true")
    run(ap.parse_args())

