"""Single frozen readout of the four completed students, patient-cluster statistics."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts import no_n4_clean_protocol as p
from scripts import no_n4_clean_analysis as analysis
from scripts import train_t1ce_wt_gcsam_inner as base
from scripts.freeze_no_n4_clean import check_implementation

AXES=((),(-1,),(-2,),(-2,-1),(-3,),(-3,-1),(-3,-2),(-3,-2,-1))


def metrics(logits,labels,case_ids,subject_ids):
    predictions=logits>0;targets=labels>0
    rows=[]
    for slot,case in enumerate(case_ids):
        prediction=predictions[slot].flatten();target=targets[slot].flatten()
        count_p=int(prediction.sum());count_t=int(target.sum());intersection=int((prediction&target).sum())
        point=int(logits[slot].flatten().argmax())
        rows.append({"case_id":case,"subject_id":subject_ids[slot],"dice":1. if count_p+count_t==0 else 2*intersection/(count_p+count_t),
            "predicted_voxels":count_p,"target_voxels":count_t,"intersection_voxels":intersection,"point_hit":bool(target[point]),
            "point_flat_index":point,"prediction_mask_sha256":hashlib.sha256(np.packbits(prediction.cpu().numpy()).tobytes()).hexdigest()})
    return rows


@torch.inference_mode()
def evaluate(model,loader,device):
    rows={"identity":[],"flip8":[]};start=time.monotonic();forwards=0
    for batch in loader:
        image=batch["image"].to(device);labels=batch["label"].to(device);total=None;identity=None
        for axes in AXES:
            view=torch.flip(image,axes) if axes else image
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):logits=model(view).segmentation_logits.float()
            logits=torch.flip(logits,axes) if axes else logits
            if identity is None:identity=logits.clone();total=logits.clone()
            else:total.add_(logits)
            forwards+=image.shape[0]
        for style,value in (("identity",identity),("flip8",total/8)):
            rows[style].extend(metrics(value,labels,list(batch["case_id"]),list(batch["subject_id"])))
    return {"per_case":rows,"forwards":forwards,"elapsed_seconds":time.monotonic()-start}


def main(args):
    p.docs();check_implementation()
    receipt_path=p.RECORD/"INNER_PREACCESS.json";receipt=json.loads(receipt_path.read_text())
    if receipt["status"]!="frozen_before_first_inner_model_access" or receipt["inner_metrics_accessed"]:raise ValueError("invalid preaccess")
    p.checked(p.RECORD/"SPLIT.json",p.SPLIT_SHA);split=json.loads((p.RECORD/"SPLIT.json").read_text())
    cache_receipt=json.loads((p.RECORD/"CACHE_RECEIPT.json").read_text())
    for fold,roles in receipt["reports"].items():
        for role,record in roles.items():p.checked(record["report_path"],record["report_sha256"]);p.checked(record["checkpoint"]["path"],record["checkpoint"]["sha256"])
    if (p.RECORD/"INNER_STARTED.json").exists() or (p.RECORD/"RAW_INNER.json").exists():raise FileExistsError("inner readout already started")
    device=torch.device(args.device)
    if device.type!="cuda" or not torch.cuda.is_available():raise RuntimeError("CUDA required")
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    p.write_new(p.RECORD/"INNER_STARTED.json",{"created_utc":datetime.now(timezone.utc).isoformat(),"preaccess_sha256":p.sha(receipt_path),"all_models_frozen":True})
    raw={};comparisons={};start=time.monotonic()
    for fold in ("e","f"):
        inner=p.read_rows(split["folds"][fold]["inner"]["path"]);ids=[r["case_id"] for r in inner]
        raw[fold]={}
        for role in ("canonical","no_n4"):
            cache=Path(cache_receipt["candidate_cache"]) if role=="no_n4" else p.CONTROL_CACHE
            data=base.BrainCropDataset(cache,expected_split="train",case_ids=ids)
            loader=DataLoader(data,batch_size=8,shuffle=False,num_workers=0)
            payload=torch.load(receipt["reports"][fold][role]["checkpoint"]["path"],map_location="cpu",weights_only=False)
            if payload["smoke"] or payload["epoch"]!=168 or payload["fold"]!=fold or payload["role"]!=role:raise ValueError("wrong checkpoint")
            model=base.AutomaticPromptLocator(center_aware=False,init_filters=16,dropout=.1).to(device).eval()
            model.load_state_dict(payload["model_state_dict"],strict=True)
            raw[fold][role]=evaluate(model,loader,device)
            if raw[fold][role]["forwards"]!=42*8:raise RuntimeError("unexpected inference count")
            print(f"completed frozen {fold}/{role}:42 cases",flush=True)
            del model;torch.cuda.empty_cache()
        n=split["folds"][fold]["inner"]["subjects"]
        draws=np.random.default_rng(p.SEEDS[fold]["bootstrap"]).integers(0,n,size=(10_000,n))
        comparisons[fold]={style:analysis.compare(raw[fold]["canonical"]["per_case"][style],raw[fold]["no_n4"]["per_case"][style],draws) for style in ("identity","flip8")}
    aggregate=analysis.aggregate(comparisons)
    report={"schema":"aflow.no_n4_clean.inner","created_utc":datetime.now(timezone.utc).isoformat(),"raw_models":raw,"comparisons":comparisons,"aggregate":aggregate,
            "stage_b_authorized":aggregate["all_gates_passed"],"bootstrap":{"unit":"patient_cluster_all_timepoints","samples":10_000,"confidence":.95,"seeds":{f:p.SEEDS[f]["bootstrap"] for f in ("e","f")}},
            "preaccess_sha256":p.sha(receipt_path),"elapsed_seconds":time.monotonic()-start,"total_forwards":1344,
            "access":{"new_train_only_inner":True,"fixed_validation":False,"outer":False,"confirmation":False,"test270":False,"ucsd75":False,"rhuh":False}}
    p.write_new(p.RECORD/"RAW_INNER.json",report)
    print(json.dumps({"aggregate":aggregate,"stage_b_authorized":report["stage_b_authorized"]},indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--device",default="cuda:0");main(ap.parse_args())
