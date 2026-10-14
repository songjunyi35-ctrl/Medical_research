"""One-time UTSW362 identity/Flip8 evaluation of two frozen baselines."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import statistics,time
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset

from scripts import train_t1ce_wt_gcsam_inner as base
from scripts import utsw_frozen_baseline_external_protocol as p


class Data(Dataset):
    def __init__(self):
        receipt=json.loads((p.RECORD/"CACHE_RECEIPT.json").read_text())
        if receipt.get("all_gates_passed") is not True:raise ValueError("cache failed")
        p.checked(p.CACHE/"index.json",receipt["index_sha256"]);self.index=json.loads((p.CACHE/"index.json").read_text())
        if self.index.get("cases")!=362 or self.index.get("split")!="external_cross_stage":raise ValueError("cache identity")
        p.checked(p.CACHE/"images.f32",self.index["images_sha256"]);p.checked(p.CACHE/"labels.u8",self.index["labels_sha256"])
        self.images=np.memmap(p.CACHE/"images.f32",mode="r",dtype=np.float32,shape=(362,128,128,128));self.labels=np.memmap(p.CACHE/"labels.u8",mode="r",dtype=np.uint8,shape=(362,128,128,128))
    def __len__(self):return 362
    def __getitem__(self,i):return {"image":torch.from_numpy(np.array(self.images[i]))[None],"label":torch.from_numpy(np.array(self.labels[i]))[None],
        "case_id":self.index["case_ids"][i],"subject_id":self.index["subject_ids"][i]}


def load_model(name,device):
    pre=p.verify_preaccess();row=p.MODELS[name];path=row["path"];p.checked(path,row["file_sha256"]);payload=torch.load(path,map_location="cpu",weights_only=False)
    if payload.get("schema")!=row["schema"] or payload.get("epoch")!=row["epoch"] or payload.get("model_state_sha256")!=row["state_sha256"]:raise ValueError("checkpoint identity "+name)
    if base.state_dict_sha256(payload["model_state_dict"])!=row["state_sha256"]:raise ValueError("checkpoint state "+name)
    model=base.AutomaticPromptLocator(center_aware=False,init_filters=16,dropout=.1).to(device).eval();model.load_state_dict(payload["model_state_dict"],strict=True)
    if sum(x.numel() for x in model.parameters())!=4_700_897:raise ValueError("parameter count")
    return model,payload,pre


def metric_rows(logits,target,batch,view):
    pred=logits>0;rows=[]
    for i,case in enumerate(batch["case_id"]):
        a=pred[i].flatten();b=target[i].flatten();na,nb=int(a.sum()),int(b.sum());inter=int((a&b).sum());point=int(logits[i].flatten().argmax())
        rows.append({"case_id":str(case),"subject_id":str(batch["subject_id"][i]),"view":view,"dice":2*inter/(na+nb),"point_hit":bool(b[point]),
            "point_flat_index":point,"predicted_voxels":na,"target_voxels":nb,"intersection_voxels":inter,
            "prediction_mask_sha256":hashlib.sha256(np.packbits(a.cpu().numpy()).tobytes()).hexdigest()})
    return rows


def summarize(rows):
    if len(rows)!=362 or len({x["case_id"] for x in rows})!=362 or any(x["case_id"]!=x["subject_id"] for x in rows):raise ValueError("summary cohort")
    dice=statistics.fmean(x["dice"] for x in rows)
    return {"cases":362,"subjects":362,"scan_dice":dice,"subject_dice":dice,"point_hits":sum(x["point_hit"] for x in rows),
        "point_hit_rate":statistics.fmean(float(x["point_hit"]) for x in rows),"predicted_voxels_mean":statistics.fmean(x["predicted_voxels"] for x in rows),
        "target_voxels_mean":statistics.fmean(x["target_voxels"] for x in rows)}


def bootstrap(kd,gcsam):
    kd=sorted(kd,key=lambda x:x["case_id"]);gcsam=sorted(gcsam,key=lambda x:x["case_id"])
    if [x["case_id"] for x in kd]!=[x["case_id"] for x in gcsam]:raise ValueError("pair IDs")
    kd_d=np.array([x["dice"] for x in kd]);gc_d=np.array([x["dice"] for x in gcsam]);kd_p=np.array([x["point_hit"] for x in kd],float);gc_p=np.array([x["point_hit"] for x in gcsam],float)
    draw=np.random.default_rng(p.BOOTSTRAP_SEED).integers(0,362,size=(p.BOOTSTRAP_SAMPLES,362));series={
        "full972_kd168_scan":kd_d[draw].mean(1),"gcsam184_scan":gc_d[draw].mean(1),"scan_delta_gcsam_minus_kd":(gc_d-kd_d)[draw].mean(1),
        "full972_kd168_point_rate":kd_p[draw].mean(1),"gcsam184_point_rate":gc_p[draw].mean(1),"point_rate_delta_gcsam_minus_kd":(gc_p-kd_p)[draw].mean(1)}
    return {"unit":"subject_one_scan_each","samples":p.BOOTSTRAP_SAMPLES,"seed":p.BOOTSTRAP_SEED,
        **{k+"_95ci":[float(x) for x in np.quantile(v,(.025,.975))] for k,v in series.items()}}


def reference_gates(summaries):
    return {"full972_kd168_flip8_scan":summaries["full972_kd168"]["flip8"]["scan_dice"]>=.70,
        "full972_kd168_flip8_subject":summaries["full972_kd168"]["flip8"]["subject_dice"]>=.70,
        "full972_kd168_flip8_point":summaries["full972_kd168"]["flip8"]["point_hits"]>=326,
        "gcsam184_identity_scan":summaries["gcsam184"]["identity"]["scan_dice"]>=.70,
        "gcsam184_identity_subject":summaries["gcsam184"]["identity"]["subject_dice"]>=.70,
        "gcsam184_identity_point":summaries["gcsam184"]["identity"]["point_hits"]>=326}


@torch.inference_mode()
def evaluate(name,device):
    loader=DataLoader(Data(),batch_size=8,shuffle=False,num_workers=0);model,_,_=load_model(name,device);rows={"identity":[],"flip8":[]};count=0;torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        image=batch["image"].to(device,dtype=torch.float32);target=batch["label"].to(device,dtype=torch.bool)
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):identity=model(image).segmentation_logits.float()
        count+=len(image);rows["identity"].extend(metric_rows(identity,target,batch,"identity"));total=None
        for axes in p.AXES:
            view=torch.flip(image,axes) if axes else image
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):logits=model(view).segmentation_logits.float()
            logits=torch.flip(logits,axes) if axes else logits;total=logits.clone() if total is None else total.add_(logits);count+=len(image)
        rows["flip8"].extend(metric_rows(total/8,target,batch,"flip8"))
    peak=torch.cuda.max_memory_allocated(device);del model;torch.cuda.empty_cache();return rows,count,peak


@torch.inference_mode()
def main(device_name):
    from scripts.freeze_utsw_frozen_baseline_external import check_implementation
    check_implementation();pre=p.verify_preaccess();smoke=json.loads((p.RECORD/"GPU_SMOKE_RECEIPT.json").read_text());cache=json.loads((p.RECORD/"CACHE_RECEIPT.json").read_text())
    if smoke.get("status")!="passed" or smoke.get("preaccess_sha256")!=p.sha(p.PREACCESS) or cache.get("all_gates_passed") is not True:raise ValueError("smoke/cache")
    output=p.RESULT_WORK/"UTSW_FROZEN_BASELINES.json";started=p.RESULT_WORK/"UTSW_FROZEN_BASELINES_STARTED.json"
    if output.exists() or started.exists():raise FileExistsError("repeat external")
    device=torch.device(device_name)
    if device.type!="cuda" or not torch.cuda.is_available():raise RuntimeError("CUDA required")
    torch.cuda.set_device(device);torch.set_num_threads(1);p.RESULT_WORK.mkdir(parents=True,exist_ok=False);p.write_new(started,{"schema":"aflow.utsw_frozen_baseline_external.started",
        "created_utc":datetime.now(timezone.utc).isoformat(),"preaccess_sha256":p.sha(p.PREACCESS),"cache_receipt_sha256":p.sha(p.RECORD/"CACHE_RECEIPT.json"),
        "views":["identity","flip8"],"external_metrics_access":True})
    begin=time.monotonic();rows={};summaries={};peaks={};forwards=0
    for name in ("full972_kd168","gcsam184"):
        rows[name],n,peaks[name]=evaluate(name,device);forwards+=n;summaries[name]={v:summarize(rows[name][v]) for v in ("identity","flip8")}
    intervals={v:bootstrap(rows["full972_kd168"][v],rows["gcsam184"][v]) for v in ("identity","flip8")}
    if forwards!=6516:raise RuntimeError("forward count")
    gates=reference_gates(summaries);report={"schema":"aflow.utsw_frozen_baseline_external.result","created_utc":datetime.now(timezone.utc).isoformat(),
        "evidence_scope":"one-time UTSW362 preoperative SRI24 cross-stage external frozen-baseline panel","models":pre["models"],"protocol":pre["protocol"],
        "summaries":summaries,"bootstrap":intervals,"per_case":rows,"reference_gates":gates,"all_reference_gates_passed":all(gates.values()),
        "paired_differences_are_descriptive_not_causal":True,"forwards":forwards,"elapsed_seconds":time.monotonic()-begin,"peak_gpu_memory_bytes":peaks,
        "preaccess_sha256":p.sha(p.PREACCESS),"cache_receipt_sha256":p.sha(p.RECORD/"CACHE_RECEIPT.json"),"evaluator_sha256":p.sha(Path(__file__)),
        "access":{"utsw_external":True,"fixed_validation":False,"outer":False,"confirmation":False,"test270":False,"ucsd75":False,"rhuh":False},
        "result_must_not_trigger_tuning_or_repeat":True,"grid_caveat":"metrics on brain-support crop rasterized128^3",
        "domain_caveat":"UTSW preoperative/SRI24 vs training post-treatment/MNI","training_source_caveat":pre["training_source_caveat"]}
    p.write_new(output,report);print(json.dumps({"summaries":summaries,"bootstrap":intervals,"gates":gates,"forwards":forwards},indent=2))


if __name__=="__main__":
    import argparse;ap=argparse.ArgumentParser();ap.add_argument("--device",default="cuda:0");main(ap.parse_args().device)
