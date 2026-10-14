#!/usr/bin/env python3
"""Evaluate frozen equal-weight seed32/40/41 Full972 logit ensemble on validation108."""
from __future__ import annotations
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.evaluate_t1ce_wt_plain_ema_validation import compare  # noqa:E402
from scripts.train_t1ce_wt_capg3d import BrainCropDataset,dice_per_case,read_case_ids,sha256_file,summarize_cases,write_json_atomic  # noqa:E402
from scripts.train_t1ce_wt_plain_voxel_kd_continuation import _model  # noqa:E402
import scripts.train_t1ce_wt_voxel_kd_fulltrajectory_final as seed32train  # noqa:E402
import scripts.train_t1ce_wt_full972_seed_replicates as seedtrain  # noqa:E402
SCHEMA="aflow.sammed3d.t1ce_wt_full972_three_seed_equal_ensemble_validation"
MANIFEST_SHA="f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb"; CACHE_SHA="9339e4575561447dd5bd99bc29e909ae1e9f566721d351a244d087a40622f322"
CKPTS={"seed32":"f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b","seed40":"48c366ef9f5d3ba5c667482deda6df90fa7cd4bf03c314017e15c13f161dfd38","seed41":"00db7860b7951d1f4b0affde6cad558f679421ba93dfe69157c31bd497b19ddb"}
def point(logits):
 flat=torch.argmax(logits[:,0].flatten(1),1); h,w=logits.shape[-2:]; z=torch.div(flat,h*w,rounding_mode="floor"); r=flat%(h*w); y=torch.div(r,w,rounding_mode="floor"); return torch.stack((z,y,r%w),1)
def load(path:Path,name:str,device):
 if sha256_file(path)!=CKPTS[name]: raise ValueError("checkpoint SHA mismatch")
 c=torch.load(path,map_location="cpu",weights_only=False)
 expected=seed32train.CHECKPOINT_SCHEMA if name=="seed32" else seedtrain.CHECKPOINT_SCHEMA
 if c.get("schema")!=expected or c.get("epoch")!=168: raise ValueError("checkpoint contract mismatch")
 for k in ("validation_access","outer_selector_access","confirmation_access","test_access"):
  if c.get(k) is not False: raise ValueError("checkpoint prior access")
 return _model(c,device).eval(),c
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser(); p.add_argument("--cache",type=Path,required=True); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--seed32",type=Path,required=True); p.add_argument("--seed40",type=Path,required=True); p.add_argument("--seed41",type=Path,required=True); p.add_argument("--preaccess",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--device",default="cuda:0"); p.add_argument("--num-workers",type=int,default=4); a=p.parse_args()
 started=a.output.with_suffix(".started.json")
 if a.output.exists() or started.exists(): raise FileExistsError("refusing repeated ensemble validation")
 if sha256_file(a.manifest)!=MANIFEST_SHA or sha256_file(a.cache/"index.json")!=CACHE_SHA: raise ValueError("validation binding")
 pre=json.loads(a.preaccess.read_text()); ev=sha256_file(Path(__file__).resolve())
 if pre.get("status")!="frozen_before_validation" or pre.get("evaluator_sha256")!=ev or pre.get("checkpoint_sha256")!=CKPTS: raise ValueError("preaccess mismatch")
 device=torch.device(a.device); torch.cuda.set_device(device)
 models={}; checkpoints={}
 for name,path in (("seed32",a.seed32),("seed40",a.seed40),("seed41",a.seed41)): models[name],checkpoints[name]=load(path,name,device)
 ids=read_case_ids(a.manifest); data=BrainCropDataset(a.cache,expected_split="validation",case_ids=ids)
 if len(data)!=108 or len({x.rsplit("-",1)[0] for x in data.case_ids})!=50: raise ValueError("cohort")
 write_json_atomic(started,{"schema":SCHEMA+".started","created_at_utc":datetime.now(timezone.utc).isoformat(),"formula":"(seed32+seed40+seed41)/3","checkpoint_sha256":CKPTS,"validation_access":True,"test_access":False})
 loader=DataLoader(data,batch_size=8,shuffle=False,num_workers=a.num_workers,pin_memory=True,persistent_workers=a.num_workers>0); cases={k:[] for k in ("seed32","seed40","seed41","ensemble")}
 for batch in loader:
  image=batch["image"].to(device,dtype=torch.float32,non_blocking=True); label=batch["label"].to(device,dtype=torch.float32,non_blocking=True); logits={}
  with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
   for name,m in models.items(): logits[name]=m(image).segmentation_logits.float()
  logits["ensemble"]=(logits["seed32"]+logits["seed40"]+logits["seed41"])/3.0
  for name,z in logits.items():
   ds=dice_per_case(z,label); pts=point(z)
   for i,cid in enumerate(batch["case_id"]):
    zz,yy,xx=map(int,pts[i].tolist()); cases[name].append({"case_id":str(cid),"subject_id":str(batch["subject_id"][i]),"locator_dsc":float(ds[i]),"automatic_point":[zz,yy,xx],"automatic_point_hits_gt":bool(label[i,0,zz,yy,xx]>.5)})
 summaries={k:summarize_cases(v) for k,v in cases.items()}; delta=compare(summaries["seed32"],summaries["ensemble"],seed=20261700)["ema_minus_raw"]; ens=summaries["ensemble"]; hits=round(ens["automatic_point_hit_rate"]*108)
 gates={"beats_seed32_scan":ens["locator_dsc_mean"]>summaries["seed32"]["locator_dsc_mean"],"beats_seed32_subject":ens["locator_subject_dsc_mean"]>summaries["seed32"]["locator_subject_dsc_mean"],"point_at_least_101":hits>=101,"scan_ci_lower":delta["scan_95ci"][0]>=-.003,"subject_ci_lower":delta["subject_95ci"][0]>=-.003}
 report={"schema":SCHEMA,"schema_version":1,"created_at_utc":datetime.now(timezone.utc).isoformat(),"evidence_scope":"post-test research; fixed equal three-seed formula; no test selection","formula":{"seed32":1/3,"seed40":1/3,"seed41":1/3},"summaries":summaries,"ensemble_minus_seed32":delta,"gates":gates,"all_gates_passed":all(gates.values()),"distillation_authorized":all(gates.values()),"checkpoint_sha256":CKPTS,"validation_access":True,"outer_selector_access":False,"confirmation_access":False,"test_access":False}; write_json_atomic(a.output,report); print(json.dumps({"summaries":{k:{"scan":v["locator_dsc_mean"],"subject":v["locator_subject_dsc_mean"],"point":v["automatic_point_hit_rate"]} for k,v in summaries.items()},"delta":delta,"distillation_authorized":report["distillation_authorized"]},indent=2,sort_keys=True))
if __name__=="__main__": main()
