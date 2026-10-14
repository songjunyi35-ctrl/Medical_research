#!/usr/bin/env python3
"""Evaluate the three frozen KD200-to-DMRFB216 interior weight interpolants."""
from __future__ import annotations
import argparse,json,math,sys
from datetime import datetime,timezone
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from sammed3d.t1ce_wt_dmrfb_adapter import DMRFBAdapterLocator
from scripts.train_t1ce_wt_capg3d import sha256_file,validate,write_json_atomic
from scripts.evaluate_t1ce_wt_plain_vm_dice_validation import BOUNDARY_TOP27,KD200_BOTTOM27,KD_EPOCH,_compare_segmentation,_validate_cache_and_boundary_stratum,_validate_kd_report,_validate_summary
SCHEMA="aflow.sammed3d.t1ce_wt_dmrfb_weight_interpolation_validation";GEN_SCHEMA="aflow.sammed3d.t1ce_wt_dmrfb_weight_interpolation_generation_receipt";CK_SCHEMA="aflow.sammed3d.t1ce_wt_dmrfb_weight_interpolation_checkpoint";DMRFB_SCHEMA="aflow.sammed3d.t1ce_wt_plain_dmrfb_adapter_validation"
IDEA_SHA="858b963e637e35af0825d6813d6da26c1ef64417f3b496dcad52da23ec485119";PREREG_SHA="2674c8f400ca2822f8a58b7ecc2fc9947034d2453f5b4987819fd10ac5453db3";KD_SHA="56f54898ded11a0208e75f8000d57d4d053e80d6dedd760402730ad345c111b7";DMRFB216_SHA="df53d9bd61f1489746ae1240fffba1f8a3897d0de1416122a389e1d420def684";DMRFB_REPORT_SHA="d164c6818fcd74fefc7d542c33edd9f0883b59281ac5ec192cac3b00ef516a88";ALPHAS=(0.25,0.5,0.75);KEYS=("0.25","0.5","0.75");KD_SEEDS={"0.25":20261020,"0.5":20261021,"0.75":20261022};DMRFB_SEEDS={"0.25":20261023,"0.5":20261024,"0.75":20261025};PARAMS=4757985
def read(p):
 d=json.loads(p.read_text())
 if not isinstance(d,dict):raise ValueError(p)
 return d
def false(d):return all(d.get(x) is False for x in ("validation_access","outer_selector_access","confirmation_access","test_access"))
def finite(d):
 if isinstance(d,(int,float)) and not isinstance(d,bool):return math.isfinite(float(d))
 if isinstance(d,dict):return all(finite(x) for x in d.values())
 if isinstance(d,list):return all(finite(x) for x in d)
 return True
def generation(p,paths):
 d=read(p)
 if d.get("schema")!=GEN_SCHEMA or d.get("schema_version")!=1 or d.get("all_gates_passed") is not True or not false(d):raise ValueError("generation receipt")
 if d.get("endpoint0_checkpoint_sha256")!=KD_SHA or d.get("endpoint1_checkpoint_sha256")!=DMRFB216_SHA:raise ValueError("generation provenance")
 c=d.get("checkpoints",{})
 if set(c)!=set(KEYS):raise ValueError("generation grid")
 for k,q in paths.items():
  if c[k].get("sha256")!=sha256_file(q) or c[k].get("alpha")!=float(k) or c[k].get("state_keys")!=90 or c[k].get("parameter_count")!=PARAMS or c[k].get("checkpoint_reload_max_abs_difference")!=0 or c[k].get("fixed_input_logits_finite") is not True or not isinstance(c[k].get("interpolated_state_sha256"),str):raise ValueError(f"generated {k}")
 return d,sha256_file(p)
def dmrfb_report(p,ids):
 if sha256_file(p)!=DMRFB_REPORT_SHA:raise ValueError("DMRFB V2 SHA")
 d=read(p)
 if d.get("schema")!=DMRFB_SCHEMA or d.get("schema_version")!=1 or d.get("selected_epoch")!=216 or d.get("validation_access") is not True or d.get("outer_selector_access") is not False or d.get("confirmation_access") is not False or d.get("test_access") is not False:raise ValueError("DMRFB V2 contract")
 s=d.get("summaries",{}).get("dmrfb_dilation2",{}).get("216")
 _validate_summary(s,source="DMRFB216",expected_case_ids=ids)
 if [float(s["locator_dsc_mean"]),float(s["locator_subject_dsc_mean"]),float(s["automatic_point_hit_rate"])]!=[0.7342424767988699,0.7247423619386696,0.9259259259259259]:raise ValueError("DMRFB216 metrics")
 return d,s
def load(p,key,sha,device):
 if sha256_file(p)!=sha:raise ValueError("checkpoint SHA")
 d=torch.load(p,map_location="cpu",weights_only=False)
 interpolation=d.get("interpolation",{});geometry=d.get("state_geometry",{});architecture=d.get("architecture",{})
 if d.get("schema")!=CK_SCHEMA or d.get("schema_version")!=1 or float(d.get("alpha"))!=float(key) or d.get("arm")!="dmrfb_dilation2" or architecture.get("deploy_parameter_count")!=PARAMS or geometry.get("state_keys")!=90 or interpolation.get("endpoint0_checkpoint_sha256")!=KD_SHA or interpolation.get("endpoint1_checkpoint_sha256")!=DMRFB216_SHA or interpolation.get("alpha_grid")!=list(ALPHAS) or interpolation.get("formula")!="(1-alpha)*endpoint0+alpha*endpoint1 in float32 then cast" or d.get("checkpoint_reload_max_abs_difference")!=0 or d.get("fixed_input_logits_finite") is not True or not false(d):raise ValueError("checkpoint contract")
 if not isinstance(d.get("model_state_dict"),dict) or len(d["model_state_dict"])!=90 or not finite(interpolation):raise ValueError("checkpoint state")
 m=DMRFBAdapterLocator(init_filters=16,dropout=.1,arm="dmrfb_dilation2").to(device);m.load_state_dict(d["model_state_dict"],strict=True);return m,d
def stratum(left,right,ids):
 a={x["case_id"]:float(x["locator_dsc"]) for x in left["per_case"]};b={x["case_id"]:float(x["locator_dsc"]) for x in right["per_case"]};v=[b[x]-a[x] for x in ids];return {"n":len(v),"mean_delta":sum(v)/len(v),"improved":sum(x>0 for x in v),"tied":sum(x==0 for x in v),"worse":sum(x<0 for x in v)}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--cache",type=Path,required=True);p.add_argument("--validation-manifest",type=Path,required=True);p.add_argument("--generation-receipt",type=Path,required=True);p.add_argument("--alpha025",type=Path,required=True);p.add_argument("--alpha050",type=Path,required=True);p.add_argument("--alpha075",type=Path,required=True);p.add_argument("--kd-validation",type=Path,required=True);p.add_argument("--dmrfb-validation-v2",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--num-workers",type=int,default=4);a=p.parse_args();started=a.output.with_suffix(".started.json")
 if a.output.exists() or started.exists():raise FileExistsError("interpolation validation exists")
 data,ids,_=_validate_cache_and_boundary_stratum(a.cache,a.validation_manifest);kd,kdsha=_validate_kd_report(a.kd_validation,ids);dmrfb,dmrfb216=dmrfb_report(a.dmrfb_validation_v2,ids);paths={"0.25":a.alpha025,"0.5":a.alpha050,"0.75":a.alpha075};gen,gensha=generation(a.generation_receipt,paths);device=torch.device(a.device)
 if device.type!="cuda" or not torch.cuda.is_available():raise RuntimeError("CUDA required")
 torch.cuda.set_device(device);models={};meta={}
 for k in KEYS:models[k],meta[k]=load(paths[k],k,gen["checkpoints"][k]["sha256"],device)
 write_json_atomic(started,{"schema":SCHEMA+".started","schema_version":1,"alpha_grid":list(ALPHAS),"selection":"(Scan,Subject,-alpha)","generation_receipt_sha256":gensha,"checkpoint_sha256":{k:gen["checkpoints"][k]["sha256"] for k in KEYS},"kd_validation_sha256":kdsha,"dmrfb_validation_v2_sha256":DMRFB_REPORT_SHA,"seeds":{"kd":KD_SEEDS,"dmrfb":DMRFB_SEEDS},"validation_access":True,"outer_selector_access":False,"confirmation_access":False,"test_access":False});loader=DataLoader(data,batch_size=8,shuffle=False,num_workers=a.num_workers,pin_memory=True,persistent_workers=a.num_workers>0);summ={}
 for k in KEYS:summ[k]=validate(models.pop(k),loader,device);_validate_summary(summ[k],source=f"alpha{k}",expected_case_ids=ids);torch.cuda.empty_cache()
 kd200=kd["summaries"]["kd"][str(KD_EPOCH)];vs_kd={k:_compare_segmentation(kd200,summ[k],seed=KD_SEEDS[k]) for k in KEYS};vs_dmrfb={k:_compare_segmentation(dmrfb216,summ[k],seed=DMRFB_SEEDS[k]) for k in KEYS};selected=max(KEYS,key=lambda k:(summ[k]["locator_dsc_mean"],summ[k]["locator_subject_dsc_mean"],-float(k)));s=summ[selected];dk=vs_kd[selected]["candidate_minus_comparator"];dd=vs_dmrfb[selected]["candidate_minus_comparator"]
 gates={"beat_kd_scan":s["locator_dsc_mean"]>kd200["locator_dsc_mean"],"beat_kd_subject":s["locator_subject_dsc_mean"]>kd200["locator_subject_dsc_mean"],"beat_dmrfb_scan":s["locator_dsc_mean"]>dmrfb216["locator_dsc_mean"],"beat_dmrfb_subject":s["locator_subject_dsc_mean"]>dmrfb216["locator_subject_dsc_mean"],"point101":s["automatic_point_hit_rate"]>=101/108,"kd_scan_ci":dk["scan_95ci"][0]>=-.003,"kd_subject_ci":dk["subject_95ci"][0]>=-.003,"dmrfb_scan_ci":dd["scan_95ci"][0]>=-.003,"dmrfb_subject_ci":dd["subject_95ci"][0]>=-.003,"generation":gen["all_gates_passed"]};passed=all(gates.values());r={"schema":SCHEMA,"schema_version":1,"created_at_utc":datetime.now(timezone.utc).isoformat(),"alpha_grid":list(ALPHAS),"summaries":summ,"endpoint_kd200":kd200,"endpoint_dmrfb216":dmrfb216,"interior_minus_kd200":vs_kd,"interior_minus_dmrfb216":vs_dmrfb,"boundary27_vs_kd":{k:stratum(kd200,summ[k],BOUNDARY_TOP27) for k in KEYS},"bottom27_vs_kd":{k:stratum(kd200,summ[k],KD200_BOTTOM27) for k in KEYS},"selected_alpha":float(selected),"selected_summary":s,"promotion_gates":gates,"all_gates_passed":passed,"candidate_authorized_for_test":passed,"validation_access":True,"outer_selector_access":False,"confirmation_access":False,"test_access":False};write_json_atomic(a.output,r);print(json.dumps({"selected_alpha":selected,"gates":gates},indent=2))
if __name__=="__main__":main()
