#!/usr/bin/env python3
"""Evaluate paired uniform mean8 and Batch-CVaR50 top4 on validation108."""
from __future__ import annotations
import argparse,json,math,sys
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator  # noqa:E402
from scripts.train_t1ce_wt_capg3d import sha256_file,validate,write_json_atomic  # noqa:E402
from scripts.evaluate_t1ce_wt_plain_vm_dice_validation import BOUNDARY_TOP27,BOUNDARY_TOP27_SHA256,KD200_BOTTOM27,KD200_BOTTOM27_SHA256,KD_EPOCH,EPOCHS as GRID,EXPECTED_ARCHITECTURE,GPU_MEMORY_STOP_BYTES,KD_VALIDATION_SHA256,PAIRING_FIELDS,START_CHECKPOINT_SHA256,TRAINING_GLOBAL_EPOCHS,VALIDATION_CACHE_INDEX_SHA256,VALIDATION_CASES,VALIDATION_LABELS_SHA256,VALIDATION_MANIFEST_SHA256,VALIDATION_SUBJECTS,_compare_segmentation,_compare_stratum,_validate_cache_and_boundary_stratum,_validate_kd_report,_validate_summary  # noqa:E402
SCHEMA="aflow.sammed3d.t1ce_wt_plain_batch_cvar50_validation";CK_SCHEMA="aflow.sammed3d.t1ce_wt_batch_cvar50_checkpoint";TRAIN_SCHEMA="aflow.sammed3d.t1ce_wt_plain_batch_cvar50_continuation";IMPL_SCHEMA="aflow.sammed3d.t1ce_wt_batch_cvar50_implementation_receipt";PAIR_SCHEMA="aflow.sammed3d.t1ce_wt_batch_cvar50_pairing_receipt";ARMS=("uniform_mean8","batch_cvar50_top4");EPOCHS=tuple(GRID);SAMPLES=10000
IDEA_SHA="a4ae824d1532981c52d72baf3985f9089bee6c35ebcf5d878a072494d11ecea4";PREREG_SHA="a0935815c13be95055d0b5a67cb3ce800483bd3ccc2fd30859b9919a6bbcaa70";AMD1_SHA="81c38d1efd551e79b56e508e5483a5c929cb7b9f8c5bcf4f03833680eb1151ee";AMD2_SHA="db188606b1715537a92654a38f5bd0523f7baa1f552fdc3630edff2d4b00fcec"
VM_EVALUATOR_HELPER_SHA="19d434e9394c385406d4649a1f9763a343e5ace575acec2247f0dbfd43d21446"
SEED_CTRL={208:20260970,216:20260971,224:20260972,232:20260973};SEED_KD={208:20260974,216:20260975,224:20260976,232:20260977};SEED_BOTTOM={208:20260978,216:20260979,224:20260980,232:20260981};SEED_BOUND={208:20260982,216:20260983,224:20260984,232:20260985}
def read(p):
 d=json.loads(p.read_text());
 if not isinstance(d,dict):raise ValueError(p)
 return d
def false(d):return all(d.get(x) is False for x in ("validation_access","outer_selector_access","confirmation_access","test_access"))
def zero(v):return math.isfinite(float(v)) and float(v)==0
def contract(arm):return {"arm":arm,"dice":{"include_background":True,"to_onehot_y":False,"sigmoid":True,"softmax":False,"squared_pred":False,"jaccard":False,"smooth_nr":1e-5,"smooth_dr":1e-5,"batch":False,"weight":None,"reduction":"none"},"bce":{"weight":None,"pos_weight":None,"reduction":"none","coefficient":0.1},"aggregation":"mean8" if arm==ARMS[0] else "stable detached top4-of8 mean","topk":8 if arm==ARMS[0] else 4,"teacher_forwards":0,"deploy":"plain final raw logit"}
def impl(p):
 d=read(p)
 if d.get("schema")!=IMPL_SCHEMA or d.get("schema_version")!=1 or not false(d) or any(d.get(x) is not True for x in ("synthetic_all_gates_passed","gpu_smoke_all_gates_passed","legacy_parity_all_gates_passed","all_gates_passed")):raise ValueError("implementation receipt")
 i=d.get("implementation",{});
 if i.get("validation_evaluator_sha256")!=sha256_file(Path(__file__).resolve()) or i.get("trainer_path")!="scripts/train_t1ce_wt_plain_batch_cvar50_continuation.py":raise ValueError("implementation bindings")
 return d,sha256_file(p)
def train(p,arm):
 d=read(p)
 if d.get("schema")!=TRAIN_SCHEMA or d.get("schema_version")!=1 or d.get("arm")!=arm or d.get("status")!="complete" or d.get("all_self_gates_passed") is not True or not false(d):raise ValueError(f"{arm} report")
 if int(d.get("peak_gpu_memory_bytes",GPU_MEMORY_STOP_BYTES+1))>GPU_MEMORY_STOP_BYTES or float(d.get("elapsed_seconds",2701))>2700 or not zero(d.get("reload_max_abs_difference")) or not zero(d.get("stripped_deploy_max_abs_difference")):raise ValueError(f"{arm} formal gates")
 if d.get("batch_cvar50")!=contract(arm) or d.get("start_checkpoint_sha256")!=START_CHECKPOINT_SHA256:raise ValueError(f"{arm} contract")
 for k,v in (("idea_sha256",IDEA_SHA),("preregistration_sha256",PREREG_SHA),("amendment_sha256",AMD1_SHA),("amendment2_sha256",AMD2_SHA)):
  if d.get(k)!=v:raise ValueError(f"{arm} {k}")
 if [x.get("global_epoch") for x in d.get("history",[])]!=list(TRAINING_GLOBAL_EPOCHS) or set(d.get("checkpoints",{}))!={str(x) for x in EPOCHS}:raise ValueError(f"{arm} schedule")
 return d,sha256_file(p)
def pairing(p,reports,shas):
 d=read(p)
 if d.get("schema")!=PAIR_SCHEMA or d.get("schema_version")!=1 or d.get("all_gates_passed") is not True or not false(d):raise ValueError("pair receipt")
 a=d.get("arms",{})
 if set(a)!=set(ARMS):raise ValueError("pair arms")
 for arm in ARMS:
  if a[arm].get("training_report_sha256")!=shas[arm] or a[arm].get("pairing")!=reports[arm]["pairing"]:raise ValueError("pair binding")
 if any(a[ARMS[0]]["pairing"][f]!=a[ARMS[1]]["pairing"][f] for f in PAIRING_FIELDS):raise ValueError("pair mismatch")
 return d,sha256_file(p)
def load(path,arm,epoch,sha,device):
 if sha256_file(path)!=sha:raise ValueError("checkpoint SHA")
 d=torch.load(path,map_location="cpu",weights_only=False)
 if d.get("schema")!=CK_SCHEMA or d.get("schema_version")!=1 or d.get("arm")!=arm or d.get("global_epoch")!=epoch or d.get("continuation_epoch")!=epoch-200 or d.get("batch_cvar50")!=contract(arm) or d.get("architecture")!=EXPECTED_ARCHITECTURE or not false(d) or not zero(d.get("checkpoint_reload_max_abs_difference")) or not zero(d.get("stripped_deploy_max_abs_difference")):raise ValueError(f"{arm} checkpoint{epoch}")
 m=AutomaticPromptLocator(center_aware=False,init_filters=16,dropout=.1).to(device);m.load_state_dict(d["model_state_dict"],strict=True);return m,d
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--cache",type=Path,required=True);p.add_argument("--validation-manifest",type=Path,required=True);p.add_argument("--uniform-dir",type=Path,required=True);p.add_argument("--candidate-dir",type=Path,required=True);p.add_argument("--uniform-report",type=Path,required=True);p.add_argument("--candidate-report",type=Path,required=True);p.add_argument("--pairing-receipt",type=Path,required=True);p.add_argument("--implementation-receipt",type=Path,required=True);p.add_argument("--kd-validation",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--batch-size",type=int,default=8);p.add_argument("--num-workers",type=int,default=4);a=p.parse_args();started=a.output.with_suffix(".started.json")
 if sha256_file(ROOT/"scripts/evaluate_t1ce_wt_plain_vm_dice_validation.py")!=VM_EVALUATOR_HELPER_SHA:raise ValueError("frozen validation helper SHA mismatch")
 if a.output.exists() or started.exists() or a.batch_size!=8:raise ValueError("output/batch")
 data,ids,_=_validate_cache_and_boundary_stratum(a.cache,a.validation_manifest);kd,kdsha=_validate_kd_report(a.kd_validation,ids);implementation,ish=impl(a.implementation_receipt);paths={ARMS[0]:a.uniform_report,ARMS[1]:a.candidate_report};reports={};shas={}
 for arm in ARMS:reports[arm],shas[arm]=train(paths[arm],arm)
 if implementation["implementation"].get("trainer_sha256")!=reports[ARMS[1]].get("runner_sha256"):raise ValueError("trainer SHA")
 pair,psh=pairing(a.pairing_receipt,reports,shas);device=torch.device(a.device)
 if device.type!="cuda" or not torch.cuda.is_available():raise RuntimeError("CUDA required")
 torch.cuda.set_device(device);dirs={ARMS[0]:a.uniform_dir,ARMS[1]:a.candidate_dir};models={x:{} for x in ARMS};meta={x:{} for x in ARMS};hashes={x:{} for x in ARMS}
 for arm in ARMS:
  for e in EPOCHS:
   q=dirs[arm]/f"plain_128_{arm}_global_epoch{e}.pth";h=reports[arm]["checkpoints"][str(e)]["sha256"];models[arm][e],meta[arm][e]=load(q,arm,e,h,device);hashes[arm][e]=h
 write_json_atomic(started,{"schema":SCHEMA+".started","schema_version":1,"selection":"(candidate Scan,Subject,-epoch)","bootstrap_samples":SAMPLES,"seeds":{"control":SEED_CTRL,"kd":SEED_KD,"bottom":SEED_BOTTOM,"boundary":SEED_BOUND},"bottom27_sha256":KD200_BOTTOM27_SHA256,"boundary27_sha256":BOUNDARY_TOP27_SHA256,"training_report_sha256":shas,"pairing_receipt_sha256":psh,"implementation_receipt_sha256":ish,"checkpoint_sha256":hashes,"validation_access":True,"outer_selector_access":False,"confirmation_access":False,"test_access":False})
 loader=DataLoader(data,batch_size=8,shuffle=False,num_workers=a.num_workers,pin_memory=True,persistent_workers=a.num_workers>0);summ={x:{} for x in ARMS}
 for arm in ARMS:
  for e in EPOCHS:summ[arm][str(e)]=validate(models[arm].pop(e),loader,device);_validate_summary(summ[arm][str(e)],source=f"{arm}{e}",expected_case_ids=ids);torch.cuda.empty_cache()
 selected=max(EPOCHS,key=lambda e:(summ[ARMS[1]][str(e)]["locator_dsc_mean"],summ[ARMS[1]][str(e)]["locator_subject_dsc_mean"],-e));key=str(selected);kd200=kd["summaries"]["kd"][str(KD_EPOCH)]
 vc={str(e):_compare_segmentation(summ[ARMS[0]][str(e)],summ[ARMS[1]][str(e)],seed=SEED_CTRL[e]) for e in EPOCHS};vk={str(e):_compare_segmentation(kd200,summ[ARMS[1]][str(e)],seed=SEED_KD[e]) for e in EPOCHS};bot={str(e):_compare_stratum(summ[ARMS[0]][str(e)],summ[ARMS[1]][str(e)],case_ids=KD200_BOTTOM27,seed=SEED_BOTTOM[e]) for e in EPOCHS};bnd={str(e):_compare_stratum(summ[ARMS[0]][str(e)],summ[ARMS[1]][str(e)],case_ids=BOUNDARY_TOP27,seed=SEED_BOUND[e]) for e in EPOCHS};s=summ[ARMS[1]][key];c=summ[ARMS[0]][key];dc=vc[key]["candidate_minus_comparator"];dk=vk[key]["candidate_minus_comparator"]
 prereq={"implementation":implementation["all_gates_passed"],"legacy_parity":implementation["legacy_parity_all_gates_passed"],"pairing":pair["all_gates_passed"],"formal":all(reports[x]["all_self_gates_passed"] for x in ARMS),"reload":all(zero(reports[x]["reload_max_abs_difference"]) for x in ARMS)};g={"beat_control_scan":s["locator_dsc_mean"]>c["locator_dsc_mean"],"beat_control_subject":s["locator_subject_dsc_mean"]>c["locator_subject_dsc_mean"],"beat_kd_scan":s["locator_dsc_mean"]>kd200["locator_dsc_mean"],"beat_kd_subject":s["locator_subject_dsc_mean"]>kd200["locator_subject_dsc_mean"],"point101":s["automatic_point_hit_rate"]>=101/108,"control_scan_ci":dc["scan_95ci"][0]>=-.003,"control_subject_ci":dc["subject_95ci"][0]>=-.003,"kd_scan_ci":dk["scan_95ci"][0]>=-.003,"kd_subject_ci":dk["subject_95ci"][0]>=-.003,"bottom27":bot[key]["gamma1_minus_gamma0_mean"]>0,"boundary27":bnd[key]["gamma1_minus_gamma0_mean"]>=0,"prerequisites":all(prereq.values())};passed=all(g.values())
 r={"schema":SCHEMA,"schema_version":1,"created_at_utc":datetime.now(timezone.utc).isoformat(),"protocol":{"selection":"maximum candidate Scan,Subject,earlier epoch","threshold":0.0,"tta":False,"sam":False,"postprocessing":False,"bootstrap_samples":SAMPLES,"quantile_method":"linear"},"summaries":summ,"historical_kd200":kd200,"candidate_minus_control":vc,"candidate_minus_kd200":vk,"bottom27":bot,"boundary27":bnd,"selected_epoch":selected,"selected_summary":s,"selected_control":c,"prerequisite_gates":prereq,"promotion_gates":g,"all_gates_passed":passed,"candidate_authorized_for_test":passed,"validation_access":True,"outer_selector_access":False,"confirmation_access":False,"test_access":False};write_json_atomic(a.output,r);print(json.dumps({"selected_epoch":selected,"promotion_gates":g,"authorized":passed},indent=2))
if __name__=="__main__":main()
