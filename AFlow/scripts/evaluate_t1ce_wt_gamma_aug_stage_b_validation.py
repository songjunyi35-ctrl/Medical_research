#!/usr/bin/env python3
"""One-shot fixed-validation comparison of fresh Full972 baseline/Gamma and archived champion."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator  # noqa:E402
from scripts.evaluate_t1ce_wt_plain_ema_validation import compare  # noqa:E402
from scripts.train_t1ce_wt_capg3d import BrainCropDataset,read_case_ids,sha256_file,validate,write_json_atomic  # noqa:E402
from scripts.train_t1ce_wt_voxel_kd_972_draw_budget import state_dict_sha256  # noqa:E402
import scripts.train_t1ce_wt_gamma_aug_stage_b_final as train  # noqa:E402
from scripts.evaluate_t1ce_wt_gamma_aug_stage_a_inner import pairing_gates as base_pairing_gates  # noqa:E402

SCHEMA="aflow.sammed3d.t1ce_wt_gamma_aug_stage_b_validation"
PREACCESS_SCHEMA=SCHEMA+".preaccess"
VAL_MANIFEST_SHA="f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb"
VAL_CACHE_SHA="9339e4575561447dd5bd99bc29e909ae1e9f566721d351a244d087a40622f322"
ARCHIVED_SHA="30cd65c78ad1ed2a2b5748652f48ec05cde3b54468b44a66eba77d062d3ce129"
ACCESS_FALSE={"train_only_inner":False,"fixed_validation108":False,"outer":False,"confirmation":False,"test270":False}

def load_report(path:Path,arm:str,runner_sha:str)->dict[str,Any]:
    d=json.loads(path.read_text())
    if d.get("schema")!=train.SCHEMA or d.get("status")!="complete" or d.get("arm")!=arm or d.get("seed")!=train.SEED or d.get("smoke") is not False or d.get("all_gates_passed") is not True or d.get("runner_sha256")!=runner_sha or d.get("access")!=ACCESS_FALSE or len(d.get("history",[]))!=168: raise ValueError("Gamma Stage B train report contract mismatch")
    if tuple(d.get("sampling_contract",{}).get(k) for k in ("population_count","requested_draws","consumed_draws","optimizer_steps"))!=(972,972,968,121): raise ValueError("Full972 compute contract mismatch")
    return d

def load_pairing(path:Path,arm:str)->dict[str,Any]:
    d=json.loads(path.read_text())
    if d.get("schema")!=train.SCHEMA+".pairing_audit" or d.get("status")!="complete" or d.get("arm")!=arm or d.get("seed")!=train.SEED or d.get("smoke") is not False or d.get("all_gates_passed") is not True or d.get("access")!=ACCESS_FALSE or d.get("sampler_epoch_count")!=168: raise ValueError("Gamma Stage B pairing contract mismatch")
    return d

def pairing_gates(pairings:dict[str,dict[str,Any]])->dict[str,bool]:
    gates=base_pairing_gates(pairings)
    gamma=pairings["gamma_candidate"]["gamma_audit"]
    gates["gamma_seed_stage_b"]=gamma.get("seed")==20261182
    return gates

def load_model(path:Path,arm:str,report:dict[str,Any],device:torch.device)->torch.nn.Module:
    if sha256_file(path)!=report["checkpoints"]["raw"]["sha256"]: raise ValueError("checkpoint SHA mismatch")
    d=torch.load(path,map_location="cpu",weights_only=False)
    if d.get("schema")!=train.CHECKPOINT_SCHEMA or d.get("arm")!=arm or d.get("epoch")!=168 or d.get("kind")!="raw_epoch168" or d.get("access")!=ACCESS_FALSE or d.get("model_state_sha256")!=state_dict_sha256(d["model_state_dict"]): raise ValueError("checkpoint contract mismatch")
    expected={"gamma":False} if arm=="baseline_control" else {"gamma":True,"probability":0.3,"range":[0.7,1.5],"invert":False,"retain_mean_population_std":True,"generator_seed":20261182,"position":"after existing augmentation"}
    if d.get("augmentation_factor")!=expected: raise ValueError("Gamma factor metadata mismatch")
    model=AutomaticPromptLocator(center_aware=False,init_filters=16,dropout=0.1).to(device);model.load_state_dict(d["model_state_dict"],strict=True);return model.eval().requires_grad_(False)

def promotion_gates(control,candidate,archived,vc,va,engineering):
    dc=vc["ema_minus_raw"];da=va["ema_minus_raw"];ch=round(candidate["automatic_point_hit_rate"]*108);rh=round(control["automatic_point_hit_rate"]*108)
    return {"beats_control_scan":candidate["locator_dsc_mean"]>control["locator_dsc_mean"],"beats_control_subject":candidate["locator_subject_dsc_mean"]>control["locator_subject_dsc_mean"],"beats_archived_scan":candidate["locator_dsc_mean"]>archived["locator_dsc_mean"],"beats_archived_subject":candidate["locator_subject_dsc_mean"]>archived["locator_subject_dsc_mean"],"point_not_lower_control":ch>=rh,"point_at_least_103":ch>=103,"scan_ci_vs_control":dc["scan_95ci"][0]>=-0.003,"subject_ci_vs_control":dc["subject_95ci"][0]>=-0.003,"scan_ci_vs_archived":da["scan_95ci"][0]>=-0.003,"subject_ci_vs_archived":da["subject_95ci"][0]>=-0.003,"engineering_pairing":engineering}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--cache",type=Path,required=True);p.add_argument("--validation-manifest",type=Path,required=True);p.add_argument("--archived-validation",type=Path,required=True)
    for arm in train.stage_a.ARMS:p.add_argument(f"--{arm}-report",type=Path,required=True);p.add_argument(f"--{arm}-pairing",type=Path,required=True);p.add_argument(f"--{arm}-checkpoint",type=Path,required=True)
    p.add_argument("--preaccess-receipt",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0");a=p.parse_args();started=a.output.with_suffix(".started.json")
    if a.output.exists() or started.exists(): raise FileExistsError("refusing repeated Gamma Stage B validation")
    evaluator_sha=sha256_file(Path(__file__).resolve());runner_sha=sha256_file(ROOT/"scripts/train_t1ce_wt_gamma_aug_stage_b_final.py")
    if sha256_file(a.validation_manifest)!=VAL_MANIFEST_SHA or sha256_file(a.cache/"index.json")!=VAL_CACHE_SHA or sha256_file(a.archived_validation)!=ARCHIVED_SHA: raise ValueError("frozen validation input SHA mismatch")
    arms=train.stage_a.ARMS;rp={x:getattr(a,f"{x}_report") for x in arms};pp={x:getattr(a,f"{x}_pairing") for x in arms};cp={x:getattr(a,f"{x}_checkpoint") for x in arms}
    reports={x:load_report(rp[x],x,runner_sha) for x in arms};pairings={x:load_pairing(pp[x],x) for x in arms};engineering=pairing_gates(pairings)
    if not all(engineering.values()): raise ValueError("engineering pairing failed prevalidation")
    hashes={"report":{x:sha256_file(rp[x]) for x in arms},"pairing":{x:sha256_file(pp[x]) for x in arms},"checkpoint":{x:sha256_file(cp[x]) for x in arms}}
    receipt=json.loads(a.preaccess_receipt.read_text())
    if receipt.get("schema")!=PREACCESS_SCHEMA or receipt.get("status")!="frozen_before_fixed_validation" or receipt.get("evaluator_sha256")!=evaluator_sha or receipt.get("runner_sha256")!=runner_sha or receipt.get("artifact_sha256")!=hashes or receipt.get("archived_validation_sha256")!=ARCHIVED_SHA or receipt.get("access")!=ACCESS_FALSE: raise ValueError("preaccess receipt mismatch")
    ids=read_case_ids(a.validation_manifest);data=BrainCropDataset(a.cache,expected_split="validation",case_ids=ids)
    if len(data)!=108 or len({x.rsplit("-",1)[0] for x in ids})!=50: raise ValueError("validation cohort mismatch")
    write_json_atomic(started,{"schema":SCHEMA+".started","schema_version":1,"created_at_utc":datetime.now(timezone.utc).isoformat(),"preaccess_receipt_sha256":sha256_file(a.preaccess_receipt),"train_only_inner_access":True,"fixed_validation108_access":True,"outer_access":False,"confirmation_access":False,"test270_access":False})
    device=torch.device(a.device)
    if device.type!="cuda" or not torch.cuda.is_available():raise RuntimeError("validation requires CUDA")
    torch.cuda.set_device(device);loader=DataLoader(data,batch_size=8,shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True);summaries={}
    for arm in arms:model=load_model(cp[arm],arm,reports[arm],device);summaries[arm]=validate(model,loader,device);del model;torch.cuda.empty_cache()
    archived=json.loads(a.archived_validation.read_text())["summaries"]["full972_candidate"];vc=compare(summaries["baseline_control"],summaries["gamma_candidate"],seed=20261084,samples=10000);va=compare(archived,summaries["gamma_candidate"],seed=20261084,samples=10000);gates=promotion_gates(summaries["baseline_control"],summaries["gamma_candidate"],archived,vc,va,all(engineering.values()))
    result={"schema":SCHEMA,"schema_version":1,"status":"complete","created_at_utc":datetime.now(timezone.utc).isoformat(),"summaries":summaries,"archived_full972_seed32":archived,"candidate_minus_control":vc,"candidate_minus_archived":va,"engineering_pairing_gates":engineering,"gamma_audit":pairings["gamma_candidate"]["gamma_audit"],"gates":gates,"all_gates_passed":all(gates.values()),"candidate_authorized_for_test":all(gates.values()),"inputs":{**hashes,"preaccess_receipt_sha256":sha256_file(a.preaccess_receipt),"evaluator_sha256":evaluator_sha,"runner_sha256":runner_sha},"access":{"train_only_inner":True,"fixed_validation108":True,"outer":False,"confirmation":False,"test270":False}}
    write_json_atomic(a.output,result);print(json.dumps({"control":{k:summaries["baseline_control"][k] for k in ("locator_dsc_mean","locator_subject_dsc_mean","automatic_point_hit_rate")},"candidate":{k:summaries["gamma_candidate"][k] for k in ("locator_dsc_mean","locator_subject_dsc_mean","automatic_point_hit_rate")},"archived":{k:archived[k] for k in ("locator_dsc_mean","locator_subject_dsc_mean","automatic_point_hit_rate")},"gates":gates,"candidate_authorized_for_test":result["candidate_authorized_for_test"]},indent=2,sort_keys=True))
if __name__=="__main__":main()
