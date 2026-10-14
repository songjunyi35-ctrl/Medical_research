#!/usr/bin/env python3
"""Evaluate paired DMRFB adapter arms on frozen validation108."""
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from sammed3d.t1ce_wt_dmrfb_adapter import DMRFBAdapterLocator
from scripts.train_t1ce_wt_capg3d import sha256_file,validate,write_json_atomic
from scripts.evaluate_t1ce_wt_plain_vm_dice_validation import BOUNDARY_TOP27,BOUNDARY_TOP27_SHA256,KD200_BOTTOM27,KD200_BOTTOM27_SHA256,KD_EPOCH,EPOCHS,GPU_MEMORY_STOP_BYTES,PAIRING_FIELDS,VALIDATION_CASES,_compare_segmentation,_compare_stratum,_validate_cache_and_boundary_stratum,_validate_kd_report,_validate_summary
SCHEMA="aflow.sammed3d.t1ce_wt_plain_dmrfb_adapter_validation";CK="aflow.sammed3d.t1ce_wt_dmrfb_adapter_checkpoint";TRAIN="aflow.sammed3d.t1ce_wt_plain_dmrfb_adapter_continuation";IMPL="aflow.sammed3d.t1ce_wt_dmrfb_adapter_implementation_receipt";PAIR="aflow.sammed3d.t1ce_wt_dmrfb_adapter_pairing_receipt";ARMS=("same_rf_control","dmrfb_dilation2");CTRL={208:20261000,216:20261001,224:20261002,232:20261003};KD={208:20261004,216:20261005,224:20261006,232:20261007};BOUND={208:20261008,216:20261009,224:20261010,232:20261011};BOTTOM={208:20261012,216:20261013,224:20261014,232:20261015}
def read(p):return json.loads(p.read_text())
def false(d):return all(d.get(x) is False for x in ("validation_access","outer_selector_access","confirmation_access","test_access"))
def train(p,arm):
 d=read(p)
 if d.get("schema")!=TRAIN or d.get("arm")!=arm or d.get("status")!="complete" or d.get("all_self_gates_passed") is not True or not false(d) or int(d.get("peak_gpu_memory_bytes",GPU_MEMORY_STOP_BYTES+1))>GPU_MEMORY_STOP_BYTES:raise ValueError("training")
 return d,sha256_file(p)
def main():
 p=argparse.ArgumentParser();p.add_argument("--cache",type=Path,required=True);p.add_argument("--validation-manifest",type=Path,required=True);p.add_argument("--control-dir",type=Path,required=True);p.add_argument("--candidate-dir",type=Path,required=True);p.add_argument("--control-report",type=Path,required=True);p.add_argument("--candidate-report",type=Path,required=True);p.add_argument("--pairing-receipt",type=Path,required=True);p.add_argument("--implementation-receipt",type=Path,required=True);p.add_argument("--kd-validation",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--num-workers",type=int,default=4);a=p.parse_args();started=a.output.with_suffix(".started.json")
 if a.output.exists() or started.exists():raise FileExistsError("validation exists")
 data,ids,_=_validate_cache_and_boundary_stratum(a.cache,a.validation_manifest);kd,kdsha=_validate_kd_report(a.kd_validation,ids);implementation=read(a.implementation_receipt)
 if implementation.get("schema")!=IMPL or not false(implementation) or any(implementation.get(x) is not True for x in ("synthetic_all_gates_passed","gpu_smoke_all_gates_passed","identity_all_gates_passed","two_step_gradient_all_gates_passed","all_gates_passed")) or implementation.get("implementation",{}).get("validation_evaluator_sha256")!=sha256_file(Path(__file__).resolve()):raise ValueError("implementation")
 paths={ARMS[0]:a.control_report,ARMS[1]:a.candidate_report};reports={};rsha={}
 for arm in ARMS:reports[arm],rsha[arm]=train(paths[arm],arm)
 pair=read(a.pairing_receipt)
 if pair.get("schema")!=PAIR or pair.get("all_gates_passed") is not True or not false(pair) or any(pair["arms"][ARMS[0]]["pairing"][f]!=pair["arms"][ARMS[1]]["pairing"][f] for f in PAIRING_FIELDS):raise ValueError("pairing")
 device=torch.device(a.device)
 if device.type!="cuda" or not torch.cuda.is_available():raise RuntimeError("CUDA")
 torch.cuda.set_device(device);dirs={ARMS[0]:a.control_dir,ARMS[1]:a.candidate_dir};models={x:{} for x in ARMS};hashes={x:{} for x in ARMS}
 for arm in ARMS:
  for e in EPOCHS:
   q=dirs[arm]/f"plain_128_{arm}_global_epoch{e}.pth";h=reports[arm]["checkpoints"][str(e)]["sha256"]
   if sha256_file(q)!=h:raise ValueError("checkpoint SHA")
   d=torch.load(q,map_location="cpu",weights_only=False)
   if d.get("schema")!=CK or d.get("arm")!=arm or d.get("global_epoch")!=e or not false(d) or d.get("checkpoint_reload_max_abs_difference")!=0 or d.get("deploy_max_abs_difference")!=0:raise ValueError("checkpoint")
   m=DMRFBAdapterLocator(init_filters=16,dropout=.1,arm=arm).to(device);m.load_state_dict(d["model_state_dict"],strict=True);models[arm][e]=m;hashes[arm][e]=h
 write_json_atomic(started,{"schema":SCHEMA+".started","selection":"(Scan,Subject,-epoch)","checkpoint_sha256":hashes,"bottom27_sha256":KD200_BOTTOM27_SHA256,"boundary27_sha256":BOUNDARY_TOP27_SHA256,"validation_access":True,"outer_selector_access":False,"confirmation_access":False,"test_access":False});loader=DataLoader(data,batch_size=8,shuffle=False,num_workers=a.num_workers,pin_memory=True,persistent_workers=a.num_workers>0);s={x:{} for x in ARMS}
 for arm in ARMS:
  for e in EPOCHS:s[arm][str(e)]=validate(models[arm].pop(e),loader,device);_validate_summary(s[arm][str(e)],source=f"{arm}{e}",expected_case_ids=ids);torch.cuda.empty_cache()
 selected=max(EPOCHS,key=lambda e:(s[ARMS[1]][str(e)]["locator_dsc_mean"],s[ARMS[1]][str(e)]["locator_subject_dsc_mean"],-e));k=str(selected);kd200=kd["summaries"]["kd"][str(KD_EPOCH)];vc={str(e):_compare_segmentation(s[ARMS[0]][str(e)],s[ARMS[1]][str(e)],seed=CTRL[e]) for e in EPOCHS};vk={str(e):_compare_segmentation(kd200,s[ARMS[1]][str(e)],seed=KD[e]) for e in EPOCHS};bnd={str(e):_compare_stratum(s[ARMS[0]][str(e)],s[ARMS[1]][str(e)],case_ids=BOUNDARY_TOP27,seed=BOUND[e]) for e in EPOCHS};bot={str(e):_compare_stratum(s[ARMS[0]][str(e)],s[ARMS[1]][str(e)],case_ids=KD200_BOTTOM27,seed=BOTTOM[e]) for e in EPOCHS};c=s[ARMS[1]][k];z=s[ARMS[0]][k];dc=vc[k]["candidate_minus_comparator"];dk=vk[k]["candidate_minus_comparator"]
 g={"control_scan":c["locator_dsc_mean"]>z["locator_dsc_mean"],"control_subject":c["locator_subject_dsc_mean"]>z["locator_subject_dsc_mean"],"kd_scan":c["locator_dsc_mean"]>kd200["locator_dsc_mean"],"kd_subject":c["locator_subject_dsc_mean"]>kd200["locator_subject_dsc_mean"],"point101":c["automatic_point_hit_rate"]>=101/VALIDATION_CASES,"control_scan_ci":dc["scan_95ci"][0]>=-.003,"control_subject_ci":dc["subject_95ci"][0]>=-.003,"kd_scan_ci":dk["scan_95ci"][0]>=-.003,"kd_subject_ci":dk["subject_95ci"][0]>=-.003,"boundary27":bnd[k]["gamma1_minus_gamma0_mean"]>0,"bottom27":bot[k]["gamma1_minus_gamma0_mean"]>=0};passed=all(g.values());r={"schema":SCHEMA,"schema_version":1,"created_at_utc":datetime.now(timezone.utc).isoformat(),"summaries":s,"historical_kd200":kd200,"candidate_minus_control":vc,"candidate_minus_kd200":vk,"boundary27":bnd,"bottom27":bot,"selected_epoch":selected,"selected_summary":c,"promotion_gates":g,"all_gates_passed":passed,"candidate_authorized_for_test":passed,"validation_access":True,"outer_selector_access":False,"confirmation_access":False,"test_access":False};write_json_atomic(a.output,r);print(json.dumps({"selected":selected,"gates":g},indent=2))
if __name__=="__main__":main()
