#!/usr/bin/env python3
"""Freeze paired DMRFB adapter formal evidence."""
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.train_t1ce_wt_capg3d import sha256_file,write_json_atomic
SCHEMA="aflow.sammed3d.t1ce_wt_dmrfb_adapter_pairing_receipt";TRAIN="aflow.sammed3d.t1ce_wt_plain_dmrfb_adapter_continuation";ARMS=("same_rf_control","dmrfb_dilation2");EPOCHS=(208,216,224,232);GLOBAL={str(x) for x in range(201,233)};LIMIT=24*1024**3;FIELDS=("initial_student_state_sha256","initial_adapter_state_sha256","first_batch_case_ids","first_batch_case_ids_sha256","first_batch_raw_image_sha256","first_batch_raw_label_sha256","first_batch_augmented_image_sha256","first_batch_augmented_label_sha256","epoch_case_order_sha256","epoch_cuda_rng_before_sha256","epoch_cuda_rng_after_sha256")
def read(p,arm):
 d=json.loads(p.read_text())
 if d.get("schema")!=TRAIN or d.get("arm")!=arm or d.get("status")!="complete" or d.get("all_self_gates_passed") is not True or int(d.get("peak_gpu_memory_bytes",LIMIT+1))>LIMIT or float(d.get("elapsed_seconds",2701))>2700:raise ValueError(arm)
 if any(d.get(k) is not False for k in ("validation_access","outer_selector_access","confirmation_access","test_access")):raise ValueError("access")
 if set(d.get("checkpoints",{}))!={str(x) for x in EPOCHS} or any(set(d["pairing"][f])!=GLOBAL for f in FIELDS[-3:]):raise ValueError("grid")
 for e in EPOCHS:
  x=d["checkpoints"][str(e)];q=Path(x["path"])
  if not q.is_file() or sha256_file(q)!=x["sha256"] or x["checkpoint_reload_max_abs_difference"]!=0 or x["deploy_max_abs_difference"]!=0:raise ValueError("checkpoint")
 return d
def main():
 p=argparse.ArgumentParser();p.add_argument("--control-report",type=Path,required=True);p.add_argument("--candidate-report",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();paths={ARMS[0]:a.control_report,ARMS[1]:a.candidate_report};d={x:read(paths[x],x) for x in ARMS};m=[f for f in FIELDS if d[ARMS[0]]["pairing"][f]!=d[ARMS[1]]["pairing"][f]]
 if m:raise RuntimeError(m)
 arms={x:{"training_report":str(paths[x].resolve()),"training_report_sha256":sha256_file(paths[x]),"checkpoints":d[x]["checkpoints"],"pairing":d[x]["pairing"],"peak_gpu_memory_bytes":d[x]["peak_gpu_memory_bytes"],"elapsed_seconds":d[x]["elapsed_seconds"]} for x in ARMS};r={"schema":SCHEMA,"schema_version":1,"status":"complete_before_validation_access","created_at_utc":datetime.now(timezone.utc).isoformat(),"arms":arms,"mismatched_fields":m,"parallel_wall_seconds":max(x["elapsed_seconds"] for x in arms.values()),"all_gates_passed":True,"validation_access":False,"outer_selector_access":False,"confirmation_access":False,"test_access":False};write_json_atomic(a.output,r);print(json.dumps({"all_gates_passed":True}))
if __name__=="__main__":main()
