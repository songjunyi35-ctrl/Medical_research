#!/usr/bin/env python3
"""Freeze paired uniform/CVaR formal evidence before validation access."""
from __future__ import annotations
import argparse,json,math,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.train_t1ce_wt_capg3d import sha256_file,write_json_atomic  # noqa:E402
SCHEMA="aflow.sammed3d.t1ce_wt_batch_cvar50_pairing_receipt";TRAIN_SCHEMA="aflow.sammed3d.t1ce_wt_plain_batch_cvar50_continuation";ARMS=("uniform_mean8","batch_cvar50_top4");EPOCHS=(208,216,224,232);GLOBAL=tuple(range(201,233));LIMIT=24*1024**3
FIELDS=("initial_student_state_sha256","first_batch_case_ids","first_batch_case_ids_sha256","first_batch_raw_image_sha256","first_batch_raw_label_sha256","first_batch_augmented_image_sha256","first_batch_augmented_label_sha256","epoch_case_order_sha256","epoch_cuda_rng_before_sha256","epoch_cuda_rng_after_sha256")
def zero(v):return math.isfinite(float(v)) and float(v)==0
def read(p,arm):
 d=json.loads(p.read_text());src=f"{arm} report"
 if d.get("schema")!=TRAIN_SCHEMA or d.get("schema_version")!=1 or d.get("arm")!=arm or d.get("status")!="complete" or d.get("all_self_gates_passed") is not True:raise ValueError(src)
 if any(d.get(k) is not False for k in ("validation_access","outer_selector_access","confirmation_access","test_access")):raise ValueError(src+" access")
 if not zero(d.get("reload_max_abs_difference")) or not zero(d.get("stripped_deploy_max_abs_difference")):raise ValueError(src+" reload")
 if int(d.get("peak_gpu_memory_bytes",LIMIT+1))>LIMIT or float(d.get("elapsed_seconds",2701))>2700:raise ValueError(src+" cost")
 if [x.get("global_epoch") for x in d.get("history",[])]!=list(GLOBAL) or any(float(x.get("learning_rate"))!=1e-5 for x in d["history"]):raise ValueError(src+" history")
 if set(d.get("checkpoints",{}))!={str(x) for x in EPOCHS}:raise ValueError(src+" grid")
 for e in EPOCHS:
  x=d["checkpoints"][str(e)];q=Path(x["path"])
  if not q.is_file() or sha256_file(q)!=x.get("sha256") or not zero(x.get("checkpoint_reload_max_abs_difference")) or not zero(x.get("stripped_deploy_max_abs_difference")):raise ValueError(src+f" ckpt{e}")
 pair=d.get("pairing",{});
 if any(not pair.get(f) for f in FIELDS):raise ValueError(src+" pairing")
 if any(set(pair[f])!={str(x) for x in GLOBAL} for f in FIELDS[-3:]):raise ValueError(src+" epochs")
 s=d.get("sampling_contract",{});
 if (s.get("weighted_sampler_num_samples"),s.get("optimizer_consumed_draws"),s.get("dropped_draws"),s.get("steps"))!=(756,752,4,94):raise ValueError(src+" sampler")
 return d
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--uniform-report",type=Path,required=True);p.add_argument("--candidate-report",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError("receipt exists")
 paths={ARMS[0]:a.uniform_report,ARMS[1]:a.candidate_report};reports={x:read(paths[x],x) for x in ARMS};m=[f for f in FIELDS if reports[ARMS[0]]["pairing"][f]!=reports[ARMS[1]]["pairing"][f]]
 if m:raise RuntimeError(f"pair mismatch {m}")
 arms={x:{"training_report":str(paths[x].resolve()),"training_report_sha256":sha256_file(paths[x]),"checkpoints":reports[x]["checkpoints"],"pairing":reports[x]["pairing"],"peak_gpu_memory_bytes":reports[x]["peak_gpu_memory_bytes"],"elapsed_seconds":reports[x]["elapsed_seconds"],"reload_max_abs_difference":0.0,"stripped_deploy_max_abs_difference":0.0} for x in ARMS}
 r={"schema":SCHEMA,"schema_version":1,"status":"complete_before_validation_access","created_at_utc":datetime.now(timezone.utc).isoformat(),"arms":arms,"mismatched_fields":m,"parallel_wall_seconds":max(x["elapsed_seconds"] for x in arms.values()),"all_gates_passed":True,"validation_access":False,"outer_selector_access":False,"confirmation_access":False,"test_access":False};write_json_atomic(a.output,r);print(json.dumps({"all_gates_passed":True,"output":str(a.output)},indent=2))
if __name__=="__main__":main()
