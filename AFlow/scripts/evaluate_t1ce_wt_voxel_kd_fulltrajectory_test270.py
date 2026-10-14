#!/usr/bin/env python3
"""User-authorized failed-gate exploratory test270 for frozen full972 KD168."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator  # noqa: E402
from scripts.evaluate_t1ce_wt_plain_ema_validation import compare  # noqa: E402
from scripts.evaluate_t1ce_wt_plain_fixed_test270 import intervals  # noqa: E402
from scripts.evaluate_t1ce_wt_plain_locator_test270 import TEST_CASES, TEST_MANIFEST_SHA256, TEST_SUBJECTS, TestBrainCropDataset  # noqa: E402
from scripts.train_t1ce_wt_capg3d import sha256_file, validate, write_json_atomic  # noqa: E402
import scripts.train_t1ce_wt_voxel_kd_fulltrajectory_final as train  # noqa: E402
SCHEMA="aflow.sammed3d.t1ce_wt_voxel_kd_fulltrajectory_failed_gate_test270"
CHECKPOINT_SHA="f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b"
VALIDATION_SHA="30cd65c78ad1ed2a2b5748652f48ec05cde3b54468b44a66eba77d062d3ce129"
HISTORICAL_SHA="91433c4b6b88b233d827e41b0d038f459c25ef41dec224c84e154e90f370bb8c"
KD200_SHA="36a72112595c8606537d2182d31a2558ea982e2f327a82c7a8cc6d4b079e7b0e"
ENSEMBLE_SHA="b49cf72a7d749afd28bc3009d0f7ad27b63ce5642424cc8eeb183e4eec7dbc7c"
OVERRIDE_SCHEMA="aflow.brats.t1ce_wt_voxel_kd_fulltrajectory_user_test_override"
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ("cache","test_manifest","checkpoint","validation_report","historical_report","kd200_report","ensemble_report","override_receipt","output"): p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
 p.add_argument("--device",default="cuda:0"); p.add_argument("--num-workers",type=int,default=4); a=p.parse_args()
 started=a.output.with_suffix(".started.json")
 if a.output.exists() or started.exists(): raise FileExistsError("refusing repeated exploratory test270")
 expected=((a.checkpoint,CHECKPOINT_SHA),(a.validation_report,VALIDATION_SHA),(a.historical_report,HISTORICAL_SHA),(a.kd200_report,KD200_SHA),(a.ensemble_report,ENSEMBLE_SHA),(a.test_manifest,TEST_MANIFEST_SHA256))
 for path,sha in expected:
  if sha256_file(path)!=sha: raise ValueError(f"frozen SHA mismatch: {path}")
 override=json.loads(a.override_receipt.read_text())
 if override.get("schema")!=OVERRIDE_SCHEMA or override.get("status")!="frozen_before_test270" or override.get("test_access") is not False: raise ValueError("invalid user override")
 if override.get("checkpoint_sha256")!=CHECKPOINT_SHA or override.get("validation_report_sha256")!=VALIDATION_SHA: raise ValueError("override binding mismatch")
 val=json.loads(a.validation_report.read_text())
 if val.get("candidate_authorized_for_test") is not False or val.get("gates",{}).get("subject_ci_vs_kd200") is not False: raise ValueError("expected failed Subject CI gate")
 rows=[json.loads(x) for x in a.test_manifest.read_text().splitlines() if x.strip()]
 if len(rows)!=TEST_CASES or len({x["subject_id"] for x in rows})!=TEST_SUBJECTS: raise ValueError("test cohort mismatch")
 ids=[x["case_id"] for x in rows]; data=TestBrainCropDataset(a.cache,ids)
 if data.index.get("manifest_sha256")!=TEST_MANIFEST_SHA256 or sha256_file(a.cache/"index.json")!=override["test_cache_index_sha256"]: raise ValueError("test cache mismatch")
 for filename,field in (("images.f32","images_sha256"),("labels.u8","labels_sha256")):
  if sha256_file(a.cache/filename)!=data.index[field]: raise ValueError("test cache tensor mismatch")
 ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
 if ck.get("schema")!=train.CHECKPOINT_SCHEMA or ck.get("arm")!="full972_candidate" or ck.get("epoch")!=168: raise ValueError("checkpoint contract mismatch")
 if ck.get("model_state_sha256")!=train.state_dict_sha256(ck["model_state_dict"]): raise ValueError("model state mismatch")
 for key in ("validation_access","outer_selector_access","confirmation_access","test_access"):
  if ck.get(key) is not False: raise ValueError("checkpoint prior access mismatch")
 historical=json.loads(a.historical_report.read_text())["summary"]
 kd200=json.loads(a.kd200_report.read_text())["kd"]
 ensemble=json.loads(a.ensemble_report.read_text())["ensemble"]
 write_json_atomic(started,{"schema":SCHEMA+".started","created_at_utc":datetime.now(timezone.utc).isoformat(),"override_receipt_sha256":sha256_file(a.override_receipt),"checkpoint_sha256":CHECKPOINT_SHA,"threshold":0.0,"failed_gate_exploratory":True,"not_pristine_blind_test":True,"test_access":True})
 device=torch.device(a.device)
 if device.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("test requires CUDA")
 torch.cuda.set_device(device); arch=ck["architecture"]
 model=AutomaticPromptLocator(center_aware=False,init_filters=int(arch["init_filters"]),dropout=float(arch["dropout"])).to(device); model.load_state_dict(ck["model_state_dict"],strict=True)
 loader=DataLoader(data,batch_size=8,shuffle=False,num_workers=a.num_workers,pin_memory=True,persistent_workers=a.num_workers>0)
 summary=validate(model,loader,device)
 report={"schema":SCHEMA,"schema_version":1,"created_at_utc":datetime.now(timezone.utc).isoformat(),"evidence_scope":"user-authorized failed-gate exploratory reused internal test270; not formal promotion","not_pristine_blind_test":True,"failed_gate_exploratory":True,"checkpoint":{"path":str(a.checkpoint.resolve()),"sha256":CHECKPOINT_SHA,"epoch":168},"data":{"manifest":str(a.test_manifest.resolve()),"manifest_sha256":TEST_MANIFEST_SHA256,"cache_index_sha256":sha256_file(a.cache/"index.json"),"cases":TEST_CASES,"subjects":TEST_SUBJECTS},"candidate":summary,"candidate_bootstrap":intervals(summary),"historical":historical,"kd200":kd200,"ensemble":ensemble,"candidate_minus_historical":compare(historical,summary,seed=20261600),"candidate_minus_kd200":compare(kd200,summary,seed=20261601),"candidate_minus_ensemble":compare(ensemble,summary,seed=20261602),"result_must_not_trigger_tuning":True,"validation_access":True,"outer_selector_access":False,"confirmation_access":False,"test_access":True}
 write_json_atomic(a.output,report)
 print(json.dumps({"candidate":{"scan":summary["locator_dsc_mean"],"subject":summary["locator_subject_dsc_mean"],"point":summary["automatic_point_hit_rate"]},"minus_kd200":report["candidate_minus_kd200"]["ema_minus_raw"],"minus_ensemble":report["candidate_minus_ensemble"]["ema_minus_raw"]},indent=2,sort_keys=True))
if __name__=="__main__": main()
