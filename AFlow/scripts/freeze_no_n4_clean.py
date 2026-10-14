"""Freeze implementation, GPU smoke pairing, and completed models before readout."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import no_n4_clean_protocol as p

FILES=["scripts/no_n4_clean_protocol.py","scripts/build_no_n4_clean_splits.py","scripts/build_no_n4_clean_cache.py",
       "scripts/train_no_n4_clean.py","scripts/no_n4_clean_analysis.py","scripts/evaluate_no_n4_clean.py","scripts/freeze_no_n4_clean.py","scripts/run_no_n4_clean.py",
       "tests/test_no_n4_clean_contract.py","scripts/train_t1ce_wt_gcsam_inner.py","scripts/train_t1ce_wt_capg3d.py",
       "scripts/train_t1ce_wt_plain_voxel_kd_continuation.py","scripts/build_t1ce_wt_native_volume_artifact.py",
       "scripts/train_t1ce_wt_voxel_kd_972_draw_budget.py","sammed3d/t1ce_wt_capg3d.py","sammed3d/t1ce_binary.py"]


def pairing(a,b,epochs):
    if not a["all_gates_passed"] or not b["all_gates_passed"]:raise ValueError("training gates failed")
    for key in ("initial_state_sha256","sampling_weight_sha256","fit_sha256"):
        if a[key]!=b[key]:raise ValueError("pair mismatch: "+key)
    if len(a["history"])!=epochs or len(b["history"])!=epochs:raise ValueError("history count mismatch")
    for x,y in zip(a["history"],b["history"]):
        for key in ("epoch","steps","draws","sampler_sha256","augmentation_rng_before_sha256","augmentation_rng_after_sha256"):
            if x[key]!=y[key]:raise ValueError("pair history mismatch: "+key)
    if a["history"][0]["first_batch"]["label"]!=b["history"][0]["first_batch"]["label"]:raise ValueError("pair first label mismatch")
    if a["teacher_checkpoints"]!=b["teacher_checkpoints"]:raise ValueError("pair teacher mismatch")
    return True


def check_implementation():
    data=json.loads((p.RECORD/"IMPLEMENTATION.json").read_text())
    for item in data["files"]:p.checked(ROOT/item["path"],item["sha256"])
    p.checked(p.RECORD/"CACHE_RECEIPT.json",data["cache_receipt_sha256"])
    p.checked(p.RECORD/"SYNTHETIC_TEST_RECEIPT.json",data["synthetic_receipt_sha256"])
    return data


def main(phase):
    p.docs();p.checked(p.RECORD/"SPLIT.json",p.SPLIT_SHA)
    if phase=="implementation":
        cache=json.loads((p.RECORD/"CACHE_RECEIPT.json").read_text());assert cache["status"]=="passed"
        tests=json.loads((p.RECORD/"SYNTHETIC_TEST_RECEIPT.json").read_text());assert tests["passed"]
        data={"status":"frozen_before_gpu_smoke_and_formal","created_utc":datetime.now(timezone.utc).isoformat(),
              "files":[{"path":path,"sha256":p.sha(ROOT/path)} for path in FILES],"cache_receipt_sha256":p.sha(p.RECORD/"CACHE_RECEIPT.json"),
              "synthetic_receipt_sha256":p.sha(p.RECORD/"SYNTHETIC_TEST_RECEIPT.json"),"idea_sha256":p.IDEA_SHA,"prereg_sha256":p.PREREG_SHA,
              "amendment_sha256":p.AMENDMENT_SHA,"split_sha256":p.SPLIT_SHA}
        p.write_new(p.RECORD/"IMPLEMENTATION.json",data)
    else:
        check_implementation();smoke=phase=="smoke";reports={};split=json.loads((p.RECORD/"SPLIT.json").read_text())
        for fold in ("e","f"):
            reports[fold]={}
            for role in ("history","ema","canonical","no_n4"):
                path=p.WORK/("smoke" if smoke else "formal")/f"{fold}_{role}"/"TRAIN_DONE.json"
                report=json.loads(path.read_text())
                if not report["all_gates_passed"] or report["smoke"]!=smoke or report["fit_sha256"]!=split["folds"][fold]["fit"]["sha256"]:raise ValueError("training report ownership failed")
                p.checked(report["checkpoint"]["path"],report["checkpoint"]["sha256"])
                reports[fold][role]=report
            pairing(reports[fold]["canonical"],reports[fold]["no_n4"],1 if smoke else 168)
        data={"status":"passed_smoke_pairing" if smoke else "frozen_before_first_inner_model_access","created_utc":datetime.now(timezone.utc).isoformat(),
              "implementation_sha256":p.sha(p.RECORD/"IMPLEMENTATION.json"),"split_sha256":p.SPLIT_SHA,
              "reports":{fold:{role:{"report_path":str(p.WORK/("smoke" if smoke else "formal")/f"{fold}_{role}"/"TRAIN_DONE.json"),
                                     "report_sha256":p.sha(p.WORK/("smoke" if smoke else "formal")/f"{fold}_{role}"/"TRAIN_DONE.json"),"checkpoint":report["checkpoint"]}
                                for role,report in items.items()} for fold,items in reports.items()},"all_pairing_gates_passed":True,"inner_metrics_accessed":False}
        if not smoke:data["smoke_receipt_sha256"]=p.sha(p.RECORD/"GPU_SMOKE_RECEIPT.json")
        p.write_new(p.RECORD/("GPU_SMOKE_RECEIPT.json" if smoke else "INNER_PREACCESS.json"),data)
    print(json.dumps({"phase":phase,"status":data["status"]},indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("phase",choices=("implementation","smoke","inner"));main(ap.parse_args().phase)
