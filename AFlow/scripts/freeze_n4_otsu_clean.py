"""Bind the Otsu candidate to its preregistration and shared control family."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import n4_otsu_protocol as p
from scripts.freeze_no_n4_clean import check_implementation as check_shared,pairing

FILES=["sammed3d/n4_otsu_input.py","scripts/n4_otsu_protocol.py","scripts/build_n4_otsu_clean_cache.py",
       "scripts/train_n4_otsu_clean.py","scripts/freeze_n4_otsu_clean.py","scripts/evaluate_n4_otsu_clean.py",
       "scripts/run_n4_otsu_clean.py","scripts/audit_n4_otsu_mechanism.py","tests/test_n4_otsu_input.py"]


def check_implementation():
    p.docs();check_shared()
    receipt=json.loads((p.RECORD/"IMPLEMENTATION.json").read_text())
    for row in receipt["files"]:p.checked(ROOT/row["path"],row["sha256"])
    p.checked(p.RECORD/"CACHE_RECEIPT.json",receipt["cache_receipt_sha256"])
    p.checked(p.RECORD/"SYNTHETIC_TEST_RECEIPT.json",receipt["test_receipt_sha256"])
    return receipt


def main(phase):
    p.docs();check_shared()
    if phase=="implementation":
        cache=json.loads((p.RECORD/"CACHE_RECEIPT.json").read_text());assert cache["status"]=="passed"
        tests=json.loads((p.RECORD/"SYNTHETIC_TEST_RECEIPT.json").read_text());assert tests["passed"]
        receipt={"status":"frozen_before_candidate_gpu_smoke_or_training","created_utc":datetime.now(timezone.utc).isoformat(),
            "files":[{"path":f,"sha256":p.sha(ROOT/f)} for f in FILES],"cache_receipt_sha256":p.sha(p.RECORD/"CACHE_RECEIPT.json"),
            "test_receipt_sha256":p.sha(p.RECORD/"SYNTHETIC_TEST_RECEIPT.json"),"shared_implementation_sha256":p.sha(p.SHARED_RECORD/"IMPLEMENTATION.json"),
            "family_note_sha256":p.FAMILY_SHA,"mechanism_sha256":p.MECHANISM_SHA}
        p.write_new(p.RECORD/"IMPLEMENTATION.json",receipt)
    else:
        check_implementation();smoke=phase=="smoke";reports={}
        for fold in ("e","f"):
            cpath=p.PARENT_WORK/("smoke" if smoke else "formal")/f"{fold}_canonical"/"TRAIN_DONE.json"
            apath=p.WORK/("smoke" if smoke else "formal")/f"{fold}_otsu"/"TRAIN_DONE.json"
            control=json.loads(cpath.read_text());candidate=json.loads(apath.read_text())
            if control["smoke"]!=smoke or candidate["smoke"]!=smoke:raise ValueError("smoke/formal confusion")
            pairing(control,candidate,1 if smoke else 168)
            for r in (control,candidate):p.checked(r["checkpoint"]["path"],r["checkpoint"]["sha256"])
            reports[fold]={"control_report":str(cpath),"control_report_sha256":p.sha(cpath),"control_checkpoint":control["checkpoint"],
                           "candidate_report":str(apath),"candidate_report_sha256":p.sha(apath),"candidate_checkpoint":candidate["checkpoint"]}
        if not smoke:
            if (p.SHARED_RECORD/"INNER_STARTED.json").exists():raise RuntimeError("family first readout already started")
            original_done=json.loads((p.SHARED_RECORD/"FORMAL_RUNNER_DONE.json").read_text());assert original_done["status"]=="complete"
        receipt={"status":"passed_smoke_pairing" if smoke else "frozen_family_before_first_ef_readout","created_utc":datetime.now(timezone.utc).isoformat(),
                 "all_pairing_gates_passed":True,"reports":reports,"first_ef_readout_occurred":False,
                 "implementation_sha256":p.sha(p.RECORD/"IMPLEMENTATION.json"),"family_note_sha256":p.FAMILY_SHA}
        p.write_new(p.RECORD/("GPU_SMOKE_RECEIPT.json" if smoke else "INNER_PREACCESS.json"),receipt)
    print(json.dumps({"phase":phase,"status":receipt["status"]},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("phase",choices=("implementation","smoke","inner"));main(parser.parse_args().phase)
