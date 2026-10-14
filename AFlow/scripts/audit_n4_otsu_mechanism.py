"""Frozen train-only mechanism gate for Otsu N4 mask; no label/model access."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import nibabel as nib
import numpy as np
import torch
import SimpleITK as sitk
from sammed3d.n4_otsu_input import n4_otsu_correct
from sammed3d.background_zero_decode import decode_background_zero
from scripts.audit_n4_scale_diagnostics_20260905 import normalize

RECORD=ROOT/"archive/research_records/2026-09-05_n4_otsu_clean_replication"
IDEA_SHA="8ada2e9b5effcd893131caa734510656eda552251c1a0f64a86046904815ae53"
PREREG_SHA="cac4ff343f933707be4535fb3b7925a940394d4710f361187468a3c6466b3907"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert sha(RECORD/"IDEA.md")==IDEA_SHA and sha(RECORD/"PREREG.md")==PREREG_SHA
    family=ROOT/"archive/research_records/2026-09-05_no_n4_clean_teacher_replication"
    assert not (family/"INNER_STARTED.json").exists()
    assert not list((ROOT/"workspace/BraTS/sammed3d_t1ce_wt/no_n4_clean_20260905/formal").glob("*_canonical/STARTED.json"))
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1);torch.set_num_threads(1)
    source=ROOT/"data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl"
    assert sha(source)=="6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65"
    rows=sorted([json.loads(x) for x in source.read_text().splitlines()],key=lambda r:r["case_id"])[:3]
    report={"status":"completed_mechanism_gate","created_utc":datetime.now(timezone.utc).isoformat(),"rows":[],"label_access":False,"model_inference":False,
            "first_ef_readout_before_registration":False,"candidate_students_started_before_registration":False,"idea_sha256":IDEA_SHA,"prereg_sha256":PREREG_SHA}
    start=time.monotonic()
    for row in rows:
        assert row["split"]=="train"
        nii=nib.as_closest_canonical(nib.load(row["image_path"]))
        original=np.ascontiguousarray(np.asanyarray(nii.dataobj),dtype=np.float32)
        values=original[original>0].astype(np.float64);sigma=values.std();offset=-values.mean()/sigma
        encoded=(original.astype(np.float64)/sigma+offset).astype(np.float32)
        decoded,_=decode_background_zero(encoded)
        a,mask_a=n4_otsu_correct(original);b,mask_b=n4_otsu_correct(decoded)
        base,geo_a=normalize(row["case_id"],a,nib.affines.voxel_sizes(nii.affine))
        candidate,geo_b=normalize(row["case_id"],b,nib.affines.voxel_sizes(nii.affine))
        error=np.abs(base-candidate)
        item={"case_id":row["case_id"],"support_exact":bool(np.array_equal(original>0,decoded>0)),"bbox_exact":geo_a==geo_b,
              "original_mask":mask_a,"encoded_mask":mask_b,"fit_mask_exact":mask_a["fit_mask_sha256"]==mask_b["fit_mask_sha256"],
              "normalized_max_error":float(error.max()),"normalized_mean_error":float(error.mean())}
        item["passed"]=item["support_exact"] and item["bbox_exact"] and item["normalized_max_error"]<=.001 and item["normalized_mean_error"]<=.00001
        report["rows"].append(item);print(json.dumps(item),flush=True)
    report["all_gates_passed"]=all(r["passed"] for r in report["rows"])
    report["elapsed_seconds"]=time.monotonic()-start
    report["runner_sha256"]=sha(Path(__file__));report["implementation_sha256"]=sha(ROOT/"sammed3d/n4_otsu_input.py")
    with (RECORD/"MECHANISM_GATE.json").open("x") as f:json.dump(report,f,indent=2,sort_keys=True);f.write("\n")
    print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2))


if __name__=="__main__":main()
