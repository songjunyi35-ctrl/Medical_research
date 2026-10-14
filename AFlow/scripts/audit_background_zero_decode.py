"""Frozen input-only equivalence checks; no label or model access."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import nibabel as nib
import numpy as np
import torch
from sammed3d.background_zero_decode import decode_background_zero
from sammed3d.t1ce_binary import prepare_t1ce_inference_array
from scripts.build_t1ce_wt_cache import n4_correct

RECORD=ROOT/"archive/research_records/2026-09-05_background_zero_decode"
IDEA_SHA="06c9688926e20f366486d58e54101623ce7c83ca273c5853a9dceb4a16cecfcc"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()


def bbox(a):
    indices=np.nonzero(a>0)
    return [int(i.min()) for i in indices],[int(i.max())+1 for i in indices]


def pipeline(case,a,spacing):
    corrected=n4_correct(a,2)
    lo,hi=bbox(corrected)
    crop=corrected[tuple(slice(x,y) for x,y in zip(lo,hi))]
    return prepare_t1ce_inference_array(case,crop,spacing,output_shape=(128,128,128)).image.numpy(),(lo,hi)


def normalized(a):
    out=np.zeros(a.shape,dtype=np.float64)
    x=a[a>0].astype(np.float64)
    out[a>0]=(x-x.mean())/x.std()
    return out


def main(action):
    assert sha(RECORD/"IDEA.md")==IDEA_SHA
    torch.set_num_threads(1)
    start=time.monotonic()
    result={"created_utc":datetime.now(timezone.utc).isoformat(),"idea_sha256":IDEA_SHA,
            "action":action,"rows":[],"label_access":False,"model_inference":False}
    if action=="train":
        source=ROOT/"data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl"
        assert sha(source)=="6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65"
        rows=sorted([json.loads(x) for x in source.read_text().splitlines()],key=lambda x:x["case_id"])[:3]
        for row in rows:
            assert row["split"]=="train"
            nii=nib.as_closest_canonical(nib.load(row["image_path"]))
            a=np.ascontiguousarray(np.asanyarray(nii.dataobj),dtype=np.float32)
            assert np.isfinite(a).all() and a.min()==0
            spacing=nib.affines.voxel_sizes(nii.affine)
            base,geometry=pipeline(row["case_id"],a,spacing)
            values=a[a>0].astype(np.float64)
            for name,scale,offset in (("source_positive_zscore",float(values.std()),-float(values.mean()/values.std())),("fixed_affine",7.,-3.)):
                encoded=(a.astype(np.float64)/scale+offset).astype(np.float32)
                decoded,meta=decode_background_zero(encoded)
                norm_error=float(np.max(np.abs(normalized(a)-normalized(decoded))))
                processed,decoded_geometry=pipeline(row["case_id"],decoded,spacing)
                diff=np.abs(base-processed)
                item={"case_id":row["case_id"],"encoding":name,"raw_image_sha256":sha(row["image_path"]),
                      "background":meta,"support_exact":bool(np.array_equal(a>0,decoded>0)),
                      "bbox_exact":bbox(a)==bbox(decoded) and geometry==decoded_geometry,
                      "source_zscore_max_error":norm_error,"pipeline_max_error":float(diff.max()),"pipeline_mean_error":float(diff.mean())}
                item["passed"]=item["support_exact"] and item["bbox_exact"] and norm_error<=1e-4 and item["pipeline_max_error"]<=1e-3 and item["pipeline_mean_error"]<=1e-5
                result["rows"].append(item)
                print(json.dumps(item),flush=True)
                if time.monotonic()-start>1200:raise RuntimeError("20min engineering time gate exceeded")
    else:
        train=json.loads((RECORD/"TRAIN_EQUIVALENCE.json").read_text())
        assert train["all_gates_passed"]
        download=ROOT/"archive/reviews/2026-09-05_30h_chain_audit/RHUH_IMAGE_DOWNLOAD.json"
        receipt=json.loads(download.read_text())
        assert receipt["all_official_match"]
        for row in receipt["files"]:
            assert sha(row["path"])==row["sha256"]
            nii=nib.as_closest_canonical(nib.load(row["path"]))
            try:
                decoded,meta=decode_background_zero(np.asanyarray(nii.dataobj))
                item={"case_id":row["case_id"],**meta,"passed":True,"bbox":bbox(decoded)}
            except ValueError as error:
                item={"case_id":row["case_id"],"passed":False,"error":str(error)}
            result["rows"].append(item)
    result["all_gates_passed"]=all(row["passed"] for row in result["rows"])
    result["elapsed_seconds"]=time.monotonic()-start
    result["runner_sha256"]=sha(Path(__file__))
    result["decoder_sha256"]=sha(ROOT/"sammed3d/background_zero_decode.py")
    path=RECORD/("TRAIN_EQUIVALENCE.json" if action=="train" else "RHUH_DECODE_FORMAT_QC.json")
    with path.open("x") as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=("train","rhuh"))
    main(p.parse_args().action)
