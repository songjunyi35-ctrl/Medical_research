"""Build train972 Otsu-fit N4 cache without changing the historical builder."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import nibabel as nib
import numpy as np
import torch
import SimpleITK as sitk
from sammed3d.n4_otsu_input import n4_otsu_correct
from sammed3d.t1ce_wt_training import _load_canonical_nifti,validate_row
from scripts.build_no_n4_clean_cache import arrays,row_arrays
from scripts import no_n4_clean_protocol as shared

RECORD=ROOT/"archive/research_records/2026-09-05_n4_otsu_clean_replication"
IDEA_SHA="8ada2e9b5effcd893131caa734510656eda552251c1a0f64a86046904815ae53"
PREREG_SHA="cac4ff343f933707be4535fb3b7925a940394d4710f361187468a3c6466b3907"
CFG={}


def prepare(row):
    validate_row(row)
    if row["split"]!="train":raise ValueError("train only")
    image=_load_canonical_nifti(Path(row["image_path"]),"train MRI")
    label=_load_canonical_nifti(Path(row["label_path"]),"train label")
    if image.shape!=label.shape or not np.allclose(image.affine,label.affine,rtol=0,atol=1e-5):raise ValueError("geometry mismatch")
    corrected,mask=n4_otsu_correct(np.asanyarray(image.dataobj))
    result=arrays(row["case_id"],corrected,np.asanyarray(label.dataobj),nib.affines.voxel_sizes(image.affine))
    return *result,mask


def init(cfg):
    torch.set_num_threads(1);sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1);CFG.update(cfg)
    CFG["image_fd"]=os.open(cfg["images"],os.O_WRONLY);CFG["label_fd"]=os.open(cfg["labels"],os.O_WRONLY)


def worker(task):
    slot,row=task;image,label,geo,mask=prepare(row);voxels=128**3
    if os.pwrite(CFG["image_fd"],image.tobytes(),slot*voxels*4)!=voxels*4:raise IOError("image write incomplete")
    if os.pwrite(CFG["label_fd"],label.tobytes(),slot*voxels)!=voxels:raise IOError("label write incomplete")
    return slot,row["case_id"],geo,mask,int(label.sum())


def main(args):
    shared.docs();shared.checked(RECORD/"IDEA.md",IDEA_SHA);shared.checked(RECORD/"PREREG.md",PREREG_SHA)
    mechanism=json.loads((RECORD/"MECHANISM_GATE.json").read_text());assert mechanism["all_gates_passed"]
    shared.checked(shared.SOURCE,shared.SOURCE_SHA);shared.checked(shared.CONTROL_CACHE/"index.json",shared.INDEX_SHA)
    control=json.loads((shared.CONTROL_CACHE/"index.json").read_text());rows=shared.read_rows(shared.SOURCE)
    assert len(rows)==972 and [r["case_id"] for r in rows]==control["case_ids"]
    out=args.output.resolve();required=972*128**3*5
    if out.exists() or not str(out).startswith("/tmp/"):raise ValueError("new /tmp directory required")
    if shutil.disk_usage(out.parent).free<required+2*1024**3:raise RuntimeError("disk reserve failed")
    out.mkdir()
    for name,size in (("images.f32",required*4//5),("labels.u8",required//5)):
        with (out/name).open("xb") as f:f.truncate(size)
    start=time.monotonic();results=[]
    with mp.get_context("spawn").Pool(args.workers,initializer=init,initargs=({"images":str(out/"images.f32"),"labels":str(out/"labels.u8")},)) as pool:
        for result in pool.imap_unordered(worker,enumerate(rows),chunksize=1):
            results.append(result)
            if len(results)%50==0:print(f"Otsu cache {len(results)}/972",flush=True)
    results.sort();geometry={case:geo for _,case,geo,_,_ in results}
    assert geometry==control["crop_geometry"]
    label_sha=shared.sha(out/"labels.u8");assert label_sha==shared.LABELS_SHA
    index={**control,"manifest":str(shared.SOURCE),"created_utc":datetime.now(timezone.utc).isoformat(),
        "preprocessing":{**control["preprocessing"],"n4_fit_mask":"OtsuThreshold(full_image,0,1,200)",
                          "n4_implementation":"SimpleITK2.5.5/ITK5.4.6,4x50,shrink2;Otsu200 fitting mask; full positive support retained"},
        "images_sha256":shared.sha(out/"images.f32"),"labels_sha256":label_sha,"crop_geometry":geometry,
        "fit_mask_stats":{case:mask for _,case,_,mask,_ in results},"builder":str(Path(__file__)),"builder_sha256":shared.sha(Path(__file__)),
        "build_seconds":time.monotonic()-start,"idea_sha256":IDEA_SHA,"prereg_sha256":PREREG_SHA}
    shared.write_new(out/"index.json",index)
    torch.set_num_threads(1);sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    image_cache=np.memmap(out/"images.f32",mode="r",dtype=np.float32,shape=(972,128,128,128))
    label_cache=np.memmap(out/"labels.u8",mode="r",dtype=np.uint8,shape=(972,128,128,128))
    checks=[]
    for slot in range(3):
        image,label,geo,mask=prepare(rows[slot])
        checks.append({"case_id":rows[slot]["case_id"],"image_exact":bool(np.array_equal(image,image_cache[slot])),
            "label_exact":bool(np.array_equal(label,label_cache[slot])),"geometry_exact":geo==geometry[rows[slot]["case_id"]]})
    passed=all(all(v for k,v in r.items() if k!="case_id") for r in checks)
    receipt={"status":"passed" if passed else "failed","candidate_cache":str(out),"candidate_index_sha256":shared.sha(out/"index.json"),
        "images_sha256":index["images_sha256"],"labels_sha256":label_sha,"all972_geometry_exact":True,"all972_labels_exact":True,
        "independent_rebuilds":checks,"elapsed_seconds":time.monotonic()-start,"mechanism_receipt_sha256":shared.sha(RECORD/"MECHANISM_GATE.json"),
        "new_evaluation_metrics":False}
    shared.write_new(RECORD/"CACHE_RECEIPT.json",receipt);print(json.dumps(receipt,indent=2))
    if not passed:raise RuntimeError("independent reproduction failed")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--workers",type=int,default=16)
    main(parser.parse_args())
