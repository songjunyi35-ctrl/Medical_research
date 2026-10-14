"""Build only the frozen Full972 train no-N4 cache and verify paired geometry."""
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
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import nibabel as nib
import numpy as np
import torch
from sammed3d.t1ce_binary import prepare_t1ce_supervised_arrays
from sammed3d.t1ce_wt_training import _load_canonical_nifti,validate_row
from sammed3d.t1ce_wt_cached import CACHE_SCHEMA,MODEL_SHAPE
from scripts.build_t1ce_wt_cache import n4_correct
from scripts import no_n4_clean_protocol as p
CFG={}


def arrays(case_id,image,label,spacing,use_n4=False):
    if image.ndim!=3 or image.shape!=label.shape or not np.isfinite(image).all():
        raise ValueError("invalid image/label input geometry or values")
    source=np.ascontiguousarray(image,dtype=np.float32)
    source=n4_correct(source,2) if use_n4 else np.where(source>0,source,0).astype(np.float32)
    indices=np.nonzero(source>0)
    if any(x.size==0 for x in indices):raise ValueError("empty positive brain support")
    origin=[int(x.min()) for x in indices];stop=[int(x.max())+1 for x in indices]
    crop=tuple(slice(a,b) for a,b in zip(origin,stop))
    view=prepare_t1ce_supervised_arrays(case_id,source[crop],np.ascontiguousarray(label[crop]),spacing,output_shape=MODEL_SHAPE)
    out_image=np.ascontiguousarray(view.image.numpy()[0],dtype=np.float32)
    out_label=np.ascontiguousarray(view.target.numpy()[0],dtype=np.uint8)
    if not np.isfinite(out_image).all():raise ValueError("nonfinite cache output")
    return out_image,out_label,{"origin":origin,"extent":[b-a for a,b in zip(origin,stop)],"native_shape":list(image.shape)}


def row_arrays(row,use_n4=False):
    validate_row(row)
    if row["split"]!="train":raise ValueError("non-train row refused")
    image=_load_canonical_nifti(Path(row["image_path"]),"train T1ce")
    label=_load_canonical_nifti(Path(row["label_path"]),"train label")
    if image.shape!=label.shape or not np.allclose(image.affine,label.affine,rtol=0,atol=1e-5):raise ValueError("raw geometry mismatch")
    return arrays(row["case_id"],np.asanyarray(image.dataobj),np.asanyarray(label.dataobj),nib.affines.voxel_sizes(image.affine),use_n4)


def init(config):
    torch.set_num_threads(1)
    import SimpleITK as sitk
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    CFG.update(config)
    CFG["image_fd"]=os.open(config["images"],os.O_WRONLY)
    CFG["label_fd"]=os.open(config["labels"],os.O_WRONLY)


def worker(task):
    slot,row=task;image,label,geometry=row_arrays(row)
    voxels=int(np.prod(MODEL_SHAPE))
    a=os.pwrite(CFG["image_fd"],image.tobytes(),slot*voxels*4)
    b=os.pwrite(CFG["label_fd"],label.tobytes(),slot*voxels)
    if a!=voxels*4 or b!=voxels:raise IOError("short cache write")
    return slot,row["case_id"],geometry,int(label.sum())


def main(args):
    p.docs();p.checked(p.SOURCE,p.SOURCE_SHA);p.checked(p.CONTROL_CACHE/"index.json",p.INDEX_SHA)
    p.checked(p.CONTROL_CACHE/"labels.u8",p.LABELS_SHA)
    output=args.output.resolve()
    if output.exists() or "/tmp/" not in str(output):raise ValueError("new /tmp output required")
    required=972*int(np.prod(MODEL_SHAPE))*5
    if shutil.disk_usage(output.parent).free<required+2*1024**3:raise RuntimeError("insufficient cache disk reserve")
    rows=p.read_rows(p.SOURCE);control=json.loads((p.CONTROL_CACHE/"index.json").read_text())
    assert len(rows)==972 and [r["case_id"] for r in rows]==control["case_ids"]
    output.mkdir()
    for name,bytes_needed in (("images.f32",required*4//5),("labels.u8",required//5)):
        with (output/name).open("xb") as f:f.truncate(bytes_needed)
    start=time.monotonic();results=[]
    config={"images":str(output/"images.f32"),"labels":str(output/"labels.u8")}
    with mp.get_context("spawn").Pool(args.workers,initializer=init,initargs=(config,)) as pool:
        for item in pool.imap_unordered(worker,enumerate(rows),chunksize=1):
            results.append(item)
            if len(results)%50==0:print(f"built {len(results)}/972",flush=True)
    results.sort();geometry={case:geo for _,case,geo,_ in results}
    if geometry!=control["crop_geometry"]:raise RuntimeError("no-N4 crop geometry differs from canonical")
    labels_sha=p.sha(output/"labels.u8")
    if labels_sha!=p.LABELS_SHA:raise RuntimeError("labels differ from canonical")
    index={"schema":CACHE_SCHEMA,"schema_version":1,"manifest":str(p.SOURCE),"manifest_sha256":p.SOURCE_SHA,
        "count":972,"case_ids":control["case_ids"],"model_shape":list(MODEL_SHAPE),"image_dtype":"float32","label_dtype":"uint8",
        "split":"train","test_access":False,"preprocessing":{**control["preprocessing"],"n4_bias_field_correction":False,
            "n4_shrink_factor":None,"n4_implementation":None,"nonpositive_input_values":"zero, same support rule as historical N4"},
        "images_sha256":p.sha(output/"images.f32"),"labels_sha256":labels_sha,"crop_geometry":geometry,
        "per_case_positive_voxels":{case:count for _,case,_,count in results},"builder_sha256":p.sha(Path(__file__)),
        "idea_sha256":p.IDEA_SHA,"prereg_sha256":p.PREREG_SHA,"build_seconds":time.monotonic()-start}
    p.write_new(output/"index.json",index)
    torch.set_num_threads(1)
    import SimpleITK as sitk
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    candidate_images=np.memmap(output/"images.f32",mode="r",dtype=np.float32,shape=(972,*MODEL_SHAPE))
    canonical_images=np.memmap(p.CONTROL_CACHE/"images.f32",mode="r",dtype=np.float32,shape=(972,*MODEL_SHAPE))
    labels=np.memmap(output/"labels.u8",mode="r",dtype=np.uint8,shape=(972,*MODEL_SHAPE))
    checks=[]
    for slot in range(3):
        fresh_image,fresh_label,fresh_geo=row_arrays(rows[slot])
        control_image,control_label,control_geo=row_arrays(rows[slot],True)
        checks.append({"case_id":rows[slot]["case_id"],"candidate_image_exact":bool(np.array_equal(fresh_image,candidate_images[slot])),
            "canonical_image_exact":bool(np.array_equal(control_image,canonical_images[slot])),
            "label_exact":bool(np.array_equal(fresh_label,labels[slot]) and np.array_equal(control_label,labels[slot])),
            "geometry_exact":fresh_geo==control_geo==geometry[rows[slot]["case_id"]]})
    passed=all(all(value for key,value in row.items() if key!="case_id") for row in checks)
    receipt={"status":"passed" if passed else "failed","candidate_cache":str(output),"candidate_index_sha256":p.sha(output/"index.json"),
        "canonical_cache":str(p.CONTROL_CACHE),"canonical_index_sha256":p.INDEX_SHA,"candidate_images_sha256":index["images_sha256"],
        "canonical_images_sha256":p.sha(p.CONTROL_CACHE/"images.f32"),"labels_sha256":labels_sha,
        "all972_labels_exact":True,"all972_geometry_exact":True,"independent_rebuilds":checks,"image_bytes":required*4//5,"label_bytes":required//5,
        "elapsed_seconds":time.monotonic()-start,"new_evaluation_metrics":False}
    if receipt["canonical_images_sha256"]!=control["images_sha256"]:raise RuntimeError("canonical images file hash changed")
    p.write_new(p.RECORD/"CACHE_RECEIPT.json",receipt)
    print(json.dumps(receipt,indent=2))
    if not passed:raise RuntimeError("cache independent reconstruction gate failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--workers",type=int,default=16)
    main(ap.parse_args())
