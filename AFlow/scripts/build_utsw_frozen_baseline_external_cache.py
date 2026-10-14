"""Build the single canonical-N4 UTSW362 cache for frozen baselines."""
import argparse
from datetime import datetime,timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch

from sammed3d import t1ce_binary
from scripts.audit_utsw_primary_fets_no_n4_input_bridge import validate_source_arrays
from scripts.build_no_n4_clean_cache import arrays as shared_arrays
from scripts import utsw_frozen_baseline_external_protocol as p

CFG={};ORIGINAL=t1ce_binary._case_id


def init(cfg):
    CFG.update(cfg);CFG["allowed"]=set(cfg["allowed"]);torch.set_num_threads(1);sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    def external(value):
        if isinstance(value,str) and value in CFG["allowed"]:return value
        return ORIGINAL(value)
    t1ce_binary._case_id=external;CFG["image_fd"]=os.open(cfg["images"],os.O_WRONLY);CFG["label_fd"]=os.open(cfg["labels"],os.O_WRONLY)


def array_sha(a):return __import__("hashlib").sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def worker(task):
    slot,row,image_sha,label_sha=task;sid=row["subject_id"];root=p.SOURCE/"copy_a"/sid
    image_path=root/"brain_t1ce.nii.gz";label_path=root/"rtumorseg_manual_correction.nii.gz"
    if p.sha(image_path)!=image_sha or p.sha(label_path)!=label_sha:raise ValueError(sid+" raw SHA changed")
    image_nii=nib.as_closest_canonical(nib.load(str(image_path),mmap=True));label_nii=nib.as_closest_canonical(nib.load(str(label_path),mmap=True))
    affine=float(np.max(np.abs(image_nii.affine-label_nii.affine)))
    if image_nii.shape!=label_nii.shape or affine>1e-6:raise ValueError(sid+" geometry")
    image=np.ascontiguousarray(np.asanyarray(image_nii.dataobj));label=np.ascontiguousarray(np.asanyarray(label_nii.dataobj));binary,source=validate_source_arrays(image,label)
    if not set(source["label_values"])<={0.,1.,2.,4.}:raise ValueError(sid+" label domain")
    out_image,out_label,geometry=shared_arrays(sid,image,binary,nib.affines.voxel_sizes(image_nii.affine),use_n4=True)
    if not np.isfinite(out_image).all() or not out_label.any():raise ValueError(sid+" invalid output")
    vox=128**3
    if os.pwrite(CFG["image_fd"],out_image.tobytes(),slot*vox*4)!=vox*4:raise IOError("short image")
    if os.pwrite(CFG["label_fd"],out_label.tobytes(),slot*vox)!=vox:raise IOError("short label")
    return slot,sid,{"native_shape":list(image_nii.shape),"native_affine_max_abs":affine,"axis_codes":list(nib.aff2axcodes(image_nii.affine)),
        "spacing":[float(x) for x in nib.affines.voxel_sizes(image_nii.affine)],"raw_label_values":source["label_values"],
        "raw_wt_voxels":source["source_wt_voxels"],"geometry":geometry,"output_image_sha256":array_sha(out_image),"output_label_sha256":array_sha(out_label)}


def main(workers):
    from scripts.freeze_utsw_frozen_baseline_external import check_implementation
    check_implementation();p.verify_preaccess();download=json.loads(p.SOURCE_DOWNLOAD.read_text())
    if download.get("status")!="passed" or download.get("files_copy_exact")!=724 or download.get("voxel_arrays_accessed") is not False:raise ValueError("source receipt")
    if p.CACHE.exists():raise FileExistsError(p.CACHE)
    needed=362*128**3*5
    if shutil.disk_usage(p.CACHE.parent).free<needed+5*1024**3:raise RuntimeError("cache disk reserve")
    rows=p.cohort();source={x["relative_path"]:x for x in download["comparisons"]};tasks=[]
    for slot,row in enumerate(rows):
        sid=row["subject_id"];tasks.append((slot,row,source[f"{sid}/brain_t1ce.nii.gz"]["sha256"],source[f"{sid}/rtumorseg_manual_correction.nii.gz"]["sha256"]))
    p.CACHE.mkdir();
    for name,size in (("images.f32",362*128**3*4),("labels.u8",362*128**3)):
        with (p.CACHE/name).open("xb") as f:f.truncate(size)
    start=time.monotonic();cfg={"allowed":[x["subject_id"] for x in rows],"images":str(p.CACHE/"images.f32"),"labels":str(p.CACHE/"labels.u8")};values=[]
    with mp.get_context("spawn").Pool(workers,initializer=init,initargs=(cfg,)) as pool:
        for x in pool.imap_unordered(worker,tasks,chunksize=1):
            values.append(x)
            if len(values)%25==0:print(f"built UTSW baseline {len(values)}/362",flush=True)
    values.sort();details={sid:v for _,sid,v in values};ids=[x["subject_id"] for x in rows]
    bridge=json.loads(p.source_contract.CANONICAL_BRIDGE_COPY_A.read_text());fixed={}
    for sid in ("BT0001","BT0002","BT0003"):
        fixed[sid]={"image_exact":details[sid]["output_image_sha256"]==bridge["cases"][sid]["output_image_sha256"],
                    "label_exact":details[sid]["output_label_sha256"]==bridge["cases"][sid]["output_label_sha256"],
                    "geometry_exact":details[sid]["geometry"]==bridge["cases"][sid]["geometry"]}
    index={"schema":"aflow.utsw_frozen_baseline_external.cache","created_utc":datetime.now(timezone.utc).isoformat(),
        "split":"external_cross_stage","cases":362,"subjects":362,"case_ids":ids,"subject_ids":ids,"model_shape":[128]*3,
        "image_dtype":"float32","label_dtype":"uint8","images_sha256":p.sha(p.CACHE/"images.f32"),"labels_sha256":p.sha(p.CACHE/"labels.u8"),
        "geometry":{sid:details[sid]["geometry"] for sid in ids},"preprocessing":{"source":"official primary FeTS brain_t1ce",
        "canonical_ras":True,"n4_bias_field_correction":True,"n4_fit_mask":"positive source voxels","spatial":"positive-support crop then resize128",
        "normalization":"source positive-support population z-score","uses_gt_for_localization":False},"target":"UTSW WT=label>0",
        "source_download_sha256":p.sha(p.SOURCE_DOWNLOAD),"preaccess_sha256":p.sha(p.PREACCESS),"builder_sha256":p.sha(Path(__file__)),
        "elapsed_seconds":time.monotonic()-start}
    p.write_new(p.CACHE/"index.json",index);images=np.memmap(p.CACHE/"images.f32",mode="r",dtype=np.float32,shape=(362,128,128,128));labels=np.memmap(p.CACHE/"labels.u8",mode="r",dtype=np.uint8,shape=(362,128,128,128))
    valid=True
    for i in range(362):valid=valid and bool(np.isfinite(images[i]).all()) and bool(np.all((labels[i]==0)|(labels[i]==1))) and bool(labels[i].any())
    gates={"all_memmaps_valid":valid,"fixed_three_exact":all(all(x.values()) for x in fixed.values()),
        "image_bytes_exact":(p.CACHE/"images.f32").stat().st_size==362*128**3*4,"label_bytes_exact":(p.CACHE/"labels.u8").stat().st_size==362*128**3}
    receipt={"schema":"aflow.utsw_frozen_baseline_external.cache_receipt","created_utc":datetime.now(timezone.utc).isoformat(),
        "status":"passed" if all(gates.values()) else "failed","gates":gates,"all_gates_passed":all(gates.values()),"cache":str(p.CACHE),
        "index_sha256":p.sha(p.CACHE/"index.json"),"images_sha256":index["images_sha256"],"labels_sha256":index["labels_sha256"],
        "fixed_three":fixed,"elapsed_seconds":time.monotonic()-start,"label_voxels_accessed":True,"model_inference":False}
    p.write_new(p.RECORD/"CACHE_RECEIPT.json",receipt);print(json.dumps(receipt,indent=2,sort_keys=True))
    if not receipt["all_gates_passed"]:raise RuntimeError("cache gates failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--workers",type=int,default=8);main(ap.parse_args().workers)
