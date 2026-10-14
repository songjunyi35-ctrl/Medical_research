"""Read train MRI only to diagnose deterministic N4 sensitivity to exact scaling."""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from sammed3d.t1ce_binary import prepare_t1ce_inference_array

OUT=ROOT/"archive/reviews/2026-09-05_n4_scale_diagnostics"


def corrected(data):
    image=sitk.GetImageFromArray(np.ascontiguousarray(data,dtype=np.float32))
    mask=sitk.GetImageFromArray((data>0).astype(np.uint8));mask.CopyInformation(image)
    n4=sitk.N4BiasFieldCorrectionImageFilter();n4.SetMaximumNumberOfIterations([50]*4)
    trace=[]
    def observe():
        trace.append({"level":int(n4.GetCurrentLevel()),"iteration":int(n4.GetElapsedIterations()),"convergence":float(n4.GetCurrentConvergenceMeasurement())})
    n4.AddCommand(sitk.sitkIterationEvent,observe)
    n4.Execute(sitk.Cast(sitk.Shrink(image,[2]*3),sitk.sitkFloat32),sitk.Shrink(mask,[2]*3))
    log_bias=n4.GetLogBiasFieldAsImage(image)
    output=np.asarray(sitk.GetArrayFromImage(sitk.Cast(image,sitk.sitkFloat32)/sitk.Exp(log_bias)),dtype=np.float32)
    output[data<=0]=0
    n4.RemoveAllCommands()
    return output,trace


def normalize(case,data,spacing):
    xyz=np.nonzero(data>0);lo=[int(v.min()) for v in xyz];hi=[int(v.max())+1 for v in xyz]
    crop=data[tuple(slice(a,b) for a,b in zip(lo,hi))]
    return prepare_t1ce_inference_array(case,crop,spacing,output_shape=(128,128,128)).image.numpy(),[lo,hi]


def main():
    assert (sitk.Version_ITKMajorVersion(),sitk.Version_ITKMinorVersion(),sitk.Version_ITKPatchVersion())==(5,4,6)
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1);torch.set_num_threads(1)
    source=ROOT/"data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl"
    assert hashlib.sha256(source.read_bytes()).hexdigest()=="6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65"
    selected=sorted([json.loads(x) for x in source.read_text().splitlines()],key=lambda r:r["case_id"])[:3]
    start=time.monotonic();rows=[]
    for row in selected:
        nii=nib.as_closest_canonical(nib.load(row["image_path"]))
        original=np.ascontiguousarray(np.asanyarray(nii.dataobj),dtype=np.float32)
        assert row["split"]=="train" and original.min()==0 and np.isfinite(original).all()
        base=None;base_trace=None;geometry=None;conditions=[]
        for name,scale in (("raw_first",1.),("raw_repeat",1.),("half",.5),("double",2.)):
            data=(original*scale).astype(np.float32)
            output,trace=corrected(data)
            normalized,geo=normalize(row["case_id"],output,nib.affines.voxel_sizes(nii.affine))
            if base is None:base=normalized;base_trace=trace;geometry=geo
            difference=np.abs(normalized-base)
            counts={level:sum(t["level"]==level for t in trace) for level in sorted({t["level"] for t in trace})}
            item={"condition":name,"scale":scale,"input_support_exact":bool(np.array_equal(data>0,original>0)),
                  "positive_below_one":int(np.count_nonzero((data>0)&(data<1))),"output_bbox_exact":geometry==geo,
                  "normalized_array_exact":bool(np.array_equal(normalized,base)),"normalized_max_error":float(difference.max()),
                  "normalized_mean_error":float(difference.mean()),"iteration_counts":counts,"trace":trace,"trace_exact_to_raw":trace==base_trace}
            conditions.append(item);print(json.dumps({"case_id":row["case_id"],**{k:v for k,v in item.items() if k!="trace"}}),flush=True)
        rows.append({"case_id":row["case_id"],"conditions":conditions})
    result={"status":"completed_train_only_numeric_diagnostic","created_utc":datetime.now(timezone.utc).isoformat(),"rows":rows,
            "itk_version":"5.4.6","simpleitk_version":"2.5.5","old_5_4_0_endpoint_hypothesis":"ruled_out_by_actual_5_4_6_source",
            "label_access":False,"model_inference":False,"new_inner_or_old_cohort_metrics":False,"elapsed_seconds":time.monotonic()-start,
            "ongoing_no_n4_experiment_changed":False,"runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (OUT/"RAW_DIAGNOSTIC.json").open("x") as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))


if __name__=="__main__":main()
