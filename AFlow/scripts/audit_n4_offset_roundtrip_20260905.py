"""Train-image-only separation of scale and affine round-trip precision effects."""
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
from scripts.audit_n4_scale_diagnostics_20260905 import corrected,normalize,OUT


def main():
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1);torch.set_num_threads(1)
    source=ROOT/"data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl"
    selected=sorted([json.loads(x) for x in source.read_text().splitlines()],key=lambda r:r["case_id"])[:3]
    start=time.monotonic();rows=[]
    for row in selected:
        assert row["split"]=="train"
        nii=nib.as_closest_canonical(nib.load(row["image_path"]))
        original=np.ascontiguousarray(np.asanyarray(nii.dataobj),dtype=np.float32)
        mask=original>0;low=mask&(original<1);values=original[mask].astype(np.float64);mu=values.mean();sigma=values.std()
        direct=(original.astype(np.float64)/sigma).astype(np.float32)
        z32=((original.astype(np.float64)-mu)/sigma).astype(np.float32)
        restored32=(z32.astype(np.float64)-float(z32.min())).astype(np.float32)
        z64=(original.astype(np.float64)-mu)/sigma
        restored64=(z64-z64.min()).astype(np.float32)
        base=None;reference=None;conditions=[]
        for name,data,undo_scale in (("original",original,1.),("direct_scale",direct,sigma),("offset_float32",restored32,sigma),("offset_float64",restored64,sigma)):
            support_exact=bool(np.array_equal(data>0,mask))
            recovered=data.astype(np.float64)*undo_scale
            rel=np.abs(recovered[mask]-original[mask])/original[mask]
            low_rel=np.abs(recovered[low]-original[low])/original[low] if low.any() else np.array([0.])
            common=mask&(recovered>0)
            log_error=np.abs(np.log(recovered[common])-np.log(original[common].astype(np.float64)))
            corrected_image,trace=corrected(data)
            output,geo=normalize(row["case_id"],corrected_image,nib.affines.voxel_sizes(nii.affine))
            if base is None:base=output
            if name=="direct_scale":reference=output
            diff=np.abs(output-base)
            item={"condition":name,"mu":float(mu),"sigma":float(sigma),"support_exact":support_exact,
                  "max_relative_recovery_error":float(rel.max()),"low_positive_count":int(low.sum()),"low_positive_max_relative_error":float(low_rel.max()),
                  "max_log_recovery_error":float(log_error.max()),"normalized_max_error_vs_original":float(diff.max()),"normalized_mean_error_vs_original":float(diff.mean()),
                  "normalized_max_error_vs_direct":float(np.abs(output-reference).max()) if reference is not None else None,
                  "normalized_mean_error_vs_direct":float(np.abs(output-reference).mean()) if reference is not None else None,
                  "iteration_counts":{level:sum(t["level"]==level for t in trace) for level in sorted({t["level"] for t in trace})},"trace":trace}
            conditions.append(item);print(json.dumps({"case_id":row["case_id"],**{k:v for k,v in item.items() if k!="trace"}}),flush=True)
        rows.append({"case_id":row["case_id"],"conditions":conditions})
    result={"status":"completed_train_only_cause_isolation","created_utc":datetime.now(timezone.utc).isoformat(),"rows":rows,
            "elapsed_seconds":time.monotonic()-start,"label_access":False,"model_inference":False,"ongoing_no_n4_changed":False,
            "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (OUT/"RAW_OFFSET_ROUNDTRIP.json").open("x") as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))


if __name__=="__main__":main()
