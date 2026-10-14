"""Recoverable UTSW362 raw-label domain diagnostic, no images/models."""
from collections import Counter
from datetime import datetime,timezone
import json,traceback
from pathlib import Path
import nibabel as nib
import numpy as np
from scripts import utsw_frozen_baseline_external_protocol as p

PROGRESS=p.RECORD/"LABEL_DOMAIN_DIAGNOSTIC_PROGRESS_V2.json"
OUTPUT=p.RECORD/"LABEL_DOMAIN_DIAGNOSTIC_AMENDMENT_01.json"
FAILURE=p.RECORD/"LABEL_DOMAIN_DIAGNOSTIC_FAILURE_V2.json"
def atomic(path,value):
    temp=path.with_suffix(path.suffix+".tmp");temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n");temp.replace(path)
def main():
    if any(x.exists() for x in (PROGRESS,OUTPUT,FAILURE)):raise FileExistsError("v2 diagnostic already started")
    p.verify_preaccess();download=json.loads(p.SOURCE_DOWNLOAD.read_text())
    if download.get("status")!="passed" or download.get("files_copy_exact")!=724 or download.get("voxel_arrays_accessed") is not False:raise ValueError("source receipt")
    rows=[];sets=Counter();value_cases=Counter();value_voxels=Counter();current=None
    try:
        for i,row in enumerate(p.cohort(),1):
            current=row["subject_id"];path=p.SOURCE/"copy_a"/current/"rtumorseg_manual_correction.nii.gz";nii=nib.load(str(path),mmap=True)
            array=np.asarray(nii.dataobj,dtype=np.float32);finite=bool(np.isfinite(array).all());integer=finite and bool(np.array_equal(array,np.rint(array)))
            values,counts=np.unique(array,return_counts=True) if finite else (np.array([]),np.array([]));value_list=[float(x) for x in values]
            rows.append({"subject_id":current,"finite":finite,"integer_exact":integer,"values":value_list,"counts":{str(float(v)):int(c) for v,c in zip(values,counts)}})
            sets[tuple(value_list)]+=1
            for v,c in zip(values,counts):value_cases[float(v)]+=1;value_voxels[float(v)]+=int(c)
            del array,nii,values,counts
            if i%25==0:atomic(PROGRESS,{"created_utc":datetime.now(timezone.utc).isoformat(),"completed":i,"last_subject":current,"image_arrays":False,"model":False})
    except BaseException as exc:
        atomic(FAILURE,{"created_utc":datetime.now(timezone.utc).isoformat(),"status":"failed","subject":current,"completed":len(rows),
            "exception_type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc(),"image_arrays":False,"model":False});raise
    union=sorted(value_cases);valid=all(x["finite"] and x["integer_exact"] and 0. in x["values"] and any(v>0 for v in x["values"]) for x in rows)
    report={"schema":"aflow.utsw_manual_label_domain_diagnostic_amendment_01","created_utc":datetime.now(timezone.utc).isoformat(),
        "status":"finite_integer_0_to_4" if valid and set(union)<={0.,1.,2.,3.,4.} else "unexpected_domain","cases":len(rows),"union_values":union,
        "set_frequencies":{str(k):v for k,v in sorted(sets.items())},"value_case_counts":{str(k):v for k,v in sorted(value_cases.items())},
        "value_voxel_counts":{str(k):v for k,v in sorted(value_voxels.items())},"label3_cases":value_cases[3.],"label4_cases":value_cases[4.],
        "label3_and_4_cases":sum(3. in x["values"] and 4. in x["values"] for x in rows),"rows":rows,"source_download_sha256":p.sha(p.SOURCE_DOWNLOAD),
        "image_voxels_accessed":False,"n4_or_cache":False,"model_inference":False,"metrics":False,"runner_sha256":p.sha(Path(__file__))}
    p.write_new(OUTPUT,report);print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2,sort_keys=True))
if __name__=="__main__":main()
