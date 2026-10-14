"""Read only the 362 frozen raw manual labels and report integer domains."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from scripts import utsw_frozen_baseline_external_protocol as p

def main():
    p.verify_preaccess();download=json.loads(p.SOURCE_DOWNLOAD.read_text())
    if download.get("status")!="passed" or download.get("files_copy_exact")!=724 or download.get("voxel_arrays_accessed") is not False:raise ValueError("source receipt")
    rows=[];sets=Counter();value_cases=Counter();value_voxels=Counter()
    for row in p.cohort():
        sid=row["subject_id"];path=p.SOURCE/"copy_a"/sid/"rtumorseg_manual_correction.nii.gz";array=np.asanyarray(nib.load(str(path),mmap=True).dataobj)
        finite=bool(np.isfinite(array).all());integer=finite and bool(np.array_equal(array,np.rint(array)));values,counts=np.unique(array,return_counts=True) if finite else ([],[])
        value_list=[float(x) for x in values];count_map={str(float(v)):int(c) for v,c in zip(values,counts)}
        rows.append({"subject_id":sid,"finite":finite,"integer_exact":integer,"values":value_list,"counts":count_map})
        sets[tuple(value_list)]+=1
        for v,c in zip(values,counts):value_cases[float(v)]+=1;value_voxels[float(v)]+=int(c)
    union=sorted(value_cases);valid=all(x["finite"] and x["integer_exact"] and 0. in x["values"] and any(v>0 for v in x["values"]) for x in rows)
    report={"schema":"aflow.utsw_manual_label_domain_diagnostic","created_utc":datetime.now(timezone.utc).isoformat(),
        "status":"finite_integer_0_to_4" if valid and set(union)<={0.,1.,2.,3.,4.} else "unexpected_domain",
        "cases":362,"union_values":union,"set_frequencies":{str(k):v for k,v in sorted(sets.items())},
        "value_case_counts":{str(k):v for k,v in sorted(value_cases.items())},"value_voxel_counts":{str(k):v for k,v in sorted(value_voxels.items())},
        "label3_cases":value_cases[3.],"label4_cases":value_cases[4.],"label3_and_4_cases":sum(3. in x["values"] and 4. in x["values"] for x in rows),
        "rows":rows,"source_download_sha256":p.sha(p.SOURCE_DOWNLOAD),"image_voxels_accessed":False,"n4_or_cache":False,"model_inference":False,"metrics":False,
        "runner_sha256":p.sha(Path(__file__))}
    out=p.RECORD/"LABEL_DOMAIN_DIAGNOSTIC.json"
    if out.exists():raise FileExistsError(out)
    p.write_new(out,report);print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2,sort_keys=True))
if __name__=="__main__":main()
