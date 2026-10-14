"""List fit-patient local filenames only; never read image or label arrays."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
from scripts import structured_background_protocol as p


def main():
    out=p.ROOT/'archive/reviews/2026-09-07_fit_modality_availability/RESULT.json'
    if out.exists():raise FileExistsError(out)
    fit,_=p.check_roles()
    counts=Counter();details=[]
    for row in fit:
        directory=Path(row['image_path']).parent
        names=sorted(path.name for path in directory.iterdir() if path.is_file())
        modalities=[]
        for modality in ('t1c','t1n','t2w','t2f'):
            if any(name in names for name in (row['case_id']+'-'+modality+'.nii',row['case_id']+'-'+modality+'.nii.gz')):
                modalities.append(modality);counts[modality]+=1
        extra=[name for name in names if name not in (Path(row['image_path']).name,Path(row['label_path']).name)]
        details.append({'case_id':row['case_id'],'modalities':modalities,'other_local_filenames':extra})
    result={'schema':'aflow.fit_modality_availability','created_utc':datetime.now(timezone.utc).isoformat(),'status':'metadata_inventory_completed','fit_scans':len(fit),'fit_subjects':len({r['subject_id'] for r in fit}),'fit_sha256':p.FIT_SHA,'local_modality_file_counts':{k:counts[k] for k in ('t1c','t1n','t2w','t2f')},'cases_with_other_files':sum(bool(r['other_local_filenames']) for r in details),'details':details,'access':{'filename_listing_only':True,'image_arrays':False,'label_arrays':False,'development_directories':False,'model_forwards':0},'scope':'Only selected fit patient directories; no claim about data on remote servers or unrelated directories.'}
    p.write_new(out,result)
    print(json.dumps({k:v for k,v in result.items() if k!='details'},indent=2))


if __name__=='__main__':main()
