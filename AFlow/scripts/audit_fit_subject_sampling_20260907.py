"""Audit fit-only sampling probability concentration across longitudinal subjects."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
from scripts import structured_background_protocol as p
from scripts.build_t1ce_wt_native_volume_artifact import native_sampling_weights


def main():
    out=p.ROOT/'archive/reviews/2026-09-07_fit_subject_sampling/RESULT.json'
    if out.exists():raise FileExistsError(out)
    fit,_=p.check_roles();p.checked(p.NATIVE,p.NATIVE_SHA)
    native=json.loads(p.NATIVE.read_text())['native_wt_voxels']
    weights=native_sampling_weights([r['case_id'] for r in fit],native).numpy()
    case_prob=weights/weights.sum()
    by_subject=defaultdict(list)
    for i,row in enumerate(fit):by_subject[row['subject_id']].append(i)
    records=[]
    for subject,slots in sorted(by_subject.items()):
        mass=float(case_prob[slots].sum())
        records.append({'subject_id':subject,'scans':len(slots),'native_sampling_probability':mass,'expected_draws_per_epoch':mass*968,'probability_no_draw_per_epoch':float((1-mass)**968)})
    probabilities=np.array([r['native_sampling_probability'] for r in records])
    sizes=np.array([r['scans'] for r in records])
    uniform=1/len(records)
    top=sorted(records,key=lambda r:r['native_sampling_probability'],reverse=True)[:32]
    result={'schema':'aflow.fit_subject_sampling_audit','created_utc':datetime.now(timezone.utc).isoformat(),'status':'fit_metadata_only_no_sampler_or_model_change','fit_scans':len(fit),'fit_subjects':len(records),'fit_sha256':p.FIT_SHA,'native_volume_artifact_sha256':p.NATIVE_SHA,'scans_per_subject_histogram':dict(sorted(Counter(int(x) for x in sizes).items())),'native_subject_probability':{'min':float(probabilities.min()),'max':float(probabilities.max()),'max_to_min':float(probabilities.max()/probabilities.min()),'top32_subject_probability_mass':sum(r['native_sampling_probability'] for r in top),'concentration_equivalent_uniform_subjects':float(1/np.square(probabilities).sum()),'expected_unique_subjects_per968_draws':float(np.sum(1-(1-probabilities)**968))},'uniform_subject_reference_only':{'probability':uniform,'expected_draws_per_subject':968*uniform,'probability_no_draw':(1-uniform)**968,'expected_unique_subjects_per968_draws':len(records)*(1-(1-uniform)**968)},'subjects':records,'interpretation':'Compare probability allocation with the subject-equal primary metric; concentration-equivalent count is not a statistical effective sample size or independent validation. Uniform-subject reference is arithmetic only, not an implemented sampler or performance result.','access':{'fit_manifest_and_archived_fit_volumes':True,'image_or_label_arrays':False,'development_metrics':False,'new_model_forwards':0,'sampler_changed':False}}
    p.write_new(out,result)
    print(json.dumps({k:v for k,v in result.items() if k!='subjects'},indent=2))


if __name__=='__main__':main()
