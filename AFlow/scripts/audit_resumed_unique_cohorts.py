"""Count unique patients separately from repeated model forwards, using saved rows only."""
from datetime import datetime,timezone
import json
from scripts import structured_background_protocol as p


def main():
    session=p.ROOT/'archive/research_records/2026-09-10_resumed_eight_hour'
    fit,dev=p.check_roles();expected={(r['case_id'],r['subject_id']) for r in dev}
    uses={};all_rows=[];sources={}
    for study in ['small_rotation','image_blur','development_error_anatomy']:
        path=p.ROOT/f'archive/research_records/2026-09-10_{study}/RAW_DEVELOPMENT.json';raw=json.loads(path.read_text());sources[str(path.relative_to(p.ROOT))]=p.sha(path)
        rows=[]
        for key,value in raw['results'].items():
            cases=value if isinstance(value,list) else value['per_case']
            assert len(cases)==180 and {(r['case_id'],r['subject_id']) for r in cases}==expected
            rows.extend(cases)
        assert len(rows)==raw['new_forwards']
        uses[study]={'model_outputs':len(raw['results']),'forwards':len(rows),'unique_scans':len({r['case_id'] for r in rows}),'unique_subjects':len({r['subject_id'] for r in rows})};all_rows.extend(rows)
    assert len(all_rows)==1800 and len({r['case_id'] for r in all_rows})==180 and len({r['subject_id'] for r in all_rows})==78
    train_reports=[]
    for study in ['small_rotation','image_blur']:
        work=p.ROOT/f'workspace/BraTS/sammed3d_t1ce_wt/{study}_20260910'
        for path in sorted(work.glob('seed*/TRAIN_DONE.json')):
            r=json.loads(path.read_text());assert r['fit_sha256']==p.FIT_SHA and not r['teacher_checkpoints'] and not r['access']['development_metrics']
            train_reports.append({'path':str(path.relative_to(p.ROOT)),'sha256':p.sha(path),'fit_sha256':r['fit_sha256']})
    assert len(train_reports)==6
    result={'status':'saved_unique_cohort_accounting_verified','created_utc':datetime.now(timezone.utc).isoformat(),'fit':{'scans':len(fit),'subjects':len({r['subject_id'] for r in fit}),'manifest_sha256':p.FIT_SHA},'development':{'unique_scans':180,'unique_subjects':78,'repeated_forwards':1800,'manifest_sha256':p.DEV_SHA,'not_1800_independent_validation_cases':True},'uses':uses,'six_training_reports':train_reports,'source_sha256':sources,'new_forwards':0,'new_patient_arrays_read':False,'test_or_protected_cohort_model_access':False,'code_sha256':p.sha(__file__)}
    p.write_new(session/'UNIQUE_COHORT_ACCOUNTING.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['six_training_reports','source_sha256']},indent=2))


if __name__=='__main__':main()
