"""Recount every candidate/case with NumPy, not the GPU scoring implementation."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import numpy as np
from sammed3d.workflow_score_audit import numpy_workflow_score
from scripts import structured_background_protocol as p

RECORD=p.ROOT/'archive/research_records/2026-09-11_automatic_workflow_search'


def main():
    output=RECORD/'SCORE_VERIFICATION.json'
    if output.exists(): raise FileExistsError(output)
    raw=json.loads((RECORD/'RAW_DEVELOPMENT.json').read_text())
    pre=json.loads((RECORD/'DEV_PREACCESS.json').read_text())
    p.checked(p.CACHE/'index.json',pre['index_sha256'])
    p.checked(p.CACHE/'labels.u8',pre['labels_sha256'])
    index=json.loads((p.CACHE/'index.json').read_text())
    lookup={case:i for i,case in enumerate(index['case_ids'])}
    labels=np.memmap(p.CACHE/'labels.u8',mode='r',dtype=np.uint8,shape=(len(lookup),128,128,128))
    count=0;started=time.monotonic();checks=[]
    try:
        for seed, info in raw['results'].items():
            p.checked(info['path'],info['sha256'])
            data=json.loads(Path(info['path']).read_text())
            cache=data['atom_cache'];p.checked(cache['path'],cache['sha256'])
            assert cache['shape']==[180,8,128,128,128] and cache['dtype']=='float32'
            atoms=np.memmap(cache['path'],mode='r',dtype=np.float32,shape=tuple(cache['shape']))
            def verify_case(slot):
                case=pre['case_ids'][slot]
                truth=np.array(labels[lookup[case]],copy=True)
                case_atoms=np.array(atoms[slot],copy=True)
                assert np.isfinite(case_atoms).all()
                for k in range(128):
                    expected=numpy_workflow_score(case_atoms,truth,k)
                    row=data['rows'][str(k)][slot]
                    assert row['case_id']==case
                    if any(row[key]!=value for key,value in expected.items()):
                        raise AssertionError(f'{seed}/{case}/{k}: {expected} vs {row}')
                return slot
            with ThreadPoolExecutor(max_workers=4) as pool:
                for slot in pool.map(verify_case,range(180)):
                    count+=128
                    if (slot+1)%24==0:
                        print(json.dumps({'seed':seed,'cases':slot+1,'candidate_case_scores_verified':count,
                                          'seconds':time.monotonic()-started}),flush=True)
            checks.append({'seed':seed,'raw_sha256':info['sha256'],'atom_sha256':cache['sha256'],
                           'cases':180,'workflows':128,'all_score_fields_exact':True})
        assert count==46080
        p.write_new(output,{'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),
            'checks':checks,'candidate_case_scores_verified':count,'seconds':time.monotonic()-started,
            'raw_sha256':p.sha(RECORD/'RAW_DEVELOPMENT.json'),
            'preaccess_sha256':p.sha(RECORD/'DEV_PREACCESS.json'),
            'verifier_sha256':p.sha(Path(__file__)),
            'numpy_implementation_sha256':p.sha(p.ROOT/'sammed3d/workflow_score_audit.py'),
            'new_model_forwards':0,'access':'same reused development labels; no protected cohorts'})
        print('Verified all 46080 scores',flush=True)
    except BaseException as error:
        p.write_new(RECORD/'SCORE_VERIFICATION_FAILURE.json',{'error':repr(error),
            'verified_candidate_case_scores':count,'checks':checks,'new_model_forwards':0})
        raise


if __name__=='__main__':main()
