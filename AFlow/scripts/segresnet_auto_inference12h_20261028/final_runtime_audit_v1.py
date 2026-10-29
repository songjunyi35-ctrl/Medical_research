"""Read-only final input/model hashes and actual disk budgets; no model execution."""
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_v2 import stable_size


def main():
    resource=read(R/'RESOURCE_CURRENT.json');actual=dict(images=sha(CACHE/'images.f32'),labels=sha(CACHE/'labels.u8'),index=sha(CACHE/'index.json'));expected=read(OLD/'CACHE_READY.json')['payload_sha256']
    persistent=stable_size(R);scratch=stable_size(TMP)+stable_size(CACHE)+read(R/'MAIN_SCRATCH_BUDGET.json')['bytes'];root_free=shutil.disk_usage(ROOT).free;scratch_free=shutil.disk_usage(TMP).free;checkpoint=sha(CHECKPOINT)
    source_checks=[]
    for q in ['queue_v2','queue_aux_v1','analysis_queue_v2','analysis_queue_aux_v1','cpu_queue_v2','cpu_aux_queue_v1']:
        for path in (R/q).glob('*.json'):
            job=read(path)
            if job.get('resource_token') not in [None,resource['token']]:continue
            for source,digest in job.get('source_sha256',{}).items():source_checks.append(dict(path=source,expected=digest,actual=sha(ROOT/source)))
    all_sources=all(s['actual']==s['expected'] for s in source_checks)
    checks=dict(native_payloads_and_index_unchanged=actual==expected,checkpoint_unchanged=checkpoint==CHECKPOINT_SHA,persistent_under2GiB=persistent<2*2**30,root_free_at_least20GiB=root_free>=20*2**30,total_new_scratch_under300GiB=scratch<300*2**30,scratch_free_at_least70GiB=scratch_free>=70*2**30,all_registered_source_hashes_unchanged=all_sources)
    result=dict(at=now(),state='passed' if all(checks.values()) else 'failed',checks=checks,worker=resource['worker'],payload_sha256=actual,expected_payload_sha256=expected,checkpoint_sha256=checkpoint,persistent_bytes=persistent,persistent_free_bytes=root_free,scratch_bytes_including_main_allowance=scratch,scratch_free_bytes=scratch_free,main_scratch_allowance=read(R/'MAIN_SCRATCH_BUDGET.json'),source_hash_checks=source_checks,source_sha256=sha(Path(__file__)),patient_model_forward=False,GT_pixel_scoring=False)
    write(R/'FINAL_RUNTIME_AUDIT.json',result);print(dict(state=result['state'],checks=checks,persistent_gib=persistent/2**30,scratch_gib=scratch/2**30));assert result['state']=='passed'

if __name__=='__main__':main()
