"""Bind final interoperability names without altering original run artifacts."""
from scripts.race_v2_inference_normalization_a_20261029.common import *

def main():
    assert read(R/'DONE.json')['state']=='complete';out=R/'NORMALIZATION_DEPLOYMENT_RECIPE.json';assert not out.exists()
    original=read(R/'DEPLOYMENT_RECIPE.json');entry=original['innovation_entry'];entry['mode']='race8_seed_health_normalization'
    result=read(R/'RESULT.json')
    write(out,dict(at=now(),innovation_entry=entry,best_overall_id=result['best']['id'],metrics=result['best_innovation']['metrics'],
        source_result_path=str(R/'RESULT.json'),source_result_sha256=sha(R/'RESULT.json'),
        original_run_recipe_path=str(R/'DEPLOYMENT_RECIPE.json'),original_run_recipe_sha256=sha(R/'DEPLOYMENT_RECIPE.json'),
        raw_path=str(R/'RAW_VALIDATION108.json'),raw_sha256=sha(R/'RAW_VALIDATION108.json'),
        mask_audit_path=str(R/'MASK_AUDIT_VALIDATION108.json'),mask_audit_sha256=sha(R/'MASK_AUDIT_VALIDATION108.json'),
        deploy_callable='scripts.race_v2_inference_normalization_a_20261029.deployment.deploy',
        cache_callable='scripts.race_v2_inference_normalization_a_20261029.cache.generate',
        input_recipe=entry,input_recipe_sha256=sha(R/'DEPLOYMENT_RECIPE.json'),
        seed_dependency='same-weight original-input identity logit only; produced first or audited cached view0',
        normalized_views=list(range(1,8)),logical_forwards_per_scan=8,new270_access=False))

if __name__=='__main__':main()
