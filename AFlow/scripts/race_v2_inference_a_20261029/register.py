"""Pre-register exactly32 global validation candidates before model access."""
from scripts.race_v2_inference_a_20261029.common import *
from scripts.race_v2_inference_a_20261029.predictor import definitions
def main():
    R.mkdir(parents=True,exist_ok=True)
    assert not (R/'PLAN.json').exists()
    cohort=rows();subjects=sorted({r['subject_id'] for r in cohort});assert len(cohort)==108 and len(subjects)==50
    subjects.sort(key=lambda s:hashlib.sha256(('race_v2_20261029:'+s).encode()).hexdigest())
    folds={s:i%5 for i,s in enumerate(subjects)}
    write(R/'PLAN.json',dict(at=now(),run_id='race164_validation108_calibration_20261029',checkpoint_sha256=CHECKPOINT_SHA,split='validation',scans=108,subjects=50,canvas=list(CANVAS),recipes=definitions(),configuration_limit=32,adaptive_search=False,folds=folds,fold_rule='sort subjects by sha256(race_v2_20261029:+subject_id),rank%5; 10 patients each; leave each fold out for selection and report held-out mean; all scans of subject stay together',selection='maximum development Scan mean; ties Subject then Point then candidate order; report stability of global candidates across held-out folds',candidate_families='3 references +5 consistency +8 temperature/bias +11 volume +5 component/hole; each change relative to fixed legacy recipe',deployment_GT_input=False,neural_forwards_per_case=8,neural_forwards_maximum=864,new270_scoring=False,test_tuned_history=True,independent_confirmation=False,bootstrap='10000 patient whole-cluster percentile seed20261029; descriptive after selection'))
    files=list((ROOT/'scripts/race_v2_inference_a_20261029').glob('*.py'))+[ROOT/'sammed3d/race_seg.py',ROOT/'scripts/race_seg_20261028/prepare.py',ROOT/'scripts/race_inference_test270_20261029/predictor.py',ROOT/'scripts/segresnet_auto_inference12h_20261028/promote_v6.py',ROOT/'scripts/segresnet_auto_inference12h_20261028/component_sweep_v1.py',ROOT/'scripts/segresnet_auto_inference12h_20261028/postprocess_sweep_v1.py',ROOT/'scripts/segresnet_auto_inference12h_20261028/cpu_binary_fast_v1.py',ROOT/'scripts/nnunet_local_full_20261022/native_builder_v2.py']
    write(R/'EXECUTION_FREEZE.json',dict(at=now(),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},plan_sha256=sha(R/'PLAN.json'),checkpoint_sha256=CHECKPOINT_SHA,scope='validation108 only; old270 audit read-only',cpu_decoder='historical case_consistency arithmetic reproduced on CPU; quantized65536 volume decoder; whole original crop scoring'))
if __name__=='__main__':main()
