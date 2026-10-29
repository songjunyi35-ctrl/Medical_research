"""Freeze16 finite local-dependency recipes before any new108 scoring."""
from scripts.race_v2_inference_innovation_a_20261029.common import *
from scripts.race_v2_inference_innovation_a_20261029.algorithm import definitions,numerical_gate

def main():
    R.mkdir(parents=True,exist_ok=True);assert not (R/'PLAN.json').exists()
    statistics=read(STAGE/'STATISTICS_FREEZE.json');reference=read(BASE/'RAW_VALIDATION108.json')['per_configuration']['equal8_logit_t05']
    atoms=read(BASE/'ATOMS_READY.json');assert atoms['checkpoint_sha256']==CHECKPOINT_SHA and atoms['case_forwards']==864
    write(R/'PLAN.json',dict(at=now(),run_id='race164_8view_local_dependency_innovation_20261029',
        recipes=definitions(),new_configuration_limit=16,reference='equal8_logit_t05',reference_scan=.7680118805556385,
        source_checkpoint_sha256=CHECKPOINT_SHA,source_atoms_manifest_path=str(BASE/'ATOMS_READY.json'),
        source_atoms_manifest_sha256=sha(BASE/'ATOMS_READY.json'),source_reference_sha256=sha(BASE/'RAW_VALIDATION108.json'),
        source_cache_path=str(CACHE),source_cache_index_sha256=sha(BASE/'CACHE_READY.json'),
        split='validation',scans=108,subjects=50,folds=statistics['fold_assignment'],
        fold_selection='four folds choose highestScan thenSubject thenPoint_refined then original order including reference first; evaluate heldout fold',
        seed=20261029,bootstrap=10000,point_primary='actual final refined score argmax',point_auxiliary='equal8 score argmax',
        image_spacing_mm=[1,1,1],native_score='whole original crop GT; exclude canvas padding only',
        new_network_forwards=0,new270_scoring=False,training_updates=0,deploy_GT_or_ID_input=False,
        maximum_wall_seconds=2700,persistent_max_bytes=64*2**20,memory_ceiling_gib=16,
        adaptive_history='new family after observing old32 RACE validation; inherits test-tuned history, not independent confirmation',
        source_models=1,retained_neural_views_per_case=8))
    write(R/'CPU_GATE.json',dict(at=now(),**numerical_gate('cpu')))
    history_files=['consensus_v1.py','case_consistency_v1.py','core_fusions_v3.py','nonlinear_fusions_v1.py',
        'joint_fusions_v1.py','local_mrf_v1.py','shape_fusion_sweep_v1.py','local_brain_joint_v1.py']
    write(R/'HISTORY_DEDUP.json',dict(at=now(),source_sha256={f'scripts/segresnet_auto_inference12h_20261028/{p}':sha(ROOT/'scripts/segresnet_auto_inference12h_20261028'/p) for p in history_files},
        old_global_consistency='one scalar weight per view/case',old_voxel_confidence='independent inverse p(1-p) per voxel',
        old_joint='product of scalar case agreement and independent voxel confidence, not8x8 dependency',
        old_consensus='global EM sensitivity/specificity or scalar self-selection',old_MRF='spatial Potts smoothing after fusion',
        old_SDF='global shape-distance mean/trim/quantile and whole-case weights',
        new_mechanism='spatial 8x8 cross-view disagreement moments and regularized inverse-dependency weights with diagonal control',
        LOO_warning='LOO mean residual is merely8/7 of mean residual; this implementation uses other7 majority sign, not that renamed identity',
        no_claim_of_first_in_literature=True,no_old790million_or512phase_repeat=True))
    paths=list((ROOT/'scripts/race_v2_inference_innovation_a_20261029').glob('*.py'))
    paths+=[ROOT/'scripts/race_v2_inference_a_20261029/statistics.py']
    write(R/'PREPARATION_FREEZE.json',dict(at=now(),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        plan_sha256=sha(R/'PLAN.json'),source_atoms_manifest_sha256=sha(BASE/'ATOMS_READY.json'),
        statistics_freeze_sha256=sha(STAGE/'STATISTICS_FREEZE.json'),clinical_arrays_accessed_by_registration=False))

if __name__=='__main__':main()
