"""Freeze eight explicit configurations before any new validation forwarding."""
from scripts.race_v2_inference_normalization_a_20261029.common import *
from scripts.race_v2_inference_normalization_a_20261029.algorithm import definitions,groups,numerical_gate

def main():
    R.mkdir(parents=True,exist_ok=True);assert not (R/'PLAN.json').exists()
    plan=dict(at=now(),run_id='race164_seed_healthy_normalization_20261029',groups=groups(),recipes=definitions(),
        maximum_new_configurations=8,maximum_new_patient_forwards=3024,maximum_wall_seconds=1800,
        checkpoint_sha256=CHECKPOINT_SHA,source_atoms_ready_path=str(BASE/'ATOMS_READY.json'),
        source_atoms_ready_sha256=sha(BASE/'ATOMS_READY.json'),source_cache_index_sha256=sha(CACHE/'index.json'),
        reference='equal8_logit_t05',source_reference_sha256=sha(BASE/'RAW_VALIDATION108.json'),
        source_raw_identity='same checkpoint original-image flip0; already predicted before healthy statistics',
        split='validation108',subjects=50,folds=read(STAGE/'STATISTICS_FREEZE.json')['fold_assignment'],
        bootstrap=10000,seed=20261029,fold_selection='four folds maxScan thenSubject thenPoint then original order including reference first',
        final_score='whole original crop GT, native1mm, threshold logit0',
        point_primary='actual final weighted logit argmax',point_auxiliary='old equal8 logit argmax',
        deployment_forwards=8,deployment_order='original identity seed1 then normalized flips1..7; never full8seed then new8',
        preserve_original_valid_reference_support=True,mirror='T@R@T with same transformed original support',
        fallback='identity when healthy support<20percent brain or SD<0.1; reuse old same-model flips',
        duplicate_inputs='per-case exact affine coefficients share earlier canonical group; no repeated identical network inputs',
        training_updates=0,new270_scoring=False,GT_or_ID_in_deploy=False,
        adaptive_history='requested after old32/local16 completed; validation reused and test-tuned history, not independent')
    write(R/'PLAN.json',plan);write(R/'CPU_GATE.json',dict(at=now(),**numerical_gate()))
    paths=['scripts/segresnet_auto_inference12h_20261028/intensity_gate_v1.py',
        'scripts/segresnet_auto_inference12h_20261028/render_v1.py',
        'scripts/segresnet_auto_inference12h_20261028/brain_phase_prepare_v1.py',
        'scripts/segresnet_auto_inference12h_20261028/intensity_ensemble_v1.py']
    write(R/'HISTORY_DEDUP.json',dict(at=now(),source_sha256={p:sha(ROOT/p) for p in paths},
        old_intensity='28 globally fixed gain/offset/signed gamma groups; no model-seed exclusion or per-image healthy moments',
        old_ROI_and_phase='brain/ROI geometry and expensive512phase, not this intensity-only affine feedback',
        old_local_dependency='spatial output covariance weighting, no new image or network views',
        new='one original seed -> exclude suspected lesion -> bounded per-image healthy moment affine -> seven mirror views',
        no_first_in_literature_claim=True,no_old_search_repeat=True))
    write(R/'LITERATURE.json',dict(at=now(),sources=[
        dict(title='Statistical normalization techniques for magnetic resonance imaging',doi='10.1016/j.nicl.2014.08.008',
            url='https://adni.loni.usc.edu/adni-publications/Statistical%20normalization%20techniques%20for%20magnetic%20resonance%20imaging.pdf',
            accessed='primary full PDF via web',summary='MRI units vary between scans. Paper motivates stable normal-appearing tissue normalization and avoiding pathology-driven intensity references. WhiteStripe uses normal-appearing white matter, unlike our coarse predicted-nonlesion support; it does not establish segmentation gains for frozen RACE T1ce.'),
        dict(title='Negligible effect of brain MRI data preprocessing for tumor segmentation',url='https://arxiv.org/abs/2204.05278',
            accessed='primary author abstract v4',summary='Experiments find many common standardization steps add no segmentation value or can hurt. This is direct negative context; healthy-seed affine correction is a bounded hypothesis rather than a promised gain.')]))
    (R/'IDEA.md').write_text('''# Seven prospective questions

1. Problem: whole-brain moment normalization can depend on abnormal tissue intensity. Can a conservative seed-excluded reference improve this fixed RACE model?
2. Evidence: WhiteStripe motivates healthy tissue references; it is not this method and does not guarantee tumor segmentation benefit. Primary preprocessing research also reports null/harmful effects.
3. Difference: old fixed gamma/gain/offset, geometric ROI/phase and output covariance do not compute per-image affine coefficients from a preceding lesion seed.
4. Mechanism: original identity seed excludes suspected disease with a fixed3-voxel Chebyshev dilation. Remaining nonzero image voxels estimate mean/SD. Bounded positive affine and partial strength retain anatomy, contrast ordering and original reference support.
5. Test: exactly4 inputs x2 output seed weights=8 new configurations against equal8, fixed patient5fold and patient-cluster10000 seed20261029 intervals; preserve all worsening cases and primary final-score Point.
6. Failure risks: seed omissions contaminate healthy reference; false-positive seed excludes normal tissue; nonlesion brain is not white matter; frozen training normalization may already be adequate; replacing7 original views can lose robustness. Fallback and bounds are prospective, not GT routing.
7. Budget/deployment: new108 neural forwards<=3024, existing seed cached; deployment1 original seed+7 adapted mirrors=8. One family, no training, no270 access, same geometry and score, <=30min. Same inputs alias cached predictions rather than repeat neural work.
''')
    source=dict(read(BASE/'EXECUTION_FREEZE.json')['source_sha256'])
    dependencies=list((ROOT/'scripts/race_v2_inference_normalization_a_20261029').glob('*.py'))
    dependencies += [ROOT/'scripts/race_v2_inference_innovation_a_20261029/io_helpers.py',
        ROOT/'scripts/race_v2_inference_innovation_a_20261029/common.py',ROOT/'scripts/race_v2_inference_a_20261029/statistics.py']
    source.update({str(p.relative_to(ROOT)):sha(p) for p in dependencies})
    write(R/'PREPARATION_FREEZE.json',dict(at=now(),source_sha256=source,plan_sha256=sha(R/'PLAN.json'),
        CPU_gate_sha256=sha(R/'CPU_GATE.json'),seed_atoms_ready_sha256=sha(BASE/'ATOMS_READY.json'),
        literature_sha256=sha(R/'LITERATURE.json'),idea_sha256=sha(R/'IDEA.md')))

if __name__=='__main__':main()
