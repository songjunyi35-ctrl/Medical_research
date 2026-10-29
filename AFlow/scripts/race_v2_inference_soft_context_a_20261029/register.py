"""Prospective four-input freeze; output weights and all fallbacks are fixed."""
from scripts.race_v2_inference_soft_context_a_20261029.common import *
from scripts.race_v2_inference_soft_context_a_20261029.algorithm import definitions,numerical_gate

def main():
    R.mkdir(parents=True,exist_ok=True);assert not (R/'PLAN.json').exists()
    write(R/'PLAN.json',dict(at=now(),run_id='race164_soft_context_20261029',recipes=definitions(),
        configuration_limit=4,maximum_new_patient_forwards=3024,maximum_stage_seconds=1200,
        checkpoint_sha256=CHECKPOINT_SHA,source_atoms_ready_sha256=sha(BASE/'ATOMS_READY.json'),
        source_reference_sha256=sha(BASE/'RAW_VALIDATION108.json'),
        cache_index_sha256=read(BASE/'CACHE_READY.json')['index_sha256'],scans=108,subjects=50,
        folds=read(STAGE/'STATISTICS_FREEZE.json')['fold_assignment'],bootstrap=10000,seed=20261029,
        seed_dependency='same-weight original image identity logit only, preceding all spatial statistics',
        formula='S=seedp>=0.1 intersect original support; d=Euclidean distance toS in1mm; phi=exp(-d²/(2r²)); field=max(phi,phi composed with original reflection), unless reference disabled; input=x*(floor+(1-floor)*field)',
        empty_seed='field1, reuse original mirrors',reference_disabled='phi only',
        mirror='same original support and R_view=T@R@T; only scalar gain field is interpolated, never image',
        decode='seed1/8 plus adapted7 each1/8; threshold0; final-score argmax Point',
        reference='equal8_logit_t05',new270_scoring=False,training_updates=0,GT_or_ID_in_deploy=False,
        native_score='whole original crop GT; only canvas padding excluded',
        prior='weak; feature-attention training papers do not validate frozen-network input gating',
        test_tuned_history=True,independent_confirmation=False))
    write(R/'CPU_GATE.json',dict(at=now(),**numerical_gate()))
    history=['scripts/segresnet_auto_inference12h_20261028/roi_render_v2.py',
        'scripts/segresnet_auto_inference12h_20261028/render_v1.py',
        'scripts/segresnet_auto_inference12h_20261028/phase512_v2.py',
        'archive/research_records/2026-10-28_segresnet_auto_inference12h/ALL_COMPLETED_ANALYSES.tsv']
    write(R/'HISTORY_AND_LITERATURE.json',dict(at=now(),source_sha256={p:sha(ROOT/p) for p in history},
        old_ROI='predicted bbox crop/resize to canvas then feather output into original5-view probability; six ROI fusion grids best0.780745, below old final0.784382; not a fresh causal comparison',
        old_phase='padding translations/512 phases, no seed-shaped continuous input gain',
        old_intensity='global fixed gain/offset/gamma and latest healthy-moment affine; no spatial context field',
        primary_source=dict(title='Attention U-Net: Learning Where to Look for the Pancreas',url='https://arxiv.org/abs/1804.03999',
            summary='Learned feature gates suppress irrelevant background in trained abdominal CT networks. It supplies only a weak analogy: our fixed input gain is neither learned feature attention nor a reproduction, and no benefit follows from that paper.'),
        key_risks=['seed omissions are still attenuated although never hard-zeroed','shared model errors may persist','instance normalization propagates context changes','symmetric protection can make change small'],
        no_novelty_or_benefit_claim=True))
    (R/'IDEA.md').write_text('''# Seven questions

1. Problem: most paired-model errors are shared or outside the existing logits convex hull; output reweighting has limited scope. Test an input intervention that may change predictions, not a claim of new patient information.
2. Evidence: trained feature attention motivates suppressing irrelevant context but does not establish this frozen RACE input intervention. It is a weak analogy.
3. History: older ROI crops/resizes and stitches probabilities; phase shifts padded inputs; fixed/global photometric groups and healthy moments are different. Older ROI grids were negative relative to the old final recipe, so success is not assumed.
4. Mechanism: original identity seed defines a smooth distance gain. Preserve the original grid, every original nonzero support voxel, and reference matrix; symmetrically protect contralateral context. No image interpolation or renormalization.
5. Falsification: exact r10/20 x floor0.75/0.9, four configurations only, fixed output equal weights, whole original GT crop, final-score Point, equal8 reference; five patient folds and10000 patient bootstrap seed20261029; retain all worsening cases.
6. Risks: omitted lesions lose contrast; false positives preserve irrelevant context; InstanceNorm changes can propagate globally; bilateral protection may leave too little intervention. Nonzero input does not guarantee preservation of undetected lesions.
7. Budget: original seed1+adapted7=8 deployment forwards, at most3024 new validation forwards, <20minutes; fields cached once per radius and shared across floors. No training, extra model or270 access. Gate failure stops rather than changing parameters.
''')
    source=dict(read(BASE/'EXECUTION_FREEZE.json')['source_sha256'])
    paths=list((ROOT/'scripts/race_v2_inference_soft_context_a_20261029').glob('*.py'))
    paths += [ROOT/'scripts/race_v2_inference_normalization_a_20261029/algorithm.py',ROOT/'scripts/race_v2_inference_normalization_a_20261029/deployment.py',
        ROOT/'scripts/race_v2_inference_innovation_a_20261029/io_helpers.py',ROOT/'scripts/race_v2_inference_innovation_a_20261029/common.py',
        ROOT/'scripts/race_v2_inference_a_20261029/statistics.py',ROOT/'scripts/followup_test270_common.py']
    source.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
    write(R/'PREPARATION_FREEZE.json',dict(at=now(),source_sha256=source,plan_sha256=sha(R/'PLAN.json'),CPU_gate_sha256=sha(R/'CPU_GATE.json'),
        idea_sha256=sha(R/'IDEA.md'),literature_and_history_sha256=sha(R/'HISTORY_AND_LITERATURE.json')))

if __name__=='__main__':main()
