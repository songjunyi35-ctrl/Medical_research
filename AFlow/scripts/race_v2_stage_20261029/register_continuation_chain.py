"""Freeze the final finite108 pool before the single/pair270 evaluation."""
import datetime
import subprocess
import sys

from .governance import ROOT, RECORD, atomic_json, sha
from .select_final_continuation import A, B, IA, IB, NA, NB, SC, EDGE, APPEARANCE, POLICY, stage_guard, verify_freeze


def hashes(paths):
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def main():
    stage_guard()
    assert not (RECORD/'FINAL_TEST_SELECTION.json').exists()
    for path in (A/'FINAL270_SOFT_CONTEXT_ENTRY_FREEZE.json', B/'FINAL270_APPEARANCE_ENTRY_FREEZE.json'):
        verify_freeze(path)
    assert not POLICY.exists()
    directory = ROOT/'scripts/race_v2_stage_20261029'
    evidence = [RECORD/'STATISTICS_FREEZE.json',
        A/'SELECTED_RECIPE.json', A/'RANKING.json', B/'RESULT.json',
        IA/'DEPLOYMENT_RECIPE.json', IA/'RESULT.json', IB/'SELECTED_RECIPE.json', IB/'RESULT.json',
        NA/'NORMALIZATION_DEPLOYMENT_RECIPE.json', NA/'RESULT.json', NA/'PLAN.json',
        NB/'DEPLOYMENT_RECIPE.json', NB/'RESULT.json', NB/'PLAN.json',
        SC/'SOFT_CONTEXT_DEPLOYMENT_RECIPE.json', SC/'RESULT.json', SC/'PLAN.json',
        EDGE/'DEPLOYMENT_RECIPE.json', EDGE/'INPUT_DEPLOYMENT_RECIPE.json', EDGE/'RESULT.json', EDGE/'PLAN.json',
        APPEARANCE/'DEPLOYMENT_RECIPE.json', APPEARANCE/'RESULT.json', APPEARANCE/'PLAN.json']
    atomic_json(POLICY, dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=hashes([directory/name for name in ('select_final_continuation.py','select_final.py','audit_validation.py','governance.py')]),
        evidence_sha256=hashes(evidence),
        policy='After whole original80 DONE and all fixed final-best108 transfers: highest108Scan of up to8 single and10 pair options; earlier options win exact ties; exactly one single and one pureSegResNet pair',
        single_families=['equal8','local dependency','seed healthy normalization','seed soft context'],
        pair_families=['bounded global fusion','paired covariance','contralateral image evidence','guided w32 input','image appearance unary'],
        base_old_and_final_new_weight_pools=True, best_epoch0_skips_new_model_predictions=True,
        transferred_new_recipes_each_family=1, no_new_family_or_grid_after_this_freeze=True,
        models_per_deployment_max=2, forwards_per_deployment_max=16, final_candidates_max=2,
        seed_conditioned_single_forwards=8, guided_w32_pair_forwards=16,
        oracle_GT_diagnostics_are_not_candidates_or_deployable_results=True,
        policy_adapted_after_reused108_results=True, inherited_test_tuned_history=True,
        independent_confirmation=False, new270_access=False, original_training_unchanged=True))
    POLICY.chmod(0o444)
    publication = RECORD/'FINAL_PUBLICATION_CONTINUATION_FREEZE.json'
    assert not publication.exists()
    test = subprocess.run([sys.executable,'-m','scripts.race_v2_stage_20261029.test_published_raw'],
        cwd=ROOT, check=True, capture_output=True, text=True)
    names = ['final_report_continuation.py','final_report_watch_continuation.py','published_raw.py',
        'test_published_raw.py','publish_continuation.py','publish_innovation_v2.py','publish_innovation.py',
        'publish_snapshot_v2.py','audit_validation.py','governance.py']
    atomic_json(publication, dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=hashes([directory/name for name in names]), previous_versions_preserved=True,
        masks_audited=540, exact_receipt_canonical_wrapper_binding=True,
        actual_seed_transformed_and_guided_new_forwards_counted=True,
        selected_weight_vs_training_budget_reported_separately=True,
        source_archive_python_and_text_only_publication=True,
        receipt_binding_regressions=dict(returncode=test.returncode,stderr=test.stderr),
        new270_access=False, original_training_unchanged=True))
    publication.chmod(0o444)


if __name__ == '__main__':
    main()
