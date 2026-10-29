"""Freeze replacement inference selection/publication, never training."""
import datetime

from .governance import ROOT, RECORD, atomic_json, sha
from .select_final_innovation import IA, IB, A, B, TRAIN, POLICY, stage_guard


def hashes(paths):
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def main():
    stage_guard()
    assert not (RECORD / 'FINAL_TEST_SELECTION.json').exists()
    common = ROOT / 'scripts/race_v2_stage_20261029'
    source_names = ['select_final_innovation.py', 'select_final.py', 'audit_validation.py', 'governance.py']
    sources = [common / name for name in source_names]
    evidence = [IA / 'DEPLOYMENT_RECIPE.json', IB / 'SELECTED_RECIPE.json',
                IA / 'RESULT.json', IB / 'RESULT.json', IA / 'PLAN.json', IB / 'PLAN.json',
                A / 'SELECTED_RECIPE.json', A / 'RANKING.json', B / 'RESULT.json',
                RECORD / 'STATISTICS_FREEZE.json', RECORD / 'INFERENCE_INNOVATION108_RESULT.json']
    assert not POLICY.exists()
    atomic_json(POLICY, dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=hashes(sources), evidence_sha256=hashes(evidence),
        policy='After full original80 training and final108 transfer, highest108 Scan among up to4 single and4 pair options; old options win exact ties; exactly one single and one pair',
        single_options=['race164 equal8', 'race164 fixed local dependency innovation',
                        'final training best equal8', 'final training best one fixed innovation transfer'],
        fusion_options=['race164/w32 old global best', 'race164/w32 one paired covariance innovation',
                        'final training best/one frozen partner old bounded global best',
                        'final training best/same partner one fixed covariance transfer'],
        fixed_recipe_transfer_per_new_model=dict(single=1, fusion=1), no_research_grid_on_new_model=True,
        best_epoch0='skip duplicate new model predictions and use old model options only',
        models_per_deployment_max=2, forwards_per_deployment_max=16, final_candidates_max=2,
        policy_adapted_after_initial_and_innovation108_results=True,
        before_final_training_result=not (TRAIN / 'RESULT.json').exists(),
        inherited_test_tuned_history=True, independent_confirmation=False,
        new270_access_during_policy_freeze=False, training_source_recipe_budget_or_process_changed=False))
    POLICY.chmod(0o444)
    publication = RECORD / 'FINAL_PUBLICATION_INNOVATION_FREEZE.json'
    assert not publication.exists()
    sources = [common / name for name in ('final_report_innovation.py', 'final_report_watch_innovation.py',
               'publish_innovation.py', 'publish_snapshot_v2.py', 'audit_validation.py', 'governance.py')]
    atomic_json(publication, dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=hashes(sources), prior_versions_preserved=True,
        changes=['include bounded innovation108 negative/uncertain evidence and exact selected algorithms',
                 'publish new inference source and text through existing suffix whitelist',
                 'preserve meaningful leading porcelain status spaces for publication retry'],
        worker_mask_audit_required=540, single_and_pure_SegResNet_pair_reported_separately=True,
        no_training_change=True, new270_access=False))
    publication.chmod(0o444)


if __name__ == '__main__':
    main()
