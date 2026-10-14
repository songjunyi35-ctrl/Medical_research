"""Freeze evaluation and independent archival implementations before training ends."""
from datetime import datetime,timezone
from scripts import weak_elastic_protocol as p


def main():
    p.check_freeze()
    assert not (p.RECORD/'DEV_PREACCESS.json').exists() and not (p.RECORD/'RAW_DEVELOPMENT.json').exists()
    files=['scripts/evaluate_weak_elastic.py','scripts/finalize_weak_elastic.py','scripts/evaluate_structured_background_aux.py','scripts/finalize_structured_background_aux.py','scripts/audit_weak_elastic_training.py','scripts/audit_subject_sampler_replay.py','tests/test_weak_elastic_evaluation.py','scripts/freeze_weak_elastic_development_code.py']
    p.write_new(p.RECORD/'DEVELOPMENT_IMPLEMENTATION_FREEZE.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'status':'frozen_before_weak_elastic_development_and_before_all_training_complete','files':{name:p.sha(p.ROOT/name) for name in files},'candidate_outputs_accessed':False,'independent_cleanup_study_closed_does_not_change_this_recipe':True,'partial_access_markers':'DEV_ACTIVE_BATCH.json + DEV_PROGRESS.json; interrupted partials cannot select checkpoints'})
    print('weak-elastic evaluation/archival code frozen prospectively',flush=True)


if __name__=='__main__':main()
