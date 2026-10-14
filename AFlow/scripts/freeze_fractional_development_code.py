"""Prospective evaluation/archival code receipt while candidate training is ongoing."""
from datetime import datetime,timezone
from scripts import fractional_occupancy_protocol as p


def main():
    p.check_freeze()
    assert not (p.RECORD/'DEV_PREACCESS.json').exists() and not (p.RECORD/'RAW_DEVELOPMENT.json').exists()
    files=['scripts/evaluate_fractional_occupancy.py','scripts/finalize_fractional_occupancy.py','scripts/audit_fractional_occupancy_training.py','scripts/audit_subject_sampler_replay.py','scripts/evaluate_structured_background_aux.py','scripts/finalize_structured_background_aux.py','tests/test_fractional_occupancy_evaluation.py','scripts/freeze_fractional_development_code.py']
    p.write_new(p.RECORD/'DEVELOPMENT_IMPLEMENTATION_FREEZE.json',{'status':'frozen_before_candidate_development_scores','created_utc':datetime.now(timezone.utc).isoformat(),'files':{name:p.sha(p.ROOT/name) for name in files},'new_candidate_development_access':False,'fixed_bootstrap_seeds':[2026091081,2026091082],'fixed_test_seed_if_eligible':2026090711})
    print('fractional target evaluation/archive code frozen before candidate readout',flush=True)


if __name__=='__main__':main()
