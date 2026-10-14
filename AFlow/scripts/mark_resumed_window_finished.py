"""After final report only, replace the timestamped progress snapshot with closure."""
from datetime import datetime,timezone
import json
from scripts.structured_background_protocol import ROOT,sha,checked,write_new


def main():
    rec=ROOT/'archive/research_records/2026-09-10_resumed_eight_hour'
    result=json.loads((rec/'RESULT.json').read_text());proof=json.loads((rec/'VERIFICATION_RECEIPT.json').read_text())
    checked(rec/'RESULT.json',proof['result_sha256'])
    assert result['elapsed_seconds']>=8*3600 and result['status']=='eight_hour_window_completed_leaderboard_target_not_achieved'
    assert not result['leaderboard_surpassed'] and result['new_test_forwards']==0
    old=rec/'STATUS.json'
    with (rec/'STATUS_BEFORE_FINAL.json').open('xb') as f:f.write(old.read_bytes())
    closed={'schema':'aflow.resumed_window.closed','status':result['status'],'created_utc':datetime.now(timezone.utc).isoformat(),'window_start_utc':result['started_utc'],'window_end_utc':result['deadline_utc'],'finished_utc':result['finished_utc'],'elapsed_seconds':result['elapsed_seconds'],'remaining_seconds':0,'all_six_training_runs_completed':True,'all_candidate_development_and_mask_verification_completed':True,'fixed_recipes_closed':True,'new_test_forwards':0,'old_confirmation_outer_external_new_performance':False,'leaderboard_surpassed':False,'result_path':str((rec/'RESULT.json').relative_to(ROOT)),'result_sha256':sha(rec/'RESULT.json'),'diagnostics':result['diagnostics'],'do_not_restart_closed_entries':True}
    old.write_text(json.dumps(closed,indent=2,ensure_ascii=False)+'\n')
    write_new(rec/'WINDOW_CLOSED.json',{'status':'closed','created_utc':datetime.now(timezone.utc).isoformat(),'result_sha256':sha(rec/'RESULT.json'),'status_sha256':sha(old),'closure_code_sha256':sha(__file__)})
    print('final closed state written',flush=True)


if __name__=='__main__':main()
