"""Admit the prospective sampler study after the preceding workflow is complete."""
from datetime import datetime,timezone
import json
from pathlib import Path
from scripts import subject_balanced_protocol as p


def main():
    if (p.RECORD/'TRAIN_PREACCESS.json').exists():raise FileExistsError('already frozen')
    p.source.check_freeze();p.check_roles()
    design=json.loads((p.RECORD/'DESIGN_FREEZE.json').read_text())
    p.checked(p.RECORD/'IDEA.md',design['idea_sha256']);p.checked(p.RECORD/'PREREG.md',design['prereg_sha256'])
    chain=json.loads((p.SOURCE_RECORD/'CHAIN_DONE.json').read_text());assert chain['status']=='completed'
    verification=json.loads((p.SOURCE_RECORD/'VERIFICATION_RECEIPT.json').read_text());assert verification['status']=='passed'
    p.checked(p.SOURCE_RECORD/'RESULT.json',chain['result_sha256'])
    p.checked(p.SOURCE_RECORD/'RAW_DEVELOPMENT.json',verification['raw_sha256'])
    smoke=json.loads((p.RECORD/'SMOKE_RECEIPT.json').read_text());assert smoke['status']=='passed' and all(smoke['gates'].values())
    p.checked(p.ROOT/'sammed3d/structured_background_aux.py',smoke['model_sha256'])
    controls={}
    for seed in p.SEEDS:
        path=p.SOURCE_WORK/f'seed{seed}_control/TRAIN_DONE.json';d=json.loads(path.read_text())
        assert d['seed']==seed and d['arm']=='control' and d['all_gates_passed'] and d['fixed_epochs']==168
        assert d['fit_sha256']==p.FIT_SHA and not d['teacher_checkpoints'] and not d['access']['development_metrics']
        p.checked(Path(d['checkpoint']['path']),d['checkpoint']['sha256'])
        controls[str(seed)]={'train_report_path':str(path),'train_report_sha256':p.sha(path),'checkpoint':d['checkpoint'],'initial_state_sha256':d['initial_state_sha256'],'native_weight_sha256':d['sampling_weight_sha256']}
    # Byte-preserved evidence alias required by the shared training entry; no cache copied.
    with (p.RECORD/'AUX_CACHE_RECEIPT.json').open('xb') as f:f.write((p.SOURCE_RECORD/'AUX_CACHE_RECEIPT.json').read_bytes())
    files=[p.RECORD/'IDEA.md',p.RECORD/'PREREG.md',p.RECORD/'DESIGN_FREEZE.json',p.ROLES/'AUTHORIZATION.md',p.ROLES/'SPLIT.json',p.FIT,p.DEV]
    files += [p.ROOT/name for name in ['scripts/subject_balanced_protocol.py','sammed3d/subject_balanced_sampling.py','scripts/train_subject_balanced_sampling.py','scripts/evaluate_subject_balanced_sampling.py','scripts/smoke_subject_balanced_sampling.py','scripts/train_structured_background_aux.py','sammed3d/structured_background_aux.py']]
    p.write_new(p.RECORD/'IMPLEMENTATION_FREEZE.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'files':{str(path.relative_to(p.ROOT)):p.sha(path) for path in files},'only_factor':'subject probability normalization','source_controls':controls})
    p.write_new(p.RECORD/'TRAIN_PREACCESS.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'status':'frozen_before_formal_training','implementation_freeze_sha256':p.sha(p.RECORD/'IMPLEMENTATION_FREEZE.json'),'aux_cache_receipt_sha256':p.sha(p.RECORD/'AUX_CACHE_RECEIPT.json'),'smoke_sha256':p.sha(p.RECORD/'SMOKE_RECEIPT.json'),'source_controls':controls,'source_raw_development_sha256':verification['raw_sha256'],'new_candidate_development_access':False,'proposal_preceded_source_development':True})
    print('subject-balanced formal admission frozen')


if __name__=='__main__':main()
