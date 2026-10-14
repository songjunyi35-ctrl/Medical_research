"""Bind documents, implementation, cache, smoke, and authorization before formal."""
import json
from pathlib import Path
from datetime import datetime,timezone
from scripts import structured_background_protocol as p


def main():
    p.check_roles()
    smoke=json.loads((p.RECORD/'SMOKE_RECEIPT.json').read_text())
    assert smoke['status']=='passed' and all(smoke['gates'].values()) and smoke['synthetic_only']
    p.checked(p.ROOT/'sammed3d/structured_background_aux.py',smoke['model_sha256'])
    cache=json.loads((p.RECORD/'AUX_CACHE_RECEIPT.json').read_text())
    assert cache['status']=='passed' and cache['all_wt_byte_exact'] and cache['fit_sha256']==p.FIT_SHA
    p.checked(Path(cache['rc_file']),cache['rc_sha256'])
    files=[p.RECORD/'IDEA.md',p.RECORD/'PREREG.md',p.ROLES/'AUTHORIZATION.md',p.ROLES/'SPLIT.json',p.FIT,p.DEV]
    files += [p.ROOT/name for name in ['scripts/structured_background_protocol.py','sammed3d/structured_background_aux.py','scripts/train_structured_background_aux.py','scripts/evaluate_structured_background_aux.py','scripts/smoke_structured_background_aux.py','scripts/build_structured_background_aux_cache.py','scripts/train_t1ce_wt_capg3d.py','scripts/train_t1ce_wt_gcsam_inner.py','scripts/build_t1ce_wt_native_volume_artifact.py','sammed3d/t1ce_wt_capg3d.py']]
    p.write_new(p.RECORD/'IMPLEMENTATION_FREEZE.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'files':{str(path.relative_to(p.ROOT)):p.sha(path) for path in files},'stage':'before_formal_training','candidate_coefficient':.1,'seeds':list(p.SEEDS),'epochs':168})
    p.write_new(p.RECORD/'TRAIN_PREACCESS.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'status':'frozen_before_formal_training','implementation_freeze_sha256':p.sha(p.RECORD/'IMPLEMENTATION_FREEZE.json'),'aux_cache_receipt_sha256':p.sha(p.RECORD/'AUX_CACHE_RECEIPT.json'),'smoke_sha256':p.sha(p.RECORD/'SMOKE_RECEIPT.json'),'dev_performance_accessed':False,'teacher_checkpoints':[]})
    print('formal training preaccess frozen')


if __name__=='__main__':
    main()
