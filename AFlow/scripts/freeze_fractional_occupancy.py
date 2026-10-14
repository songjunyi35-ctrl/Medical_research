"""Admit only after primary closes, cache/smoke pass, and full168 fits budget."""
from datetime import datetime,timezone
import json
from pathlib import Path
import time
from scripts import fractional_occupancy_protocol as p
from scripts.smoke_fractional_occupancy import require_primary_closed


def main():
    if (p.RECORD/'TRAIN_PREACCESS.json').exists():raise FileExistsError('already admitted')
    require_primary_closed();p.source.check_freeze();p.check_roles()
    cache=json.loads((p.RECORD/'CACHE_RECEIPT.json').read_text());assert cache['status']=='passed' and cache['all_hard_cache_byte_exact']
    verification=json.loads((p.RECORD/'CACHE_VERIFICATION.json').read_text());assert verification['status']=='passed'
    p.checked(p.RECORD/'CACHE_RECEIPT.json',verification['cache_receipt_sha256'])
    dataset=json.loads((p.RECORD/'DATASET_CPU_RECEIPT.json').read_text());assert dataset['status']=='passed'
    for name,digest in dataset['files'].items():p.checked(p.ROOT/name,digest)
    assert cache['max_relative_volume_error_float64']<=1e-10 and cache['max_relative_volume_error_float16']<=.001
    p.checked(Path(cache['target_path']),cache['target_sha256']);p.checked(p.SOFT_CACHE/'index.json',cache['index_sha256'])
    smoke=json.loads((p.RECORD/'SMOKE_RECEIPT.json').read_text());assert smoke['status']=='passed' and all(smoke['gates'].values())
    p.checked(p.RECORD/'CACHE_RECEIPT.json',smoke['cache_receipt_sha256'])
    for name,digest in smoke['files'].items():p.checked(p.ROOT/name,digest)
    cache_pre=json.loads((p.RECORD/'CACHE_PREACCESS.json').read_text())
    for name,digest in cache_pre['files'].items():p.checked(p.ROOT/name,digest)
    controls={};times=[]
    for seed in p.SEEDS:
        path=p.source.WORK/f'seed{seed}_control/TRAIN_DONE.json';d=json.loads(path.read_text())
        assert d['all_gates_passed'] and d['fixed_epochs']==168 and d['fit_sha256']==p.FIT_SHA and not d['teacher_checkpoints']
        p.checked(d['checkpoint']['path'],d['checkpoint']['sha256']);times.append(d['elapsed_seconds'])
        controls[str(seed)]={'report_path':str(path),'report_sha256':p.sha(path),'checkpoint':d['checkpoint']}
    estimate=max(times)*1.1+1200;deadline=datetime(2026,9,10,11,28,19,tzinfo=timezone.utc).timestamp()
    if time.time()+estimate>deadline:raise RuntimeError(f'full168 pair budget insufficient: need {estimate:.1f}s, remaining {deadline-time.time():.1f}s')
    with (p.RECORD/'AUX_CACHE_RECEIPT.json').open('xb') as f:f.write((p.source.RECORD/'AUX_CACHE_RECEIPT.json').read_bytes())
    aux=json.loads((p.RECORD/'AUX_CACHE_RECEIPT.json').read_text());p.checked(p.CACHE/'images.f32',aux['source_images_sha256']);p.checked(p.CACHE/'labels.u8',aux['source_labels_sha256'])
    files=[p.RECORD/n for n in ['IDEA.md','PREREG.md','CACHE_PREACCESS.json','CACHE_RECEIPT.json','CACHE_INDEX.json','CACHE_VERIFICATION.json','CACHE_RESULT.md','DATASET_CPU_RECEIPT.json','SMOKE_RECEIPT.json']]
    files.append(p.ROOT/'scripts/verify_fractional_occupancy_cache.py')
    files += [p.ROOT/n for n in ['scripts/fractional_occupancy_protocol.py','scripts/train_fractional_occupancy.py','scripts/smoke_fractional_occupancy.py','scripts/freeze_fractional_occupancy.py','scripts/run_fractional_occupancy_training.py','sammed3d/fractional_occupancy.py','tests/test_fractional_occupancy.py','tests/test_fractional_occupancy_training.py','scripts/train_structured_background_aux.py','sammed3d/structured_background_aux.py','scripts/train_t1ce_wt_capg3d.py','scripts/train_t1ce_wt_gcsam_inner.py','scripts/build_t1ce_wt_native_volume_artifact.py']]
    primary=p.ROOT/'archive/research_records/2026-09-10_weak_elastic'
    files += [primary/n for n in ['RESULT.json','VERIFICATION_RECEIPT.json']]
    files += [p.FIT,p.DEV,p.NATIVE,p.CACHE/'index.json']
    p.write_new(p.RECORD/'IMPLEMENTATION_FREEZE.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'files':{str(path.relative_to(p.ROOT)):p.sha(path) for path in files},'controls':controls,'budget_estimate_seconds':estimate,'fixed_test_seed_if_eligible':2026090711})
    p.write_new(p.RECORD/'TRAIN_PREACCESS.json',{'status':'frozen_before_formal_training','created_utc':datetime.now(timezone.utc).isoformat(),'implementation_freeze_sha256':p.sha(p.RECORD/'IMPLEMENTATION_FREEZE.json'),'aux_cache_receipt_sha256':p.sha(p.RECORD/'AUX_CACHE_RECEIPT.json'),'smoke_sha256':p.sha(p.RECORD/'SMOKE_RECEIPT.json'),'cache_receipt_sha256':p.sha(p.RECORD/'CACHE_RECEIPT.json'),'new_candidate_development_access':False,'primary_closed_before_start':True})
    print('fractional occupancy fixed168 two-seed pair admitted',flush=True)


if __name__=='__main__':main()
