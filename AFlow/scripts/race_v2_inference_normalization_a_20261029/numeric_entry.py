"""Set the historical numeric mode in the actual execution process.

This versioned entry delegates in-process to immutable V1 transfer/final modules;
it does not spawn a fresh Python interpreter that could lose backend settings.
"""
import argparse,os,time
import torch
from scripts.race_v2_inference_normalization_a_20261029.common import *
from scripts.followup_test270_common import numeric_mode

ENTRY_SOURCE=ROOT/'scripts/race_v2_inference_normalization_a_20261029/numeric_entry.py'
MODE_SOURCE=ROOT/'scripts/followup_test270_common.py'
ORIGINAL={
    'transfer':R/'FINAL_SNAPSHOT_TRANSFER_FREEZE.json',
    'final270':BASE/'FINAL270_NORMALIZATION_ENTRY_FREEZE.json'}
V2={
    'transfer':R/'FINAL_SNAPSHOT_TRANSFER_FREEZE_V2.json',
    'final270':BASE/'FINAL270_NORMALIZATION_ENTRY_FREEZE_V2.json'}

def mode_state():
    return dict(cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)

def check_mode():
    state=mode_state()
    assert state==dict(cudnn_deterministic=True,cudnn_benchmark=False,cuda_matmul_allow_tf32=False,cudnn_allow_tf32=True)
    return state

def freeze():
    assert not (STAGE/'FINAL_TEST_SELECTION.json').exists()
    numeric_mode();state=check_mode()
    for kind in ('transfer','final270'):
        assert not V2[kind].exists();old=read(ORIGINAL[kind]);sources=dict(old['source_sha256'])
        sources.update({str(p.relative_to(ROOT)):sha(p) for p in (ENTRY_SOURCE,MODE_SOURCE)})
        write(V2[kind],dict(at=now(),source_sha256=sources,original_freeze_path=str(ORIGINAL[kind]),
            original_freeze_sha256=sha(ORIGINAL[kind]),same_process_numeric_entry=True,
            backend_mode=state,version_reason='Set historical numeric mode before new-final model load and synthetic/patient inference',
            missing_mode_branch_had_not_run=True,original164_validation_unchanged=True,
            normalization_recipe_path=str(R/'NORMALIZATION_DEPLOYMENT_RECIPE.json'),
            normalization_recipe_sha256=sha(R/'NORMALIZATION_DEPLOYMENT_RECIPE.json'),
            no_new_patient_access_or_forward_by_freeze=True))

def verify(kind):
    guard();f=read(V2[kind]);assert sha(f['original_freeze_path'])==f['original_freeze_sha256']
    for p,d in f['source_sha256'].items():assert sha(ROOT/p)==d,p
    assert sha(f['normalization_recipe_path'])==f['normalization_recipe_sha256']
    return f

def main(kind,watch,statistics_only):
    f=verify(kind);numeric_mode();state=check_mode()
    write(R/f'NUMERIC_MODE_RECEIPT_{kind}.json',dict(at=now(),pid=os.getpid(),kind=kind,backend_mode=state,
        V2_freeze_path=str(V2[kind]),V2_freeze_sha256=sha(V2[kind]),numeric_source_sha256=sha(MODE_SOURCE),
        entry_source_sha256=sha(ENTRY_SOURCE),same_process_delegate=True,new_python_child_spawned=False,
        prior_new_model_patient_forwards=0,new270_yet=False))
    if kind=='transfer':
        from scripts.race_v2_inference_normalization_a_20261029 import transfer
        check_mode();transfer.watch() if watch else transfer.execute()
    else:
        from scripts.race_v2_inference_normalization_a_20261029 import final270
        check_mode();final270.main(watch,statistics_only)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--kind',choices=['transfer','final270'])
    p.add_argument('--watch',action='store_true');p.add_argument('--statistics-only',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:
        assert a.kind
        try:main(a.kind,a.watch,a.statistics_only)
        except BaseException as error:
            import traceback
            folder=R/'final_snapshot' if a.kind=='transfer' else BASE/'final270'
            write(folder/f'NUMERIC_ENTRY_FAILURE_{time.time_ns()}.json',dict(at=now(),kind=a.kind,error=repr(error),traceback=traceback.format_exc(),automatic_neural_retry=False));raise
