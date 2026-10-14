"""Local preparation: unit tests, environment receipt, and FREEZE.json before anything runs on the worker."""
import sys,os,json,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.segresnet_native_res_common import *
PY=str(ROOT.parent/'sammed3d-venv/bin/python')
FROZEN=['scripts/segresnet_native_res_common.py','scripts/build_segresnet_native_res_cache.py','scripts/train_segresnet_native_res.py','scripts/evaluate_plain972_e193_native_validation.py','scripts/watch_segresnet_native_res.py','scripts/run_segresnet_native_res.py','tests/test_segresnet_native_res.py',
        'sammed3d/t1ce_wt_capg3d.py','sammed3d/t1ce_binary.py','sammed3d/t1ce_wt_training.py','sammed3d/t1ce_wt_cached.py','scripts/train_t1ce_wt_capg3d.py','scripts/train_t1ce_wt_plain_long_schedule.py','scripts/build_t1ce_wt_cache.py',
        'archive/research_records/2026-09-18_segresnet_plain972_native_res/IDEA.md','archive/research_records/2026-09-18_segresnet_plain972_native_res/ENVIRONMENT.json','archive/research_records/2026-09-18_segresnet_plain972_native_res/UNIT_TESTS.json',
        'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train/index.json','workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_ema_20260819/validation108/index.json',
        'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl','data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/validation_t1ce_wt.jsonl','data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/sealed_test/test_cases_t1ce.jsonl',
        'archive/research_records/2026-09-17_segresnet_plain972_e193_test270/checkpoint_epoch193.pth','workspace/BraTS/segresnet_t1ce_wt/plain972_baseline_20260917/validation_e193.json']


def main():
    if (R/'PIPELINE_START.json').exists() and not (R/'PIPELINE_FAILURE.json').exists():raise RuntimeError('pipeline running or completed; do not re-freeze')
    if (R/'TRAIN_PREACCESS.json').exists() or OUT.exists():raise RuntimeError('training already touched patient data; do not re-freeze')
    if sha(BASELINE_CKPT)!=BASELINE_CKPT_SHA:raise RuntimeError('baseline checkpoint hash changed')
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}
    p=subprocess.run([PY,'-m','pytest','tests/test_segresnet_native_res.py','-q'],cwd=ROOT,env=env,capture_output=True,text=True)
    passed=p.returncode==0;n=sum(1 for line in p.stdout.splitlines() if ' passed' in line)
    write(R/'UNIT_TESTS.json',{'utc':now(),'status':'passed' if passed else 'failed','command':'pytest tests/test_segresnet_native_res.py -q','summary':p.stdout.strip().splitlines()[-1] if p.stdout.strip() else '','scope':['canvas covers every canonical crop extent and is divisible by 8','pad/unpad round trip and oversize rejection','full972 coverage, equal ranks, even tail for every batch candidate','metrics exclude canvas padding','test-accessible or wrong-canvas cache rejected before payload','augmentation empty-foreground fallback','learning-rate endpoints'],'model_patient_inference':False})
    if not passed:print(p.stdout[-4000:],p.stderr[-4000:]);raise RuntimeError('unit tests failed')
    import torch,monai,inspect
    from monai.networks.nets import segresnet
    from monai.losses import dice
    write(R/'ENVIRONMENT.json',{'utc':now(),'torch':torch.__version__,'monai':monai.__version__,'SegResNet_source_path':inspect.getsourcefile(segresnet),'SegResNet_source_sha256':sha(inspect.getsourcefile(segresnet)),'DiceCELoss_source_sha256':sha(inspect.getsourcefile(dice)),'note':'cudnn.benchmark and TF32 enabled for throughput (not bitwise deterministic)'})
    files={f:sha(ROOT/f) for f in FROZEN}
    write(R/'FREEZE.json',{'utc':now(),'status':'frozen_before_worker_cache_build_smoke_and_training','files':files,'single_arm':'pure SegResNet4700897, native 1mm crop in fixed 160x208x160 canvas (no resize)','initialization':'random seed20260819 (same seed as the baseline); no learned checkpoints','target_epochs':EPOCHS,'train_scans_per_epoch':TRAIN_SCANS,'per_gpu_batch_candidates':list(BATCH_CANDIDATES),'memory_ceiling_bytes':MEMORY_CEILING_BYTES,'batch_rule':'largest candidate that fits on both ranks in the synthetic probe; recorded in BATCH_DECISION.json before training','world_size':2,'loss':'Dice +0.1 BCE; squared_pred false','optimizer':'AdamW2e-4 wd1e-5 clip5 fused','precision':'bfloat16 autocast; TF32; cudnn.benchmark','schedule':'constant100, cosine101..200 to1e-5','augmentation':'flips/shift8/intensity scale-offset-noise (unchanged)','selection':'fixed validation108 scan Dice on the native crop grid, every epoch, earlier tie','comparison_row':'frozen e193 baseline resampled to the same native crop grid (BASELINE_E193_NATIVE_VALIDATION.json)','test_authorized':False,'prior_test_informed_development':True})
    print(json.dumps({'unit_tests':'passed','frozen_files':len(files)}))


if __name__=='__main__':main()
