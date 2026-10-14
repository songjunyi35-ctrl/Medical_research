"""Keep the batch; adapt only the engineering memory reserve to A10080GB."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'archive/research_records/2026-09-29_segresnet_native_multiscale_resume50'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    a = ROOT / 'scripts/train_native_multiscale_resume50_20260929.py'
    b = ROOT / 'scripts/supervise_native_multiscale_resume50_20260929.py'
    source = a.read_text()
    old = "    if peak > 36 * 2**30:\n        raise RuntimeError('synthetic peak exceeds36GiB ceiling')"
    new = "    write(R / f'GPU_MEMORY_RANK{rank}.json', {'peak_reserved_bytes': peak, 'worker': 1114913, 'engineering_ceiling_gib': 64, 'global_batch': 8, 'patient_pixels_access': False})\n    if peak > 64 * 2**30:\n        raise RuntimeError('synthetic peak exceeds64GiB ceiling on A10080GB')"
    if source.count(old) != 1:
        raise RuntimeError('unexpected source memory gate')
    generated = {
        'scripts/train_native_multiscale_resume50_20260929_v2.py': source.replace(old, new),
        'scripts/supervise_native_multiscale_resume50_20260929_v2.py': b.read_text().replace(
            'scripts/train_native_multiscale_resume50_20260929.py',
            'scripts/train_native_multiscale_resume50_20260929_v2.py'),
    }
    for name, text in generated.items():
        p = ROOT / name
        if p.exists():
            raise FileExistsError(name)
        ast.parse(text)
        p.write_text(text)
    receipt = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'stage': 'synthetic smoke; no new patient model forward',
               'observed': 'full/tail synthetic updates finished, then inherited36GiB reserve gate rejected on80GB device; no CUDA OOM recorded',
               'old_v1_trainer_sha256': sha(a), 'old_v1_supervisor_sha256': sha(b),
               'new_sources_sha256': {p: sha(ROOT / p) for p in generated},
               'old_engineering_ceiling_gib': 36, 'new_engineering_ceiling_gib': 64,
               'physical_gpu_gib': 80, 'batch_change': False, 'global_batch': 8,
               'model_loss_optimizer_schedule_change': False,
               'resume_checkpoint_unchanged': sha(ROOT / 'workspace/BraTS/segresnet_t1ce_wt/native_multiscale_resume50_20260929/latest.pth') == '93a969ad592849e2d057111c0111837726cb44b9bd126668aa6567faa6dba8a6'}
    (R / 'ENGINEERING_AMENDMENT_003.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
