"""Resource and actual CUDA checks only; no patient pixels or model weights."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo

ROOT = Path('/mlx_devbox/users/songjunyi.666/playground/AFlow')
R = ROOT / 'archive/reviews/2026-10-10_worker1127648_preflight'


def main():
    report = {'checked_at': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'worker': 1127648, 'timezone': 'Asia/Shanghai', 'status': 'checking',
              'new_patient_pixel_access': False, 'patient_model_forwards': 0,
              'training_epochs': 0, 'optimizer_updates': 0,
              'nvidia_uvm_exists': Path('/dev/nvidia-uvm').exists(),
              'cpu_count': os.cpu_count(),
              'project_free_bytes': shutil.disk_usage(ROOT).free,
              'scratch_free_bytes': shutil.disk_usage('/tmp').free}
    try:
        smi = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version',
                              '--format=csv,noheader'], capture_output=True, text=True, check=True)
        report['nvidia_smi'] = smi.stdout.strip().splitlines()
        import torch
        report.update(torch_version=torch.__version__, torch_cuda_version=torch.version.cuda,
                      cuda_is_available=torch.cuda.is_available(), device_count=torch.cuda.device_count())
        if not report['cuda_is_available'] or report['device_count'] != 2:
            raise RuntimeError('expected two usable CUDA GPUs')
        torch.set_num_threads(1)
        results = []
        for i in range(2):
            device = torch.device(f'cuda:{i}')
            properties = torch.cuda.get_device_properties(device)
            if 'A100' not in properties.name or not 38 * 2**30 <= properties.total_memory <= 45 * 2**30:
                raise RuntimeError('expected A10040GB per card')
            torch.cuda.reset_peak_memory_stats(device)
            with torch.inference_mode():
                x = torch.eye(256, device=device)
                if not torch.equal(x @ x, x):
                    raise RuntimeError('float32 CUDA matrix arithmetic failed')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    y = x @ x
                if y.dtype != torch.bfloat16 or not torch.equal(y.float(), x):
                    raise RuntimeError('bfloat16 CUDA matrix arithmetic failed')
                torch.cuda.synchronize(device)
                results.append({'index': i, 'name': properties.name,
                                'total_memory_bytes': properties.total_memory,
                                'fp32_and_bf16_arithmetic_passed': True,
                                'peak_allocated_bytes': torch.cuda.max_memory_allocated(device)})
                del x, y
            torch.cuda.empty_cache()
        report['gpus'] = results
        if report['project_free_bytes'] < 20 * 2**30:
            raise RuntimeError('persistent free space below20GiB')
        if report['scratch_free_bytes'] < 40 * 2**30:
            raise RuntimeError('scratch free space below40GiB')
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
    (R / 'RESULT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
