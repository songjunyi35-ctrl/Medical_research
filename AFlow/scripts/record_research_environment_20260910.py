"""Selected public software/hardware metadata only, not environment secrets."""
from datetime import datetime,timezone
import importlib.metadata
import platform
import subprocess
import sys
import torch
from scripts import weak_elastic_protocol as p


def main():
    packages={name:importlib.metadata.version(name) for name in ['torch','monai','numpy','scipy','nibabel','SimpleITK']}
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total','--format=csv,noheader'],text=True).strip().splitlines()
    out=p.RECORD/'ENVIRONMENT_20260910.json'
    p.write_new(out,{'created_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':packages,'torch_cuda_build':torch.version.cuda,'cudnn_version':torch.backends.cudnn.version(),'gpu_metadata':gpu,'source_engine_sha256':p.sha(p.ROOT/'scripts/train_structured_background_aux.py'),'requested_engine_settings_from_frozen_source':{'cudnn_benchmark':False,'cudnn_deterministic':True,'torch_num_threads':1,'autocast_dtype':'bfloat16','batch_size':8},'scope':'version_snapshot_not_full_binary_environment_lock; metadata only, no checkpoint or patient pixels loaded','historical_controls_retrained_on_current_hardware':False,'historical_vs_current_memory_sku':'historical A10080GB controls reused; current A10040GB candidates','current_hardware_baseline_inference_parity_evidence':'independent cleanup study: both native control dev180 per-case counts/points exactly reproduced','whole_historical_control_training_trajectory_bitwise_reproduction_claimed':False})
    print({'path':str(out),'packages':packages,'gpus':gpu},flush=True)


if __name__=='__main__':main()
