"""DDP synthetic resource probe of official ResidualEncoderUNet; not a planned patient experiment."""
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

ROOT = Path('/mlx_devbox/users/songjunyi.666/playground/AFlow')
OUT = ROOT / 'archive/reviews/2026-10-13_resenc_feasibility'


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.manual_seed(20261013)
    dist.init_process_group('nccl')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Same topology family as official ResEnc planner. Patch sizes here are synthetic probes,
    # not M/L plans from patient fingerprints, and the simplified local Dice is not a trainer.
    model = ResidualEncoderUNet(
        input_channels=1, n_stages=6, features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=nn.Conv3d, kernel_sizes=[3]*6, strides=[1,2,2,2,2,2],
        n_blocks_per_stage=[1,3,4,6,6,6], num_classes=2, n_conv_per_stage_decoder=[1]*5,
        conv_bias=True, norm_op=nn.InstanceNorm3d, norm_op_kwargs={'eps':1e-5,'affine':True},
        dropout_op=None, nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace':True},
        deep_supervision=True,
    ).cuda(rank)
    count = sum(p.numel() for p in model.parameters())
    model = DistributedDataParallel(model, device_ids=[rank])
    opt = torch.optim.SGD(model.parameters(), lr=.01, momentum=.99, nesterov=True, weight_decay=3e-5)
    scaler = torch.amp.GradScaler('cuda')
    records = []
    for shape in [(96,128,128), (128,160,128)]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(rank)
        x = torch.randn((1,1,*shape), device=rank)
        y = torch.zeros((1,1,*shape), device=rank)
        y[:,:,24:56,32:72,32:72] = 1
        timings, losses = [], []
        for step in range(40):
            dist.barrier()
            torch.cuda.synchronize(rank)
            start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                outputs = model(x)
                weights = np.array([1/(2**i) for i in range(len(outputs))])
                weights[-1] = 1e-6
                weights /= weights.sum()
                loss = 0
                for w,z in zip(weights,outputs):
                    target = F.interpolate(y, size=z.shape[2:], mode='nearest')[:,0].long()
                    ce = F.cross_entropy(z, target)
                    prob = z.float().softmax(1)[:,1]
                    dice = 1-(2*(prob*target).sum()+1e-5)/(prob.sum()+target.sum()+1e-5)
                    loss = loss + float(w)*(ce+dice)
            if not bool(torch.isfinite(loss)): raise RuntimeError('nonfinite synthetic loss')
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),12.)
            if not bool(torch.isfinite(norm)): raise RuntimeError('nonfinite synthetic gradient')
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize(rank)
            seconds = time.perf_counter()-start
            if step >= 10:
                timings.append(seconds)
                losses.append(float(loss))
        digest = torch.stack([p.detach().float().sum() for p in model.parameters()])
        ref = digest.clone()
        dist.broadcast(ref,src=0)
        if not torch.equal(digest,ref): raise RuntimeError('rank parameter sums differ')
        med = float(np.median(timings))
        records.append({'patch_size':list(shape),'per_gpu_batch':1,'global_batch':2,
            'rank':rank,'parameters':count,'timed_updates':30,'warmup_updates':10,
            'median_update_seconds':med,'p90_update_seconds':float(np.quantile(timings,.9)),
            'model_only_250000_updates_hours':250000*med/3600,
            'peak_allocated_gib':torch.cuda.max_memory_allocated(rank)/2**30,
            'peak_reserved_gib':torch.cuda.max_memory_reserved(rank)/2**30,
            'finite_loss_and_gradients':True,'rank_parameter_sums_exact':True,
            'loss_start':losses[0],'loss_end':losses[-1]})
        del x,y,outputs,loss,z,prob,target,ce,dice
    gathered = [None,None]
    dist.all_gather_object(gathered,records)
    if rank == 0:
        package_root = Path('/tmp/aflow_nnunet_probe_20261013/packages')
        files = [package_root/'dynamic_network_architectures/architectures/unet.py',
            package_root/'nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py',
            package_root/'nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py',Path(__file__)]
        report={'recorded_at':datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'worker':1132660,'status':'synthetic_probe_complete','records_by_rank':gathered,
            'versions':{n:importlib.metadata.version(n) for n in ['torch','nnunetv2','dynamic-network-architectures']},
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'patient_pixel_access':False,'patient_model_forwards':0,'formal_training_updates':0,
            'limitations':'Synthetic patch shapes, not dataset-planned M/L. Simplified local Dice+CE/deep supervision; excludes augmentation, loading, validation, checkpointing, preprocessing. Short warm steady-state timing extrapolation is not a measured full training duration. No claim of full nnU-Net numerical/smoke equivalence.'}
        (OUT/'PROBE_V3.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
    # Drop DDP and optimizer owners before explicit communicator destruction.
    del model, opt, scaler, digest, ref
    import gc
    gc.collect()
    torch.cuda.synchronize(rank)
    dist.barrier(device_ids=[rank])
    (OUT/f'EXIT_V3_RANK{rank}.json').write_text(json.dumps({'stage':'before_destroy'}))
    # Match repository's existing post-success bounded cleanup pattern.
    # This branch is reached only after all measurements/checks and the final barrier.
    import sys, threading
    def cleanup_timeout():
        (OUT/f'EXIT_V3_RANK{rank}.json').write_text(json.dumps({'stage':'post_success_cleanup_timeout','metrics_committed':True}))
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    timer = threading.Timer(20, cleanup_timeout)
    timer.daemon = True
    timer.start()
    dist.destroy_process_group()
    (OUT/f'EXIT_V3_RANK{rank}.json').write_text(json.dumps({'stage':'destroy_returned'}))
    timer.cancel()


if __name__ == '__main__':
    main()
