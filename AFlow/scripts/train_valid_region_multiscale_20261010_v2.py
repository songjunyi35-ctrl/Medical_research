"""One native SegResNet arm with training-only half/quarter scale supervision."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import threading
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.segresnet_pretrained_native_common import (
    NativeCanvasDataset, batches_for_epoch, deterministic_validation, fast_cudnn,
    model_for, safe_augment, seed_all, learning_rate_for_epoch,
)
from scripts.train_segresnet_pretrained_native import evaluation
from scripts.valid_region_multiscale_20261010_common_v2 import ValidRegionDataset, augment_with_mask, valid_multiscale_loss, valid_dice_bce

R = ROOT / 'archive/research_records/2026-10-10_segresnet_valid_region_supervision'
OUT = ROOT / 'workspace/BraTS/segresnet_t1ce_wt/valid_region_20261010'
CACHE = Path('/tmp/aflow_valid_region_20261010')
PRETRAIN = ROOT / 'workspace/BraTS/segresnet_t1ce_wt/two_ideas_20260921/pretrain/final.pth'
PRETRAIN_SHA = '3726eac66918b7afd32ae93500a554534d27528bc86a0fe9dd2a626ce5ac6b76'
CANVAS = (160, 208, 160)
SEED = 20260819
EPOCHS = 200


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def disk():
    import shutil
    if shutil.disk_usage(ROOT).free < 20 * 2**30:
        raise RuntimeError('persistent free below20GiB')
    if shutil.disk_usage('/tmp').free < 100 * 2**30:
        raise RuntimeError('worker scratch free below100GiB')
    persisted = sum(p.stat().st_size for tree in [R, OUT] if tree.exists() for p in tree.rglob('*') if p.is_file())
    if persisted > 8 * 2**30:
        raise RuntimeError('new persistent material exceeds8GiB')


def save(path, data):
    disk()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(data, temp)
    os.replace(temp, path)


class SupervisedBackbone(torch.nn.Module):
    """Original pretrained model plus discardable training-only heads."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.aux_quarter = torch.nn.Conv3d(64, 1, 1)
        self.aux_half = torch.nn.Conv3d(32, 1, 1)
        self._features = {}
        self._hooks = [self.backbone.network.up_layers[0].register_forward_hook(self._capture('quarter')),
                       self.backbone.network.up_layers[1].register_forward_hook(self._capture('half'))]

    def _capture(self, key):
        def hook(_module, _input, output):
            self._features[key] = output
        return hook

    def forward(self, x):
        self._features.clear()
        main = self.backbone(x).segmentation_logits
        if set(self._features) != {'quarter', 'half'}:
            raise RuntimeError('missing SegResNet decoder features')
        return main, self.aux_half(self._features['half']), self.aux_quarter(self._features['quarter'])


def loss_fn(logits, target, mask, criterion):
    return valid_multiscale_loss(logits, target, mask, criterion)


def setup():
    rank = int(os.environ['RANK'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=60))
    if dist.get_world_size() != 2:
        raise RuntimeError('must use both assigned A100 GPUs')
    seed_all(SEED)
    fast_cudnn()
    return rank, device


def committed_exit(marker):
    """Leave NCCL promptly only after a durable rank-0 success marker exists."""
    dist.barrier()
    if not Path(marker).exists():
        raise RuntimeError('refuse successful exit without committed marker')
    def force():
        os._exit(0)
    timer = threading.Timer(20, force)
    timer.daemon = True
    timer.start()
    dist.destroy_process_group()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def model(device):
    backbone = model_for(device)
    system = SupervisedBackbone(backbone).to(device)
    return system


def training_step(ddp, opt, criterion, image, label, mask, distributed=True):
    if not torch.isfinite(image).all() or not torch.isfinite(label).all() or not bool((label.flatten(1).sum(1) > 0).all()):
        raise RuntimeError('bad source batch')
    image, label, mask, fallbacks = augment_with_mask(image, label, mask)
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        outputs = ddp(image)
        loss, parts = loss_fn(outputs, label, mask, criterion)
    if not torch.isfinite(loss):
        raise RuntimeError('nonfinite multiscale loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 5, error_if_nonfinite=True)
    if not bool(norm > 0):
        raise RuntimeError('zero gradient')
    opt.step()
    vals = torch.tensor([float(loss.detach()), *(float(p.detach()) for p in parts), float(norm), float(fallbacks)], device=image.device, dtype=torch.float64)
    if distributed:
        dist.all_reduce(vals)
        vals[:5] /= 2
    return dict(zip(['loss', 'main_loss', 'half_loss', 'quarter_loss', 'grad_norm', 'augmentation_fallbacks'], vals.cpu().tolist()))


def source():
    if sha(PRETRAIN) != PRETRAIN_SHA:
        raise RuntimeError('public pretrain150 checkpoint identity mismatch')
    state = torch.load(PRETRAIN, map_location='cpu', weights_only=False)
    if state.get('epoch') != 150:
        raise RuntimeError('pretrain checkpoint epoch mismatch')
    return state['model_state_dict']


def verify():
    freeze = read(R / 'EXECUTION_FREEZE.json')
    for relative, expected in freeze['source_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError('frozen source changed: ' + relative)
    data = read(R / 'DATA_AUDIT.json')
    if data['status'] != 'passed' or not data['historical_pixel_payload_exact']:
        raise RuntimeError('native cache not verified')
    for cache in data['caches']:
        path = CACHE / cache['role']
        if sha(path / 'images.f32') != cache['images_sha256'] or sha(path / 'labels.u8') != cache['labels_sha256']:
            raise RuntimeError(cache['role'] + ' cache payload changed')
    disk()
    return data


def smoke():
    rank, device = setup()
    frozen = read(R / 'SMOKE_FREEZE_V2.json')
    for path, digest in frozen['source_sha256'].items():
        if sha(ROOT / path) != digest:
            raise RuntimeError('synthetic source changed: ' + path)
    if read(R / 'CPU_CHECKS_V2.json')['status'] != 'passed':
        raise RuntimeError('CPU checks not passed')
    # Check bf16 numerical definition on both GPUs before any patient forward.
    c = DiceCELoss(sigmoid=True, squared_pred=False, lambda_dice=1., lambda_ce=.1)
    z = torch.linspace(-3, 3, 2*8*12*16, device=device).reshape(2,1,8,12,16).to(torch.bfloat16).requires_grad_(True)
    target = (torch.arange(z.numel(), device=device).reshape(z.shape) % 5 == 0).float()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        old = c(z, target)
        actual = valid_dice_bce(z, target, torch.ones_like(target), c)
    ga = torch.autograd.grad(old, z, retain_graph=True)[0]
    gb = torch.autograd.grad(actual, z)[0]
    equivalence = {'value_error': float(abs(old-actual)), 'gradient_error': float((ga-gb).abs().max())}
    if equivalence['value_error'] > 2e-6 or equivalence['gradient_error'] > 2e-6:
        raise RuntimeError('bf16 all-ones equivalence failed')
    del z,target,ga,gb,old,actual
    system = model(device)
    system.backbone.load_state_dict(source(), strict=True)
    ddp = DDP(system, device_ids=[device.index])
    opt = torch.optim.AdamW(ddp.parameters(), lr=2e-4, weight_decay=1e-5, fused=True)
    criterion = DiceCELoss(sigmoid=True, squared_pred=False, lambda_dice=1., lambda_ce=.1)
    before = next(system.parameters()).detach().clone()
    torch.manual_seed(SEED + rank)
    torch.cuda.reset_peak_memory_stats(device)
    examples = []
    for n in [4, 2]:
        x = torch.randn((n, 1, *CANVAS), device=device)
        y = torch.zeros_like(x)
        y[:, :, 40:90, 70:120, 40:90] = 1
        mask = torch.zeros_like(x, dtype=torch.uint8)
        mask[:, :, 9:150, 13:193, 11:151] = 1
        result = training_step(ddp, opt, criterion, x, y, mask)
        examples.append({'per_gpu_batch': n, 'losses': result})
        del x,y,mask
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_reserved(device)
    if peak > 38 * 2**30:
        raise RuntimeError('synthetic peak exceeds38GiB ceiling')
    difference = torch.zeros((), device=device)
    for parameter in system.parameters():
        other = parameter.detach().clone()
        dist.broadcast(other, 0)
        difference = torch.maximum(difference, (other-parameter).abs().max())
    changed = float((next(system.parameters())-before).abs().max())
    counters = {int(v['step']) for v in opt.state.values()}
    if float(difference) != 0 or changed <= 0 or counters != {2}:
        raise RuntimeError('DDP parameter sync or optimizer counters failed')
    row = {'rank': rank, 'peak_reserved_bytes': peak,
           'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
           'bf16_all_ones_equivalence': equivalence, 'steps': examples,
           'max_parameter_difference_between_ranks': float(difference),
           'parameter_change': changed, 'optimizer_steps': sorted(counters)}
    gathered = [None, None]; dist.all_gather_object(gathered, row)
    write(R / f'GPU_MEMORY_RANK{rank}.json', row)
    if rank == 0:
        stripped = model_for(torch.device('cpu'))
        stripped.load_state_dict({k:v.cpu() for k,v in system.backbone.state_dict().items()}, strict=True)
        write(R / 'GPU_SMOKE.json', {'utc': now(), 'status': 'passed', 'worker': 1127648,
              'global_batch': 8, 'tail_global_batch': 4, 'ddp_checks': gathered,
              'peak_reserved_bytes': max(x['peak_reserved_bytes'] for x in gathered),
              'memory_ceiling_bytes': 38*2**30, 'strict_deployment_backbone_reload': True,
              'patient_pixels_access': False, 'model_source_sha256': PRETRAIN_SHA})
    committed_exit(R / 'GPU_SMOKE.json')


def train():
    rank, device = setup()
    data = verify()
    if read(R / 'GPU_SMOKE.json')['status'] != 'passed':
        raise RuntimeError('GPU smoke not passed')
    train_ids = json.loads((CACHE / 'train/index.json').read_text())['case_ids']
    val_ids = json.loads((CACHE / 'validation/index.json').read_text())['case_ids']
    train_data = ValidRegionDataset(CACHE / 'train', expected_split='train', case_ids=train_ids)
    val_data = NativeCanvasDataset(CACHE / 'validation', expected_split='validation', case_ids=val_ids)
    if len(train_data) != 972 or len(val_data) != 108:
        raise RuntimeError('wrong cohort length')
    val_indices = list(range(108))[rank::2]
    system = model(device)
    previous = OUT / 'latest.pth'
    resume = previous.exists()
    if not resume:
        system.backbone.load_state_dict(source(), strict=True)
    ddp = DDP(system, device_ids=[device.index])
    opt = torch.optim.AdamW(ddp.parameters(), lr=2e-4, weight_decay=1e-5, fused=True)
    criterion = DiceCELoss(sigmoid=True, squared_pred=False, lambda_dice=1., lambda_ce=.1)
    start, best, best_epoch = 1, -1., 0
    if resume:
        checkpoint = torch.load(previous, map_location='cpu', weights_only=False)
        system.load_state_dict(checkpoint['system_state_dict'], strict=True)
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start, best, best_epoch = checkpoint['epoch'] + 1, checkpoint['best_scan'], checkpoint['best_epoch']
    if rank == 0:
        if OUT.exists() and any(OUT.iterdir()) and not resume:
            raise RuntimeError('nonempty output without existing latest checkpoint')
        OUT.mkdir(parents=True, exist_ok=True)
        if not resume:
            write(R / 'TRAIN_PREACCESS.json', {'utc': now(), 'train': 972, 'validation': 108,
                  'source_checkpoint_sha256': PRETRAIN_SHA, 'new_test_model_forward': False,
                  'test_tuned_development': True, 'freeze_sha256': sha(R / 'EXECUTION_FREEZE.json'),
                  'research_change': 'geometry-derived valid-region main/auxiliary losses',
                  'no_epoch50_score_budget_stop': True})
    dist.barrier()
    started = time.monotonic()
    stop_reason = 'epoch budget completed' if start > EPOCHS else None
    torch.cuda.reset_peak_memory_stats()
    last_complete = start - 1
    for epoch in range(start, EPOCHS + 1):
        if stop_reason:
            break
        disk()
        seed_all(SEED + epoch * 10 + rank)
        system.train()
        lr = learning_rate_for_epoch(epoch, base_lr=2e-4, min_lr=1e-5, constant_epochs=100, total_epochs=EPOCHS)
        for group in opt.param_groups:
            group['lr'] = lr
        loader = DataLoader(train_data, batch_sampler=list(batches_for_epoch(epoch, rank, 4)),
                            num_workers=6, pin_memory=True, prefetch_factor=2,
                            generator=torch.Generator().manual_seed(SEED + epoch * 100 + rank))
        seen, metrics = [], []
        for step, batch in enumerate(loader, 1):
            value = training_step(ddp, opt, criterion,
                                  batch['image'].to(device, non_blocking=True),
                                  batch['label'].to(device, non_blocking=True),
                                  batch['valid_mask'].to(device, non_blocking=True))
            seen.extend(batch['case_id'])
            metrics.append((len(batch['case_id']), value))
            if rank == 0 and (step == 1 or step % 20 == 0 or step == 122):
                write(OUT / 'STATUS.json', {'utc': now(), 'state': 'training', 'epoch': epoch,
                      'step': step, 'last_complete_epoch': epoch - 1, 'best_epoch': best_epoch,
                      'best_scan': best, 'test_pixels_access': False})
                print(json.dumps({'utc': now(), 'epoch': epoch, 'step': step, 'loss': value['loss']}), flush=True)
        collected = [None, None]
        dist.all_gather_object(collected, seen)
        ids = collected[0] + collected[1]
        if len(ids) != 972 or len(set(ids)) != 972 or len(metrics) != 122 or set(ids) != set(train_ids):
            raise RuntimeError('972 unique training coverage failed')
        if rank == 0 and epoch == 1:
            write(R / 'VALIDATION_PREACCESS.json', {'utc': now(), 'cohort': 'fixed108/50',
                  'native_grid': True, 'selection': 'maxScan, earlier tie', 'test_pixels_access': False})
        result = evaluation(system.backbone, val_data, val_indices, device, rank, epoch)
        if rank == 0:
            report = OUT / f'validation_e{epoch:03d}.json'
            write(report, result)
            score = float(result['locator_dsc_mean'])
            if score > best:
                best = score
                best_epoch = epoch
                save(OUT / 'best.pth', {'model_state_dict': system.backbone.state_dict(),
                     'epoch': epoch, 'source_pretrain_sha256': PRETRAIN_SHA,
                     'training_only_multiscale_heads': True, 'selection': 'fixed108 native Scan Dice'})
            save(previous, {'system_state_dict': system.state_dict(), 'optimizer_state_dict': opt.state_dict(),
                 'epoch': epoch, 'best_scan': best, 'best_epoch': best_epoch,
                 'completed_updates': epoch * 122, 'source_pretrain_sha256': PRETRAIN_SHA})
            entry = {'utc': now(), 'epoch': epoch, 'validation_scan': score,
                     'validation_subject': result['locator_subject_dsc_mean'],
                     'validation_point': result['automatic_point_hit_rate'],
                     'best_epoch': best_epoch, 'best_scan': best, 'updates': epoch * 122,
                     'train_loss': sum(n * m['loss'] for n, m in metrics) / 486,
                     'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
            with (OUT / 'TRAIN.jsonl').open('a') as stream:
                stream.write(json.dumps(entry) + '\n')
            print(json.dumps(entry), flush=True)
            if epoch == EPOCHS:
                stop_reason = 'epoch budget completed'
            write(OUT / 'STATUS.json', {'utc': now(), 'state': 'training' if not stop_reason else 'stopped',
                  'last_complete_epoch': epoch, 'best_epoch': best_epoch, 'best_scan': best,
                  'completed_updates': epoch * 122, 'reason': stop_reason,
                  'test_pixels_access': False})
            write(R / 'PIPELINE_STATUS.json', {'utc': now(), 'state': 'training' if not stop_reason else 'training_ended',
                  'running': not bool(stop_reason), 'last_complete_epoch': epoch,
                  'best_epoch': best_epoch, 'best_scan': best, 'worker': 1127648,
                  'new_test_model_forward': False})
        marker = [stop_reason]
        dist.broadcast_object_list(marker, src=0)
        stop_reason = marker[0]
        dist.barrier()
        last_complete = epoch
        if stop_reason:
            break
    # Re-evaluate the selected and final snapshots in the same deterministic
    # numeric mode. A resume after a post-training engineering failure enters
    # here without training an extra epoch.
    check_rows = []
    for filename in ['best.pth', 'latest.pth']:
        snapshot = torch.load(OUT / filename, map_location='cpu', weights_only=False)
        if filename == 'best.pth':
            stored_epoch = int(snapshot['epoch'])
            system.backbone.load_state_dict(snapshot['model_state_dict'], strict=True)
        else:
            stored_epoch = int(snapshot['epoch'])
            system.load_state_dict(snapshot['system_state_dict'], strict=True)
        actual = evaluation(system.backbone, val_data, val_indices, device, rank, stored_epoch)
        if rank == 0:
            expected = read(OUT / f'validation_e{stored_epoch:03d}.json')
            if [r['case_id'] for r in actual['per_case']] != [r['case_id'] for r in expected['per_case']]:
                raise RuntimeError('reload case order mismatch')
            error = max(abs(a['locator_dsc'] - b['locator_dsc']) for a, b in zip(actual['per_case'], expected['per_case']))
            points = sum(a['automatic_point'] != b['automatic_point'] for a, b in zip(actual['per_case'], expected['per_case']))
            if error > 1e-6 or points:
                raise RuntimeError('checkpoint reload differs from frozen validation')
            check_rows.append({'checkpoint': filename, 'epoch': stored_epoch,
                  'checkpoint_sha256': sha(OUT / filename), 'max_scan_dice_error': error,
                  'point_coordinates_differing': points})
    if rank == 0:
        write(R / 'RELOAD_CHECK.json', {'utc': now(), 'status': 'passed', 'checks': check_rows,
              'test_pixels_access': False})
        write(R / 'TRAIN_STOP.json', {'utc': now(), 'reason': stop_reason,
              'last_complete_epoch': last_complete, 'best_epoch': best_epoch,
              'best_scan': best, 'completed_updates': last_complete * 122,
              'new_test_model_forward': False})
    committed_exit(R / 'TRAIN_STOP.json')


if __name__ == '__main__':
    try:
        {'smoke': smoke, 'train': train}[sys.argv[1]]()
    except BaseException:
        write(R / f'FAILURE_{sys.argv[1]}_rank{os.environ.get("RANK", "none")}_{int(time.time())}.json',
              {'utc': now(), 'traceback': traceback.format_exc()})
        raise
