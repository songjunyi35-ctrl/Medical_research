"""Matched fit-only training; no development loader or metrics in this entrypoint."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss
from sammed3d.structured_background_aux import StructuredBackgroundLocator, conditional_rc_loss, export_plain_state
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from scripts import structured_background_protocol as p
from scripts.train_t1ce_wt_capg3d import BrainCropDataset, augment_batch, dice_per_case, write_json_atomic
from scripts.train_t1ce_wt_gcsam_inner import AuditedWeightedSampler, learning_rate_for_epoch
from scripts.build_t1ce_wt_native_volume_artifact import native_sampling_weights


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_sha(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class FitData(BrainCropDataset):
    def __init__(self, fit):
        super().__init__(p.CACHE, expected_split='train', case_ids=[r['case_id'] for r in fit])
        receipt = json.loads((p.RECORD / 'AUX_CACHE_RECEIPT.json').read_text())
        p.checked(Path(receipt['rc_file']), receipt['rc_sha256'])
        self.rc_path = receipt['rc_file']
        self._rc = None

    def __getitem__(self, slot):
        out = super().__getitem__(slot)
        if self._rc is None:
            self._rc = np.memmap(self.rc_path, dtype=np.uint8, mode='r', shape=(684,128,128,128))
        out['label'] = torch.cat((out['label'], torch.from_numpy(np.array(self._rc[slot], dtype=np.float32))[None]))
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=p.SEEDS, required=True)
    parser.add_argument('--arm', choices=p.ARMS, required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    p.check_freeze()
    preaccess = json.loads((p.RECORD / 'TRAIN_PREACCESS.json').read_text())
    assert preaccess['status'] == 'frozen_before_formal_training'
    p.checked(p.RECORD / 'IMPLEMENTATION_FREEZE.json', preaccess['implementation_freeze_sha256'])
    p.checked(p.RECORD / 'AUX_CACHE_RECEIPT.json', preaccess['aux_cache_receipt_sha256'])
    p.checked(p.RECORD / 'SMOKE_RECEIPT.json', preaccess['smoke_sha256'])
    fit, _ = p.check_roles()
    p.checked(p.CACHE / 'index.json', p.INDEX_SHA)
    p.checked(p.NATIVE, p.NATIVE_SHA)
    output = p.WORK / f'seed{args.seed}_{args.arm}'
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    data = FitData(fit)
    native = json.loads(p.NATIVE.read_text())['native_wt_voxels']
    weights = native_sampling_weights(data.case_ids, native)
    sampler = AuditedWeightedSampler(weights, 972, args.seed)
    loader = DataLoader(data, batch_size=8, sampler=sampler, drop_last=True, num_workers=2, pin_memory=True, persistent_workers=True, generator=torch.Generator().manual_seed(args.seed))
    assert len(loader) == 121
    model = StructuredBackgroundLocator().to(device)
    initial = state_sha(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5, foreach=None, fused=None)
    loss_fn = DiceCELoss(sigmoid=True, squared_pred=False, lambda_dice=1., lambda_ce=.1)
    coefficient = .1 if args.arm == 'candidate' else 0.
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    p.write_new(output / 'STARTED.json', {'created_utc':datetime.now(timezone.utc).isoformat(), 'seed':args.seed,'arm':args.arm,'initial_state_sha256':initial,'fit_sha256':p.FIT_SHA,'train_preaccess_sha256':p.sha(p.RECORD/'TRAIN_PREACCESS.json'),'teacher_checkpoints':[], 'development_metrics_accessed':False})
    history = []
    for epoch in range(1,169):
        model.train()
        lr = learning_rate_for_epoch(epoch, 2e-4, 1e-5)
        for group in optimizer.param_groups:
            group['lr'] = lr
        total_losses = []; primary_losses = []; aux_losses = []; train_dices = []
        aug_hash = hashlib.sha256()
        first = None
        for step, batch in enumerate(loader):
            image = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            image, labels = augment_batch(image, labels, max_shift=8)
            aug_hash.update(torch.cuda.get_rng_state(device).cpu().numpy().tobytes())
            if step == 0:
                first = {'image_sha256':tensor_sha(image),'labels_sha256':tensor_sha(labels)}
            wt, rc = labels[:, :1], labels[:, 1:2]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, aux_logits = model(image)
                primary = loss_fn(logits, wt)
            auxiliary = conditional_rc_loss(aux_logits, rc, wt)
            loss = primary + coefficient * auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total_losses.append(float(loss.detach())); primary_losses.append(float(primary.detach())); aux_losses.append(float(auxiliary.detach()))
            train_dices.extend(dice_per_case(logits.detach().float(), wt).cpu().tolist())
        item = {'epoch':epoch, 'lr':lr, 'loss':float(np.mean(total_losses)), 'primary_loss':float(np.mean(primary_losses)), 'aux_loss':float(np.mean(aux_losses)), 'train_dice':float(np.mean(train_dices)), 'steps':len(total_losses), 'sampler_sha256':hashlib.sha256(np.asarray(sampler.last_indices[:968], dtype=np.int64).tobytes()).hexdigest(), 'augmentation_rng_sha256':aug_hash.hexdigest(),'first_batch':first,'elapsed_seconds':time.monotonic()-start}
        history.append(item)
        write_json_atomic(output / 'progress.json', {'history':history,'development_metrics_accessed':False})
        print(f'{args.seed}/{args.arm} e{epoch} primary={item["primary_loss"]:.5f} aux={item["aux_loss"]:.5f} fit_dice={item["train_dice"]:.5f} seconds={item["elapsed_seconds"]:.1f}', flush=True)
        if item['elapsed_seconds'] > 6*3600 or torch.cuda.max_memory_allocated(device) > 24*1024**3:
            raise RuntimeError('formal time/memory stop gate')
        if epoch % 32 == 0:
            temporary = output / 'recovery.tmp'
            torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'sampler_rng':sampler.generator.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(device),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'role':'fit_only_recovery_not_selected'}, temporary)
            temporary.replace(output / 'recovery_latest.pth')
    model.eval()
    plain_state = export_plain_state(model)
    plain = AutomaticPromptLocator(center_aware=False, init_filters=16, dropout=.1).to(device).eval()
    plain.load_state_dict(plain_state, strict=True)
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        original = model(image[:1])[0].float()
        exported = plain(image[:1]).segmentation_logits.float()
    export_error = float((original-exported).abs().max())
    if export_error != 0:
        raise RuntimeError(f'export output mismatch: {export_error}')
    checkpoint = output / 'fixed_epoch168.pth'
    torch.save({'schema':'aflow.structured_background_aux.checkpoint','model_state_dict':plain_state,'epoch':168,'seed':args.seed,'arm':args.arm,'architecture':{'init_filters':16,'dropout':.1,'center_aware':False},'fit_sha256':p.FIT_SHA,'development_metrics_accessed':False}, checkpoint)
    reload = torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state_sha(reload['model_state_dict']) == state_sha(plain_state)
    p.write_new(output / 'TRAIN_DONE.json', {'status':'complete','seed':args.seed,'arm':args.arm,'fixed_epochs':168,'fit_sha256':p.FIT_SHA,'dev_sha256':p.DEV_SHA,'initial_state_sha256':initial,'sampling_weight_sha256':tensor_sha(weights),'history':history,'checkpoint':{'path':str(checkpoint),'sha256':p.sha(checkpoint)},'export_max_abs_error':export_error,'elapsed_seconds':time.monotonic()-start,'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(device),'all_gates_passed':True,'teacher_checkpoints':[],'access':{'fit':True,'development_metrics':False,'test':False,'confirmation':False,'outer':False,'external':False},'runner_sha256':p.sha(Path(__file__))})
    print('TRAIN_DONE', str(checkpoint), flush=True)


if __name__ == '__main__':
    main()
