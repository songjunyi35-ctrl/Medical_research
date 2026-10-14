"""Bounded same-batch AMP replay; never skip samples or update on invalid gradients."""
import random
import numpy as np
import torch
import torch.distributed as dist


def capture_rng(device):
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state(device)


def restore_rng(state, device):
    random.setstate(state[0]); np.random.set_state(state[1])
    torch.set_rng_state(state[2]); torch.cuda.set_rng_state(state[3], device)


def step(ddp, opt, scaler, image, target):
    device = image.device
    rng = capture_rng(device)
    buffers = {n: b.detach().clone() for n, b in ddp.named_buffers()}
    for attempt in range(9):
        if attempt:
            restore_rng(rng, device)
            with torch.no_grad():
                for n, b in ddp.named_buffers(): b.copy_(buffers[n])
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            loss, gray = ddp(image, target)
        valid = (torch.isfinite(loss) & torch.isfinite(gray)).to(torch.int32)
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
        if not bool(valid): raise RuntimeError('nonfinite loss/gray: hard stop')
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grads = [p.grad for p in ddp.parameters() if p.grad is not None]
        if not grads: raise RuntimeError('missing gradients')
        norms = torch.stack([g.detach().float().norm() for g in grads])
        norm = norms.double().norm()
        valid = torch.isfinite(norm).to(torch.int32)
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
        if not bool(valid):
            if attempt == 8 or scaler.get_scale() <= 1.:
                raise RuntimeError('nonfinite gradients persist after bounded scale backoff')
            old_scale = scaler.get_scale()
            scaler.update(new_scale=old_scale * scaler.get_backoff_factor())
            state = scaler.state_dict(); state['_growth_tracker'] = 0; scaler.load_state_dict(state)
            if dist.get_rank() == 0:
                print(__import__('json').dumps({'event':'amp_same_batch_replay','attempt':attempt+1,'scale_before':old_scale,'scale_after':scaler.get_scale(),'optimizer_updated':False}), flush=True)
            continue
        positive = (norm > 0).to(torch.int32)
        dist.all_reduce(positive, op=dist.ReduceOp.MIN)
        if not bool(positive): raise RuntimeError('zero gradient: hard stop')
        scaler.step(opt); scaler.update()
        summary = torch.stack([loss.detach().double(), gray.detach().double(), norm])
        dist.all_reduce(summary); summary /= dist.get_world_size()
        result = dict(zip(['loss','gray','grad_norm'], summary.cpu().tolist()))
        return {**result, 'amp_retries':attempt, 'amp_scale':scaler.get_scale()}
    raise AssertionError('unreachable')
