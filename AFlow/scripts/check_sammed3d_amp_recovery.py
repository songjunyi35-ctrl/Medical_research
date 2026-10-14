"""Two-rank CUDA regression: one-rank overflow, RNG replay, one update, hard stops."""
import sys, os, json, random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from scripts.sammed3d_amp_recovery import step
from scripts.sammed3d_graymean_single972 import write, ROOT

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.p=torch.nn.Parameter(torch.ones((),device='cuda')); self.draws=[]; self.mode='finite'
    def forward(self,x,y):
        self.draws.append((random.random(),float(np.random.rand()),float(torch.rand(())),float(torch.rand((),device='cuda'))))
        value=(self.p*x-y).square().mean()
        if self.mode=='nan' and dist.get_rank()==1: value=value*float('nan')
        if self.mode=='zero': value=value*0
        return value,value.detach()*0

if __name__=='__main__':
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl')
    model=Tiny();ddp=DDP(model,device_ids=[rank]);opt=torch.optim.AdamW(model.parameters(),lr=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=128.)
    calls=[0]
    def hook(g):
        calls[0]+=1
        return g*float('inf') if rank==1 and calls[0]==1 else g
    handle=model.p.register_hook(hook)
    x=torch.ones(2,device='cuda')*2;y=torch.zeros_like(x)
    result=step(ddp,opt,scaler,x,y);handle.remove()
    assert result['amp_retries']==1 and result['amp_scale']==64
    assert model.draws[0]==model.draws[1],model.draws
    assert opt.state[model.p]['step'].item()==1
    other=model.p.detach().clone();dist.broadcast(other,0);assert torch.equal(other,model.p)
    for mode in ['nan','zero']:
        model.mode=mode;before=model.p.detach().clone()
        try:step(ddp,opt,scaler,x,y)
        except RuntimeError as exc:assert mode in str(exc) or (mode=='nan' and 'nonfinite loss' in str(exc))
        else:raise AssertionError('expected hard stop')
        assert torch.equal(before,model.p) and opt.state[model.p]['step'].item()==1
    if rank==0:write(ROOT/'archive/research_records/2026-09-17_sammed3d_resume139/AMP_TEST.json',{'status':'passed','ranks':2,'one_rank_overflow':True,'rng_replay_exact':True,'updates':1,'hard_stops':['nonfinite_loss','zero_gradient'],'result':result})
    dist.destroy_process_group()
