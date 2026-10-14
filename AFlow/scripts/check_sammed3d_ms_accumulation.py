"""Two-rank toy regression: sample-weighted tail, overflow replay and one update."""
import sys,os,random,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from scripts.train_sammed3d_mumford_shah_972 import setup,run_window,successful_exit,R,write,now

class Tiny(torch.nn.Module):
    def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor(.25,device='cuda'));self.draws=[]
    def forward(self,x,y):
        self.draws.append((random.random(),float(np.random.rand()),float(torch.rand(())),float(torch.rand((),device='cuda'))))
        loss=((self.w*x-y)**2).mean()
        return loss,torch.stack([loss.detach()]*6)
class CountingSGD(torch.optim.SGD):
    def __init__(self,p):super().__init__(p,lr=.01);self.calls=0
    def step(self,*args,**kwargs):self.calls+=1;return super().step(*args,**kwargs)

if __name__=='__main__':
    rank,device=setup();model=Tiny();ddp=DDP(model,device_ids=[device.index]);opt=CountingSGD(model.parameters());scaler=torch.amp.GradScaler('cuda',init_scale=128.)
    data=[{'image':torch.tensor([float(rank*3+i+1)]),'label':torch.zeros(1)} for i in range(3)]
    called=[0]
    def hook(g):
        called[0]+=1
        return g*float('inf') if rank==1 and called[0]==1 else g
    handle=model.w.register_hook(hook)
    result=run_window(ddp,opt,scaler,data,[[0,1],[2]],device,821+rank,workers=0);handle.remove()
    expected=.25-.01*(2*.25*sum(i*i for i in range(1,7))/6)
    assert abs(float(model.w)-expected)<1e-6,(float(model.w),expected)
    assert opt.calls==1 and result['amp_window_retries']==1 and result['samples']==6
    assert model.draws[:2]==model.draws[2:]
    other=model.w.detach().clone();dist.broadcast(other,0);assert torch.equal(other,model.w)
    if rank==0:
        write(R/'ACCUMULATION_TEST.json',{'utc':now(),'status':'passed','ranks':2,'expected_parameter':expected,'actual_parameter':float(model.w),'optimizer_calls':opt.calls,'sample_weighted_tail_correct':True,'one_rank_overflow_replays_whole_window':True,'random_draws_replayed':True,'result':result,'patient_access':False})
        print('accumulation and overflow replay checks passed',flush=True)
    successful_exit()
