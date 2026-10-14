"""Synthetic GPU memory check only: no patient data or persisted model updates."""
import sys,os,time,json,hashlib,datetime,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from scripts import train_segresnet_plain972_no_n4 as base
R=Path(__file__).parent

def write(name,d):
    p=R/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)

def main():
    rank,device=base.setup()
    free,total=torch.cuda.mem_get_info(device)
    okay=torch.tensor(int(free>=20*2**30),device=device);dist.all_reduce(okay,op=dist.ReduceOp.MIN)
    if not bool(okay):raise RuntimeError('less than20GiB free on a card; do not compete with other jobs')
    model=base.model_for(device);ddp=DDP(base.Forward(model),device_ids=[device.index]);opt=base.optimizer_for(model);criterion=base.loss_for()
    base.seed_all(base.SEED+rank);before=next(model.parameters()).detach().clone()
    torch.cuda.reset_peak_memory_stats(device);start=time.monotonic();steps=[]
    for i in range(3):
        x=torch.randn(6,1,128,128,128,device=device);y=torch.zeros_like(x);y[:,:,36:92,36:92,36:92]=1;x+=1.5*y
        steps.append(base.step(ddp,opt,criterion,x,y))
        del x,y
    torch.cuda.synchronize();change=float((next(model.parameters())-before).abs().max());assert change>0
    row={'rank':rank,'gpu':torch.cuda.get_device_name(device),'per_gpu_batch':6,'global_batch':12,'shape':[6,1,128,128,128],'precision':'bfloat16 autocast','optimizer':'AdamW with original Dice+0.1BCE, augmentation, backward, clip5 and optimizer.step','steps':steps,'parameter_change':change,'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),'gpu_total_bytes':total,'free_before_bytes':free,'elapsed_seconds':time.monotonic()-start}
    results=[None,None];dist.all_gather_object(results,row)
    if rank==0:
        write('RESULT.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'passed_no_oom','ranks':results,'patient_pixels_access':False,'formal_training_resumed':False,'model_checkpoints_modified':False,'new_accuracy_result':False,'scope':'3 full forward/backward/optimizer steps on synthetic data; same128cubed SegResNet and bf16 recipe; not a long-run guarantee','source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'scripts/train_segresnet_plain972_no_n4.py',ROOT/'scripts/segresnet_plain972_no_n4_common.py']}})
        print(json.dumps(results),flush=True)
    dist.barrier();torch.cuda.synchronize();sys.stdout.flush();os._exit(0)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write('FAILURE_rank'+os.environ.get('RANK','none')+'.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
