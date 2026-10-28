"""Full-shape synthetic hardware benchmark, checkpoint and tail-batch checks."""
import argparse
import time
import torch
from monai.losses import DiceCELoss
from scripts.race_seg_20261028.common import *
from scripts.race_seg_20261028.train import setup,make_optimizer,step


def main(arm):
    TMP.mkdir(parents=True,exist_ok=True);results=[]
    for micro in (1,2):
        try:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            model=setup(arm);opt=make_optimizer(model);fn=DiceCELoss(sigmoid=True,lambda_dice=1.,lambda_ce=.1)
            x=torch.randn(8,1,*CANVAS);y=torch.zeros_like(x);y[:,:,48:80,80:100,64:80]=1
            classes=y.byte();classes[:,:,85:100,90:110,65:85]=2
            reflection=torch.eye(4).repeat(8,1,1);reflection[:,0,0]=-1;reflection[:,0,3]=CANVAS[0]-8
            batch={'image':x,'label':y,'classes':classes,'valid':torch.ones_like(x),'reflection':reflection}
            values=[]
            for iteration in range(1,5):
                torch.cuda.synchronize();tick=time.monotonic()
                value=step(model,opt,fn,batch,20,iteration,micro,augment=False)
                torch.cuda.synchronize();values.append((time.monotonic()-tick,value))
            tail={k:v[:4] for k,v in batch.items()}
            step(model,opt,fn,tail,20,6,micro,augment=False)
            assert all(int(s['step'])==5 for s in opt.state.values())
            path=TMP/f'smoke_{arm}_{micro}.pth';torch.save({'model':model.state_dict(),'optimizer':opt.state_dict()},path)
            saved=torch.load(path,map_location='cpu',weights_only=False)
            assert all(torch.equal(v.detach().cpu(),saved['model'][k]) for k,v in model.state_dict().items())
            if arm=='race':
                assert abs(float(model.strength))>0
                assert any(abs(v['strength_grad'])>0 for _,v in values)
            peak=torch.cuda.max_memory_reserved()/2**30
            results.append({'microbatch':micro,'status':'passed' if peak<=38 else 'over_memory_ceiling',
                            'seconds_per_update':sum(v[0] for v in values[1:])/3,'raw_blocks_per_second':24/sum(v[0] for v in values[1:]),
                            'peak_reserved_gib':peak,'tail_updates_and_reload':'passed','metrics':values[-1][1]})
            del model,opt,saved,batch,x,y,classes,tail;path.unlink();torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            results.append({'microbatch':micro,'status':'oom'})
            torch.cuda.empty_cache()
    write(R/f'SMOKE_{arm}.json',{'at':now(),'arm':arm,'device':torch.cuda.get_device_name(),'results':results,'patient_forwards':0})
    assert any(x['status']=='passed' for x in results),'no safe microbatch'

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('arm',choices=ARMS);args=parser.parse_args();main(args.arm)
