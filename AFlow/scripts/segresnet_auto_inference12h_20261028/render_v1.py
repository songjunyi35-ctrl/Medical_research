"""Frozen model, exact inverse geometry, online view statistics; labels never read."""
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from scripts.segresnet_auto_inference12h_20261028.common import *


def blur(x,sigma):
    if not sigma:return x
    radius=max(1,math.ceil(3*sigma))
    axis=torch.arange(-radius,radius+1,device=x.device,dtype=torch.float32)
    kernel=torch.exp(-.5*(axis/sigma).square());kernel/=kernel.sum()
    original=x
    for dim in range(3):
        shape=[1,1,1];shape[dim]=len(kernel)
        padding=[0,0,0];padding[dim]=radius
        x=F.conv3d(x,kernel.reshape(1,1,*shape),padding=tuple(padding))
    return torch.where(original!=0,x,original)


def forward_view(model,image,spec,flip):
    original_shape=tuple(image.shape[-3:])
    x=image
    if spec.get('gaussian_sigma',0):x=blur(x,spec['gaussian_sigma'])
    if spec.get('gain',1.)!=1.:x=x*spec['gain']
    if spec.get('offset',0.):x=torch.where(image!=0,x+spec['offset'],x)
    if spec.get('gamma',1.)!=1.:x=x.sign()*x.abs().pow(spec['gamma'])
    if spec.get('noise_std',0):x=x+torch.randn_like(x)*spec['noise_std']*(image!=0)
    extra=tuple(spec.get('padding_extra',[0,0,0]));phase=tuple(spec.get('phase',[0,0,0]))
    offsets=tuple(e//2+p for e,p in zip(extra,phase))
    if any(extra):
        assert all(e%8==0 and 0<=o<=e for e,o in zip(extra,offsets))
        shape=tuple(a+b for a,b in zip(original_shape,extra))
        padded=torch.zeros((len(x),1,*shape),device=x.device,dtype=x.dtype)
        sl=tuple(slice(o,o+s) for o,s in zip(offsets,original_shape))
        padded[(slice(None),slice(None))+sl]=x;x=padded
    before_resize=tuple(x.shape[-3:])
    target=tuple(spec.get('resize_shape',before_resize))
    if target!=before_resize:
        assert all(s>0 and s%8==0 for s in target)
        x=F.interpolate(x,size=target,mode='trilinear',align_corners=False)
    permutation=tuple(spec.get('permutation',[0,1,2]))
    assert sorted(permutation)==[0,1,2]
    x=x.permute(0,1,*[d+2 for d in permutation]).contiguous()
    dims=tuple(d+2 for d in range(3) if flip & (1<<d))
    if dims:x=torch.flip(x,dims)
    with torch.autocast('cuda',dtype=torch.bfloat16):z=model(x).segmentation_logits
    z=z.float()
    if dims:z=torch.flip(z,dims)
    inverse=np.argsort(permutation).tolist()
    z=z.permute(0,1,*[d+2 for d in inverse]).contiguous()
    if target!=before_resize:z=F.interpolate(z,size=before_resize,mode='trilinear',align_corners=False)
    if any(extra):z=z[(slice(None),slice(None))+tuple(slice(o,o+s) for o,s in zip(offsets,original_shape))]
    assert tuple(z.shape)==tuple(image.shape)
    return z


def dropout(model,p):
    model.eval()
    layers=[]
    if p:
        for name,module in model.named_modules():
            if isinstance(module,(torch.nn.Dropout,torch.nn.Dropout2d,torch.nn.Dropout3d)):
                module.p=float(p);module.train();layers.append(name)
        assert layers,'no existing dropout modules'
    return layers


@torch.inference_mode()
def gate(model,job):
    spec=job['spec'];batch=int(spec.get('batch',8))
    dropout(model,spec.get('dropout_p',0.))
    # Coordinate identity verifies the entire noninterpolating transform roundtrip.
    class Identity:
        def __call__(self,x):
            class Output:pass
            out=Output();out.segmentation_logits=x;return out
    x=torch.arange(8*16*24,device='cuda').reshape(1,1,8,16,24).float()
    check={k:v for k,v in spec.items() if k in ['padding_extra','phase','permutation']}
    for flip in [0,7]:assert torch.equal(forward_view(Identity(),x,check,flip),x)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    image=torch.zeros((batch,1,*CANVAS),device='cuda')
    torch.manual_seed(20261028)
    z=forward_view(model,image,spec,job['views'][0])
    assert bool(torch.isfinite(z).all()) and tuple(z.shape)==tuple(image.shape)
    assert torch.cuda.max_memory_reserved()/2**30<=68
    result=dict(at=now(),status='passed',batch=batch,tail=270%batch,shape=CANVAS,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                geometry_roundtrip=True,dropout_layers=dropout(model,spec.get('dropout_p',0.)),
                patient_model_forward=False)
    del image,z
    return result


def execute(model,job):
    verify_job(job)
    jobdir=R/'jobs'/job['id'];jobdir.mkdir(parents=True,exist_ok=True)
    outputs=TMP/'groups'/job['id'];outputs.mkdir(parents=True,exist_ok=True)
    if not (jobdir/'GATE.json').exists():write(jobdir/'GATE.json',gate(model,job))
    rows,geometry,images,_=data()
    spec=job['spec'];batch=int(spec.get('batch',8));views=job['views']
    existing=sum(p.stat().st_size for p in TMP.rglob('*.npy'))
    required=sum(int(np.prod(geometry[d['case_id']]['extent'])) for d in rows)*4*(3+(len(views) if job.get('keep_atoms') else 0))
    assert existing+required<=300*2**30, 'new scratch payload ceiling; researcher must clear archived unused intermediates'
    dropout(model,spec.get('dropout_p',0.))
    append_log('ACCESS_LEDGER.jsonl',dict(at=now(),job=job['id'],test_tuned=True,scans=270,
          views=views,spec=spec,source_sha256=job['source_sha256'],model_gradients=False,
          checkpoint_sha256=CHECKPOINT_SHA))
    started=time.monotonic();new_forwards=0;repeated_case_forwards=0
    with torch.inference_mode():
        for start in range(0,270,batch):
            guard();slots=list(range(start,min(start+batch,270)))
            done=[(outputs/'mean_probability'/(rows[i]['case_id']+'.npy')).exists() and
                  (outputs/'mean_logit'/(rows[i]['case_id']+'.npy')).exists() and
                  (outputs/'std_probability'/(rows[i]['case_id']+'.npy')).exists() for i in slots]
            if all(done):continue
            repeated_case_forwards+=sum(done)*len(views)
            image=torch.from_numpy(np.array(images[slots],dtype=np.float32))[:,None].cuda()
            total=totalp=square=None
            for vi,flip in enumerate(views):
                torch.manual_seed(int(job.get('seed',20261028))+start*1000+vi)
                z=forward_view(model,image,spec,flip)
                assert torch.isfinite(z).all()
                p=torch.sigmoid(z)
                if total is None:total=z.clone();totalp=p.clone();square=p.square()
                else:total.add_(z);totalp.add_(p);square.add_(p.square())
                if job.get('keep_atoms'):
                    for k,i in enumerate(slots):
                        path=outputs/'atoms'/f'flip{flip}'/(rows[i]['case_id']+'.npy')
                        if not path.exists():save_array(path,z[(k,0)+slices(geometry[rows[i]['case_id']])].cpu().numpy())
                new_forwards+=len(slots)
                del z,p
            mean=totalp/len(views)
            std=(square/len(views)-mean.square()).clamp_min(0).sqrt()
            logits=total/len(views)
            for k,i in enumerate(slots):
                if done[k]:continue
                cid=rows[i]['case_id'];sl=(k,0)+slices(geometry[cid])
                for name,value in [('mean_probability',mean),('mean_logit',logits),('std_probability',std)]:
                    save_array(outputs/name/(cid+'.npy'),value[sl].cpu().numpy())
            status('gpu_render',job=job['id'],completed=min(start+batch,270),total=270,
                   seconds=time.monotonic()-started,case_forwards=new_forwards)
            del image,total,totalp,square,mean,std,logits
    torch.cuda.synchronize()
    files=[dict(path=str(p.relative_to(TMP)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(outputs.rglob('*.npy'))]
    write(jobdir/'DONE.json',dict(at=now(),state='complete',job=job['id'],spec=spec,views=views,
          scans=270,seconds=time.monotonic()-started,new_case_forwards=new_forwards,
          repeated_case_forwards=repeated_case_forwards,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
          test_tuned=True,files=files,scratch_bytes=sum(d['bytes'] for d in files)))
    model.eval();torch.cuda.empty_cache()
