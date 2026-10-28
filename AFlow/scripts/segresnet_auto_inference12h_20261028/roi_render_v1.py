"""Prediction-defined local zoom; outside ROI remains the original five-view probability."""
import numpy as np
import torch
import torch.nn.functional as F
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.render_v1 import forward_view
from scripts.segresnet_auto_inference12h_20261028.cpu_binary_v1 import unpack


def box(mask,geometry,margin,min_scale):
    """Only a predicted mask and its input geometry enter this decision."""
    if not mask.any():return (0,0,0),CANVAS
    locations=np.where(mask);offset=np.asarray(geometry['canvas_offset'])
    low=np.array([v.min() for v in locations])+offset
    high=np.array([v.max()+1 for v in locations])+offset
    required=(high-low)+2*margin
    scale=max(min_scale,float(np.max(required/np.array(CANVAS))))
    if scale>=1:return (0,0,0),CANVAS
    extent=np.minimum(np.array(CANVAS),np.ceil(np.array(CANVAS)*scale/8).astype(int)*8)
    center=(low+high)/2
    begin=np.minimum(np.maximum(np.floor(center-extent/2).astype(int),0),np.array(CANVAS)-extent)
    assert np.all(begin<=low) and np.all(begin+extent>=high)
    return tuple(map(int,begin)),tuple(map(int,extent))


def feather(shape,width,device):
    factors=[]
    for n in shape:
        k=torch.arange(n,device=device,dtype=torch.float32)
        factors.append(torch.minimum((k+1)/max(width,1),(n-k)/max(width,1)).clamp(max=1))
    return factors[0][:,None,None]*factors[1][None,:,None]*factors[2][None,None,:]


def compose(base,roi_probability,begin,extent,geometry,width):
    canvas=torch.zeros(CANVAS,device=base.device);canvas[slices(geometry)]=base
    region=tuple(slice(a,a+n) for a,n in zip(begin,extent));weight=feather(extent,width,base.device)
    old=canvas[region].clone();canvas[region]=old*(1-weight)+roi_probability*weight
    return canvas[slices(geometry)]


def baseline(cid,core_source,device='cuda'):
    value=None
    for flip in [2,3,4,6,7]:
        z=torch.from_numpy(np.load(TMP/'groups'/core_source/'atoms'/f'flip{flip}'/(cid+'.npy'))).to(device)
        if value is None:value=torch.zeros_like(z)
        value.add_(z.sigmoid(),alpha=.2)
    return value


def gate(model,spec,views):
    model.eval();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    image=torch.zeros((8,1,*CANVAS),device='cuda');z=forward_view(model,image,{},views[0]);assert z.shape==image.shape and torch.isfinite(z).all()
    assert torch.cuda.max_memory_reserved()/2**30<=68
    geom=dict(canvas_offset=[0,0,0],extent=list(CANVAS));mask=np.zeros(CANVAS,bool);mask[60:80,90:110,60:80]=True
    begin,extent=box(mask,geom,spec['margin'],spec['min_scale'])
    base=torch.rand(CANVAS,device='cuda');roi=torch.full(extent,.75,device='cuda')
    out=compose(base,roi,begin,extent,geom,spec['feather_width']);outside=torch.ones(CANVAS,device='cuda',dtype=torch.bool)
    outside[tuple(slice(a,a+n) for a,n in zip(begin,extent))]=False
    assert torch.equal(out[outside],base[outside])
    # A full input box with unchanged probabilities composes identically.
    assert torch.allclose(compose(base,base,(0,0,0),CANVAS,geom,spec['feather_width']),base,atol=6e-8,rtol=0)
    result=dict(at=now(),status='passed',batch=8,patient_model_forward=False,GT_input=False,
                predicted_bbox_enclosed=True,outside_ROI_exact=True,full_box_identity_tolerance=6e-8,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    del image,z,base,roi,out,outside
    return result


@torch.inference_mode()
def execute(model,job):
    verify_job(job);spec=job['spec'];views=job['views'];jobdir=R/'jobs'/job['id'];jobdir.mkdir(parents=True,exist_ok=True)
    if not (jobdir/'GATE.json').exists():write(jobdir/'GATE.json',gate(model,spec,views))
    rows,geometry,images,_=data();outputs=TMP/'groups'/job['id'];outputs.mkdir(parents=True,exist_ok=True)
    required=sum(int(np.prod(geometry[r['case_id']]['extent'])) for r in rows)*12
    assert sum(p.stat().st_size for p in TMP.rglob('*.npy'))+required<=300*2**30
    append_log('ACCESS_LEDGER.jsonl',dict(at=now(),job=job['id'],role='prediction_only_ROI_zoom',views=views,spec=spec,
          source_sha256=job['source_sha256'],GT_input=False,test_tuned=True,model_gradients=False))
    plans=[]
    for row in rows:
        cid=row['case_id'];mask=unpack(OLD/'predictions'/(cid+'.npz'));begin,extent=box(mask,geometry[cid],spec['margin'],spec['min_scale'])
        plans.append(dict(case_id=cid,begin=begin,extent=extent,zoomed=extent!=CANVAS))
    write(jobdir/'ROI_PLANS.json',dict(at=now(),cases=plans,zoomed_scans=sum(x['zoomed'] for x in plans),GT_input=False))
    start=time.monotonic();actual_forwards=0
    for first in range(0,270,8):
        guard();slots=list(range(first,min(first+8,270)));active=[i for i in slots if plans[i]['zoomed']]
        done=lambda i:all((outputs/k/(rows[i]['case_id']+'.npy')).exists() for k in ['mean_probability','mean_logit','std_probability'])
        active=[i for i in active if not done(i)]
        if active:
            inputs=[]
            for i in active:
                begin,extent=plans[i]['begin'],plans[i]['extent'];sl=tuple(slice(a,a+n) for a,n in zip(begin,extent))
                x=torch.from_numpy(np.array(images[i][sl],np.float32)).cuda()[None,None]
                inputs.append(F.interpolate(x,size=CANVAS,mode='trilinear',align_corners=False))
            x=torch.cat(inputs,0);total=torch.zeros_like(x);square=torch.zeros_like(x)
            for flip in views:
                p=forward_view(model,x,{},flip).sigmoid();total.add_(p,alpha=1/len(views));square.add_(p.square(),alpha=1/len(views));actual_forwards+=len(active)
            mean=total;std=(square-mean.square()).clamp_min(0).sqrt()
        for i in slots:
            if done(i):continue
            cid=rows[i]['case_id'];base=baseline(cid,job['core_source'])
            if i in active:
                k=active.index(i);extent=plans[i]['extent'];roi=F.interpolate(mean[k:k+1],size=extent,mode='trilinear',align_corners=False)[0,0]
                p=compose(base,roi,plans[i]['begin'],extent,geometry[cid],spec['feather_width'])
                uncertainty_canvas=torch.zeros(CANVAS,device=base.device)
                region=tuple(slice(a,a+n) for a,n in zip(plans[i]['begin'],extent))
                uncertainty_canvas[region]=F.interpolate(std[k:k+1],size=extent,mode='trilinear',align_corners=False)[0,0]*feather(extent,spec['feather_width'],base.device)
                uncertainty=uncertainty_canvas[slices(geometry[cid])]
            else:p=base;uncertainty=torch.zeros_like(base)
            for name,value in [('mean_probability',p),('mean_logit',torch.logit(p.clamp(1e-7,1-1e-7))),('std_probability',uncertainty)]:save_array(outputs/name/(cid+'.npy'),value.cpu().numpy())
        status('prediction_roi_zoom',job=job['id'],completed=min(first+8,270),total=270,case_forwards=actual_forwards,seconds=time.monotonic()-start)
        if active:del inputs,x,total,square,mean,std
    files=[dict(path=str(p.relative_to(TMP)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(outputs.rglob('*.npy'))]
    write(jobdir/'DONE.json',dict(at=now(),state='complete',job=job['id'],spec=spec,views=views,scans=270,
          seconds=time.monotonic()-start,new_case_forwards=actual_forwards,base_case_forwards_reused=1350,
          zoomed_scans=sum(x['zoomed'] for x in plans),files=files,scratch_bytes=sum(x['bytes'] for x in files),
          deployment_mean_case_forwards=5+len(views)*sum(x['zoomed'] for x in plans)/270,test_tuned=True))
    model.eval();torch.cuda.empty_cache()
