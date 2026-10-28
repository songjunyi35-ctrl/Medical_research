"""Online stochastic image averaging; store only mean probability to bound scratch."""
import numpy as np
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.render_v1 import gate,forward_view,dropout


def execute(model,job):
    verify_job(job);jobdir=R/'jobs'/job['id'];jobdir.mkdir(parents=True,exist_ok=True)
    if not (jobdir/'GATE.json').exists():write(jobdir/'GATE.json',gate(model,job))
    outputs=TMP/'groups'/job['id'];outputs.mkdir(parents=True,exist_ok=True);rows,geometry,images,_=data()
    spec=job['spec'];views=job['views'];dropout(model,0.)
    required=sum(int(np.prod(geometry[r['case_id']]['extent'])) for r in rows)*4
    assert sum(p.stat().st_size for p in TMP.rglob('*.npy'))+required<=300*2**30
    append_log('ACCESS_LEDGER.jsonl',dict(at=now(),job=job['id'],views=views,spec=spec,GT_input=False,training=False,test_tuned=True,source_sha256=job['source_sha256']))
    start=time.monotonic();count=0
    with torch.inference_mode():
        for first in range(0,270,8):
            guard();slots=list(range(first,min(first+8,270)))
            if all((outputs/'mean_probability'/(rows[i]['case_id']+'.npy')).exists() for i in slots):continue
            x=torch.from_numpy(np.array(images[slots],np.float32))[:,None].cuda();mean=torch.zeros_like(x)
            for vi,flip in enumerate(views):
                torch.manual_seed(job['seed']+first*1000+vi);p=forward_view(model,x,spec,flip).sigmoid()
                mean.add_(p,alpha=1/len(views));count+=len(slots)
            for k,i in enumerate(slots):save_array(outputs/'mean_probability'/(rows[i]['case_id']+'.npy'),mean[(k,0)+slices(geometry[rows[i]['case_id']])].cpu().numpy())
            status('stochastic_noise_average',job=job['id'],completed=min(first+8,270),total=270,case_forwards=count,seconds=time.monotonic()-start)
            del x,mean,p
    files=[dict(path=str(p.relative_to(TMP)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(outputs.rglob('*.npy'))]
    write(jobdir/'DONE.json',dict(at=now(),state='complete',job=job['id'],spec=spec,views=views,scans=270,seconds=time.monotonic()-start,
          new_case_forwards=count,probability_only=True,files=files,scratch_bytes=sum(x['bytes'] for x in files),test_tuned=True))
    model.eval();torch.cuda.empty_cache()
