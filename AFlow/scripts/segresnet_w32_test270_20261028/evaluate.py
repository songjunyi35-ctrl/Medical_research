"""Disjoint135-case rank, exactly once; persistent masks and journal precede statistics."""
import sys,time,traceback
import numpy as np
import torch
from scripts.segresnet_w32_test270_20261028.common import *

def main(rank):
 verify();validate_manifests();numeric_mode();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.95,0);torch.manual_seed(20260819)
 ready=read(R/'CACHE_READY.json');assert ready['historical_payload_exact'] and sha(CACHE/'index.json')==ready['index_sha256']
 access=read(R/'MODEL_PREACCESS.json');assert access['checkpoint_sha256']==CHECKPOINT_SHA
 batch=read(R/'BATCH_DECISION.json')['batch'];indices=list(range(rank,270,2));expected=rows('test')
 assert access['rank_case_ids'][str(rank)]==[expected[i]['case_id'] for i in indices]
 dataset=NativeTestDataset(CACHE,[d['case_id'] for d in expected]);model=load_model(torch.device('cuda:0'))
 write_new(R/f'MODEL_RANK_ACCESS_{rank}.json',{'at':now(),'rank':rank,'case_ids':access['rank_case_ids'][str(rank)],'checkpoint_sha256':CHECKPOINT_SHA,'batch':batch,'no_automatic_repeated_inference':True})
 predictions=R/'predictions';predictions.mkdir(exist_ok=True);scored=[];files=[];batches=0;started=time.monotonic();torch.cuda.reset_peak_memory_stats()
 with torch.inference_mode():
  for start in range(0,len(indices),batch):
   deadline();items=[dataset[i] for i in indices[start:start+batch]]
   image=torch.stack([d['image'] for d in items]).cuda()
   with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(image).segmentation_logits.float()
   batches+=1
   if not torch.isfinite(logits).all():raise RuntimeError('nonfinite patient output')
   for k,item in enumerate(items):
    cid=item['case_id'];geo=dataset.geometry[cid];label=item['label'].cuda();metric=case_metrics(logits[k],label,geo)
    sl=tuple(slice(o,o+e) for o,e in zip(geo['canvas_offset'],geo['extent']))
    pred=(logits[k,0][sl]>0).cpu().numpy();target=item['label'][0][sl].numpy()>.5
    tp=int((pred&target).sum());fp=int((pred&~target).sum());fn=int((~pred&target).sum())
    independent=float(np.float32(2*tp)/np.float32(2*tp+fp+fn)) if 2*tp+fp+fn else 1.
    assert independent==metric['locator_dsc']
    point=metric['automatic_point'];assert bool(target[tuple(point)])==metric['automatic_point_hits_gt']
    path=predictions/f'{cid}.npz'
    if path.exists():raise RuntimeError('refuse overwriting existing prediction')
    disk(pred.size//8+2**20)
    np.savez_compressed(path,mask_bits=np.packbits(pred.reshape(-1)),shape=np.array(pred.shape),point=np.array(point))
    metric.update(case_id=cid,subject_id=item['subject_id'],tp=tp,fp=fp,fn=fn,source_slot=indices[start+k])
    scored.append(metric);files.append({'case_id':cid,'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
   write(R/f'INFERENCE_JOURNAL_{rank}.json',{'at':now(),'rank':rank,'completed_scans':len(scored),'per_case':scored,'predictions':files})
   write(R/f'HEARTBEAT_{rank}.json',{'at':now(),'rank':rank,'completed':len(scored),'total':135})
   del image,logits
 torch.cuda.synchronize()
 assert len(scored)==135 and {d['case_id'] for d in scored}==set(access['rank_case_ids'][str(rank)])
 write(R/f'RAW_RANK_{rank}.json',{'at':now(),'rank':rank,'per_case':scored,'predictions':files,'seconds':time.monotonic()-started,'batch_forward_calls':batches,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30})
if __name__=='__main__':
 rank=int(sys.argv[1])
 try:main(rank)
 except BaseException as e:write(R/f'INFERENCE_FAILURE_{rank}.json',{'at':now(),'error':repr(e),'traceback':traceback.format_exc(),'model_access_started':(R/f'MODEL_RANK_ACCESS_{rank}.json').exists(),'no_automatic_retry':True});raise
