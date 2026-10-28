"""Hardware-only inference batch choice, no patient input."""
import sys,time
import torch
from scripts.segresnet_w32_test270_20261028.common import *
def main(rank,batch):
 verify();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.95,0);numeric_mode();torch.manual_seed(20260819)
 m=load_model(torch.device('cuda:0'));torch.cuda.reset_peak_memory_stats()
 cpu=torch.randn(batch,1,*CANVAS);times=[]
 try:
  with torch.inference_mode():
   for i in range(8):
    torch.cuda.synchronize();tick=time.monotonic();x=cpu.cuda()
    with torch.autocast('cuda',dtype=torch.bfloat16):z=m(x).segmentation_logits
    assert z.shape==(batch,1,*CANVAS) and torch.isfinite(z).all()
    torch.cuda.synchronize();times.append(time.monotonic()-tick);del x,z
   for tail in sorted({1,135%batch}-{0}):
    with torch.autocast('cuda',dtype=torch.bfloat16):z=m(cpu[:tail].cuda()).segmentation_logits
    assert z.shape==(tail,1,*CANVAS) and torch.isfinite(z).all();del z
  peak=torch.cuda.max_memory_reserved()/2**30
  d={'status':'passed' if peak<=37.5 else 'memory_ceiling_failed','batch':batch,'rank':rank,'seconds_per_batch':sum(times[3:])/5,'scans_per_second':batch/(sum(times[3:])/5),'peak_reserved_gib':peak,'tails_checked':sorted({1,135%batch}-{0}),'strict_model_reload':True,'patient_forward':False}
 except torch.OutOfMemoryError as e:d={'status':'oom','batch':batch,'rank':rank,'error':str(e)[:350],'patient_forward':False}
 write(R/f'PROBE_r{rank}_b{batch}.json',{'at':now(),**d})
if __name__=='__main__':
 rank,batch=map(int,sys.argv[1:])
 try:main(rank,batch)
 except BaseException as e:write(R/f'PROBE_FAILURE_r{rank}_b{batch}.json',{'at':now(),'error':repr(e)});raise
