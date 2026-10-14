import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.sammed3d_graymean_single972 import *
import gc

def trial(batch):
 device=torch.device('cuda:0');seed_all(SEED);torch.set_num_threads(2)
 model=build_model(device);model.train();opt=optimizer_for(model);scaler=torch.amp.GradScaler('cuda',init_scale=128.)
 image=torch.randn(batch,1,128,128,128,device=device);target=torch.zeros_like(image);target[:,:,40:88,40:88,40:88]=1;image=image+1.5*target
 train_step(model,opt,scaler,image,target,1.);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 t=time.monotonic();values=[train_step(model,opt,scaler,image,target,1.) for _ in range(2)];torch.cuda.synchronize();seconds=time.monotonic()-t
 return {'batch_per_gpu':batch,'seconds_two_steps':seconds,'samples_per_second':2*batch/seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'values':values,'status':'passed'}
if __name__=='__main__':
 report={'utc':now(),'mode':'synthetic_only','formal_data_access':False,'device':'A10040GB idle second GPU','trials':[]}
 for b in [2,4]:
  try:result=trial(b)
  except torch.cuda.OutOfMemoryError as e:result={'batch_per_gpu':b,'status':'out_of_memory','error':str(e)}
  report['trials'].append(result);print(json.dumps(result),flush=True);gc.collect();torch.cuda.empty_cache()
 valid=[x for x in report['trials'] if x['status']=='passed' and x['peak_reserved_bytes']<=34*(1<<30)]
 if not valid:raise RuntimeError('no batch passes memory gate')
 best=max(valid,key=lambda x:x['samples_per_second']);near=[x for x in valid if x['samples_per_second']>=best['samples_per_second']/1.05]
 report['selected_batch_per_gpu']=min(x['batch_per_gpu'] for x in near);report['world_size']=2;report['global_batch']=2*report['selected_batch_per_gpu'];write(RECORD/'BATCH_SELECTION.json',report)
