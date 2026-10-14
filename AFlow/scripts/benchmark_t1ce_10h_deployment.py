"""Warm synthetic GPU timing of frozen exported models, no patient access."""
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.t1ce_10h_common import *

@torch.no_grad()
def main():
 numeric();seed(SEED);torch.set_num_threads(2);device=torch.device('cuda:0')
 configs={'baseline128':{'threshold':.5,'components':[{'checkpoint_path':str(source('pure193').relative_to(ROOT)),'checkpoint_sha256':SOURCES['pure193'][1],'mode':'128'}]},'overall':read(OUT/'deployment/manifest.json'),'best_single':read(OUT/'deployment/best_single_manifest.json')}
 results={}
 for name,cfg in configs.items():
  models=[]
  for c in cfg['components']:
   p=ROOT/c['checkpoint_path'];assert sha(p)==c['checkpoint_sha256'];m=model_for(device,p);m.eval();models.append((c['mode'],m))
  timings=[]
  for batch in [1,2]:
   x128=torch.randn(batch,1,128,128,128,device=device);xn=torch.randn(batch,1,160,208,168,device=device)
   def forward():
    logits=[]
    with torch.autocast('cuda',dtype=torch.bfloat16):
     for mode,model in models:
      z=predict(model,x128 if mode=='128' else xn,mode)
      if name!='baseline128' and mode=='128':z=F.interpolate(z,size=(160,208,168),mode='trilinear',align_corners=False)
      logits.append(z)
    return (sum(logits)/len(logits))>math.log(cfg['threshold']/(1-cfg['threshold']))
   for _ in range(2):forward()
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
   for _ in range(5):y=forward()
   end.record();torch.cuda.synchronize();elapsed=start.elapsed_time(end)/1000
   timings.append({'batch':batch,'repeats':5,'seconds_per_scan':elapsed/(5*batch),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
   del x128,xn,y
  results[name]=timings;del models;torch.cuda.empty_cache()
 for name in ['overall','best_single']:
  for row,base in zip(results[name],results['baseline128']):row['measured_gpu_time_ratio_vs128']=row['seconds_per_scan']/base['seconds_per_scan']
 write(R/'DEPLOYMENT_BENCHMARK.json',{'utc':now(),'results':results,'patient_pixels_access':False,'selected_models_or_thresholds_changed':False,'scope':'warm synthetic GPU model+fusion+threshold time; excludes N4, disk I/O, process startup and model loading','analytic_forward_equivalents':{'overall':configs['overall']['forward_equivalents'],'best_single':configs['best_single']['forward_equivalents']},'code_sha256':sha(Path(__file__))})
 print(json.dumps(results,indent=2))

if __name__=='__main__':main()
