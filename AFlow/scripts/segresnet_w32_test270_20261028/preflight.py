import torch,numpy as np
from scripts.segresnet_w32_test270_20261028.common import *
def main():
 verify();validate_manifests();assert torch.cuda.device_count()==2
 for i in range(2):
  x=torch.eye(32,device=f'cuda:{i}');assert torch.equal(x@x,x);del x
 label=np.zeros((2,3,4),np.float32);label[0,1,2]=1;z=np.full_like(label,-4);z[0,1,2]=4
 py,off=pad_to_canvas(label);pz,_=pad_to_canvas(z,fill=99)
 m=case_metrics(torch.from_numpy(pz)[None],torch.from_numpy(py)[None],{'canvas_offset':off,'extent':[2,3,4]})
 assert m['locator_dsc']==1 and m['automatic_point']==[0,1,2]
 m=case_metrics(torch.zeros(1,*CANVAS),torch.from_numpy(py)[None],{'canvas_offset':off,'extent':[2,3,4]});assert m['locator_dsc']==0
 write(R/'PREFLIGHT.json',{'at':now(),'status':'passed','gpu_names':[torch.cuda.get_device_name(i) for i in range(2)],'patient_isolation':True,'crop_padding_threshold_cpu_checks':True,'patient_model_forward':False,'disk_free_bytes':shutil.disk_usage(ROOT).free})
if __name__=='__main__':main()
