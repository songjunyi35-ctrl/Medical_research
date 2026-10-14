"""Fixed selected state and repaired inference initialization, synthetic input only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.nnunet_resenc_20261013.common import *
os.environ.update(env())
import torch
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

def main():
    verify_source_freeze('EXECUTION_FREEZE.json');verify_manifests()
    mismatches=[p for p,h in read(R/'INSTALLED_SOURCE_SHA256.json').items() if not Path(p).exists() or sha(p)!=h]
    if mismatches:raise RuntimeError('recovered public implementation differs: '+str(mismatches[:5]))
    source=OUT/'model/fold_0/checkpoint_best.pth';expected=read(R/'SELECTED_CHECKPOINT_CPU_AUDIT.json')
    if sha(source)!=expected['checkpoint_sha256']:raise RuntimeError('selected checkpoint changed')
    c=torch.load(source,map_location='cpu',weights_only=False)
    manager=PlansManager(read(R/'TRAIN972_PLANS.json'));config=manager.get_configuration('3d_fullres')
    dataset=read(OUT/'model/dataset.json');labels=manager.get_label_manager(dataset)
    model=get_network_from_plans(config.network_arch_class_name,config.network_arch_init_kwargs,
        config.network_arch_init_kwargs_req_import,1,labels.num_segmentation_heads,allow_init=False,deep_supervision=False).cuda(0)
    model.load_state_dict(c['network_weights'],strict=True);model.eval()
    for k,v in model.state_dict().items():
        if not torch.equal(v.detach().cpu(),c['network_weights'][k]):raise RuntimeError('selected GPU state differs')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        z=model(torch.zeros(1,1,64,64,64,device='cuda:0'))
    assert z.shape==(1,2,64,64,64) and torch.isfinite(z).all()
    write(R/'RECOVERY_GPU_CHECK.json',{'at':now(),'status':'passed','worker':1133854,
      'all_installed_source_hashes_match_training':True,'selected_epoch':c['current_epoch'],
      'selected_checkpoint_sha256':expected['checkpoint_sha256'],'strict_gpu_state_exact':True,
      'synthetic_fp16_forward_finite':True,'patient_model_forwards':0,'test_pixels_access':False})

if __name__=='__main__':main()
