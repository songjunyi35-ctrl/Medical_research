"""Three prespecified108 cases, same cached logits; no network or GT input."""
import fcntl,time
import numpy as np
import torch
from scripts.race_v2_inference_a_20261029.common import *
from scripts.race_v2_inference_a_20261029.predictor import decode_all
from scripts.race_inference_test270_20261029.predictor import decode_outputs
def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0';verify();torch.set_num_threads(1)
    lock=Path('/tmp/aflow_race_v2_20261029/locks/worker1154121_gpu0.lock').open('a')
    while True:
        guard()
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:time.sleep(1)
    assert (R/'ATOMS_READY.json').exists()
    index=read(CACHE/'index.json');images=np.memmap(CACHE/'images.f32',np.float32,'r',shape=(108,*CANVAS))
    entries={(a['slot'],a['view']):a for a in read(R/'ATOMS_READY.json')['atoms']}
    recipe=read(R/'PLAN.json')['recipes'][2];assert recipe['id']=='legacy_full';audits=[]
    for slot in (0,53,107):
        g=index['geometry'][index['case_ids'][slot]];sl=slices(g);image=np.array(images[slot][sl],np.float32)
        logits=[]
        for v in range(8):
            entry=entries[slot,v];assert sha(entry['path'])==entry['sha256'];logits.append(np.load(entry['path'],allow_pickle=False))
        cpu=decode_all(logits,image,[recipe])[0];gpu=decode_outputs(logits,image,recipe['config'])['enhanced']
        audits.append(dict(slot=slot,case_id=index['case_ids'][slot],mask_different_voxels=int((cpu['mask']!=gpu['mask']).sum()),point_equal=cpu['automatic_point']==gpu['automatic_point'],cut_equal=cpu['quantized_cut']==gpu['quantized_cut'],cpu_cut=cpu['quantized_cut'],gpu_cut=gpu['quantized_cut']))
    write(R/'LEGACY_CPU_GPU_DECODER_AUDIT.json',dict(at=now(),status='passed' if all(a['mask_different_voxels']==0 and a['point_equal'] and a['cut_equal'] for a in audits) else 'numerical_execution_difference_reported',prespecified_slots=[0,53,107],audits=audits,GT_loaded=False,network_forwards=0,new270_access=False,source_sha256=sha(Path(__file__)),decoder_parameters_changed=False))
if __name__=='__main__':main()
