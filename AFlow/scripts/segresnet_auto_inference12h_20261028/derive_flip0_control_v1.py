"""Current-resource original-view threshold mask from already computed logits."""
import numpy as np
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_v2 import guard
from scripts.segresnet_auto_inference12h_20261028.cpu_binary_v1 import unpack


@torch.inference_mode()
def main():
    rows,geometry,_,labels=data();folder=R/'control_from_core_flip0';folder.mkdir(parents=True,exist_ok=True)
    write(folder/'FREEZE.json',dict(at=now(),source_sha256=sha(Path(__file__)),checkpoint_sha256=CHECKPOINT_SHA,
          source_render_done_sha256=sha(R/'jobs/r1_0000_core_mirror8/DONE.json'),role='same-resource control from existing logits',neural_forwards_added=0))
    records=[];changed=0
    historical=ROOT/'archive/research_records/2026-10-28_segresnet_w32_e171_test270'
    for slot,row in enumerate(rows):
        guard();cid=row['case_id'];z=torch.from_numpy(np.load(TMP/'groups/r1_0000_core_mirror8/atoms/flip0'/(cid+'.npy'))).cuda()
        mask=(z.sigmoid()>.5).cpu().numpy();point=list(map(int,np.unravel_index(int(z.argmax()),z.shape)));target=np.array(labels[slot][slices(geometry[cid])]>0)
        tp=int((mask&target).sum());pred=int(mask.sum());positive=int(target.sum());den=pred+positive;score=float(np.float32(2*tp)/np.float32(den)) if den else 1.
        old=unpack(historical/'predictions'/(cid+'.npz'));different=int((mask!=old).sum());changed+=different
        path=folder/'predictions'/(cid+'.npz');path.parent.mkdir(exist_ok=True);np.savez_compressed(path,mask_bits=np.packbits(mask.ravel()),shape=np.array(mask.shape),point=np.array(point))
        records.append(dict(case_id=cid,subject_id=row['subject_id'],locator_dsc=score,automatic_point=point,automatic_point_hits_gt=bool(target[tuple(point)]),tp=tp,fp=pred-tp,fn=positive-tp,source_slot=slot,historical_mask_different_voxels=different))
    write(folder/'RAW_TEST270.json',dict(at=now(),per_case=records,test_tuned=True,model_forwards_added=0,role='current-resource original flip0 mask; audit/control not new method'))
    write(folder/'DONE.json',dict(at=now(),state='complete',scan_dice=float(np.mean([r['locator_dsc'] for r in records])),historical_mask_different_voxels=changed,point_hits=sum(r['automatic_point_hits_gt'] for r in records),model_forwards_added=0,GT_input_to_mask=False))
    print(read(folder/'DONE.json'))
if __name__=='__main__':main()
