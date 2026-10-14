"""Only training WT targets change; original images/sampler/RNG/engine are fixed."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from scripts import fractional_occupancy_protocol as p
from scripts import train_structured_background_aux as training


class OccupancyData(training.FitData):
    def __init__(self,fit):
        super().__init__(fit)
        receipt=json.loads((p.RECORD/'CACHE_RECEIPT.json').read_text())
        assert receipt['status']=='passed' and receipt['fit_sha256']==p.FIT_SHA and receipt['all_hard_cache_byte_exact']
        self.soft_path=Path(receipt['target_path'])
        assert self.soft_path==p.SOFT_CACHE/'targets.f16'
        p.checked(self.soft_path,receipt['target_sha256']);p.checked(p.SOFT_CACHE/'index.json',receipt['index_sha256'])
        index=json.loads((p.SOFT_CACHE/'index.json').read_text())
        assert index['case_ids']==self.case_ids and index['scans']==684 and index['shape']==[128,128,128]
        self._soft=None

    def __getitem__(self,slot):
        item=super().__getitem__(slot)
        if self._soft is None:self._soft=np.memmap(self.soft_path,mode='r',dtype=np.float16,shape=(684,128,128,128))
        item['label'][0].copy_(torch.from_numpy(np.array(self._soft[slot],dtype=np.float32)))
        return item


def main():
    p.check_freeze()
    seed=int(sys.argv[sys.argv.index('--seed')+1]);arm=sys.argv[sys.argv.index('--arm')+1]
    if seed not in p.SEEDS or arm!='fractional_occupancy':raise ValueError('unregistered seed/arm; aux coefficient must stay0')
    p.write_new(p.RECORD/f'WRAPPER_ENTRY_{seed}.json',{'seed':seed,'arm':arm,'only_override':'WT training target from exact occupancy cache','training_image_bytes_unchanged':True,'sampling_weights_unchanged':True,'auxiliary_coefficient':0,'teacher_checkpoints':[],'development_access':False,'source_engine_sha256':p.sha(Path(training.__file__)),'wrapper_sha256':p.sha(Path(__file__))})
    training.p=p;training.FitData=OccupancyData
    try:training.main()
    finally:
        work=p.WORK/f'seed{seed}_{arm}'
        p.write_new(work/'TRAINING_TARGET_RECEIPT.json',{'target':'exact_cell_occupancy_float16_storage_float32_training','cache_receipt_sha256':p.sha(p.RECORD/'CACHE_RECEIPT.json'),'fit_metric_target':'soft_target_thresholded_above0.5_not_original_hard_GT','images_sampler_base_augmentation_unchanged':True,'teacher_checkpoints':[],'auxiliary_coefficient':0,'train_done_exists':(work/'TRAIN_DONE.json').exists(),'development_access':False})


if __name__=='__main__':main()
