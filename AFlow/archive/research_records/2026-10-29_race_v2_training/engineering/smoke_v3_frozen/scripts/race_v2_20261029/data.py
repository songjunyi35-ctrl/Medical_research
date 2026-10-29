"""Read-only worker-local train/validation caches with historical augmentation."""
import numpy as np
import torch
from torch.utils.data import Dataset
from scripts.race_v2_20261029.common import CACHE, CANVAS, read
from scripts.train_t1ce_wt_capg3d import subject_id
from scripts.race_seg_20261028.data import augment_batch

class RaceDataset(Dataset):
    def __init__(self, split):
        if split not in ('train','validation'): raise ValueError('no test dataset')
        self.split=split; self.folder=CACHE/split; self.index=read(self.folder/'index.json')
        assert self.index['schema']=='aflow.race.native' and self.index['test_access'] is False
        assert tuple(self.index['canvas'])==CANVAS
        self.case_ids=self.index['case_ids']; self.geometry=self.index['geometry']; self.arrays={}
    def __len__(self): return len(self.case_ids)
    def __getitem__(self,i):
        for name,dtype in [('images.f32',np.float32),('labels.u8',np.uint8)]+([('classes.u8',np.uint8)] if self.split=='train' else []):
            if name not in self.arrays:
                self.arrays[name]=np.memmap(self.folder/name,dtype=dtype,mode='r',shape=(len(self),*CANVAS))
        cid=self.case_ids[i]; geo=self.geometry[cid]
        image=torch.from_numpy(np.array(self.arrays['images.f32'][i],dtype=np.float32))[None]
        target=torch.from_numpy(np.array(self.arrays['labels.u8'][i],dtype=np.float32))[None]
        valid=(image!=0).float()
        if not geo['reference_enabled']: valid.zero_()
        classes=(torch.from_numpy(np.array(self.arrays['classes.u8'][i],dtype=np.uint8))[None]
                 if self.split=='train' else torch.empty(0,dtype=torch.uint8))
        return dict(image=image,label=target,classes=classes,valid=valid,
                    reflection=torch.tensor(geo['reflection'],dtype=torch.float32),
                    case_id=cid,subject_id=subject_id(cid))

