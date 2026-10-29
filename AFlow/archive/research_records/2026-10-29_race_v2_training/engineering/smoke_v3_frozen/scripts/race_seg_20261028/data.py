import numpy as np
import torch
from torch.utils.data import Dataset
from scripts.race_seg_20261028.common import CACHE,CANVAS,read
from scripts.train_t1ce_wt_capg3d import subject_id
from sammed3d.race_seg import spatial_augment


class RaceDataset(Dataset):
    def __init__(self,split):
        if split not in ('train','validation'): raise ValueError('no test dataset')
        self.split=split;self.folder=CACHE/split;self.index=read(self.folder/'index.json')
        if self.index['schema']!='aflow.race.native' or self.index['test_access'] is not False: raise ValueError('cache role')
        self.case_ids=self.index['case_ids'];self.geometry=self.index['geometry'];self.arrays={}
    def __len__(self): return len(self.case_ids)
    def __getitem__(self,i):
        for name,dtype in [('images.f32',np.float32),('labels.u8',np.uint8)]+([('classes.u8',np.uint8)] if self.split=='train' else []):
            if name not in self.arrays:
                self.arrays[name]=np.memmap(self.folder/name,dtype=dtype,mode='r',shape=(len(self),*CANVAS))
        cid=self.case_ids[i];geo=self.geometry[cid]
        image=torch.from_numpy(np.array(self.arrays['images.f32'][i],dtype=np.float32))[None]
        target=torch.from_numpy(np.array(self.arrays['labels.u8'][i],dtype=np.float32))[None]
        valid=(image!=0).float()
        if not geo['reference_enabled']: valid.zero_()
        classes=(torch.from_numpy(np.array(self.arrays['classes.u8'][i],dtype=np.uint8))[None]
                 if self.split=='train' else torch.empty(0,dtype=torch.uint8))
        return {'image':image,'label':target,'classes':classes,'valid':valid,
                'reflection':torch.tensor(geo['reflection'],dtype=torch.float32),
                'case_id':cid,'subject_id':subject_id(cid)}


def augment_batch(batch,seed):
    """CPU spatial augmentation; same case RNG for both experiment arms."""
    generator=torch.Generator().manual_seed(seed)
    image=batch['image'].clone();target=batch['label'].clone()
    classes=batch['classes'].clone();valid=batch['valid'].clone();matrix=batch['reflection'].clone()
    for i in range(len(image)):
        flips=tuple(bool(v) for v in torch.rand(3,generator=generator)<.5)
        shifts=tuple(int(v) for v in torch.randint(-8,9,(3,),generator=generator))
        values=spatial_augment(image[i],target[i],classes[i],valid[i],matrix[i],flips=flips,shifts=shifts)
        if bool(values[1].sum()>0): image[i],target[i],classes[i],valid[i],matrix[i]=values
    scale=torch.empty((len(image),1,1,1,1)).uniform_(.9,1.1,generator=generator)
    offset=torch.empty((len(image),1,1,1,1)).uniform_(-.1,.1,generator=generator)
    noise=torch.empty((len(image),1,1,1,1)).uniform_(0,.05,generator=generator)
    image=image*scale+offset+torch.randn(image.shape,generator=generator)*noise
    return {'image':image,'label':target,'classes':classes,'valid':valid,'reflection':matrix}
