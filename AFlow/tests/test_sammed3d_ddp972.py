import torch
import pytest
from torch.utils.data import DataLoader
from scripts.train_sammed3d_single972_ddp import epoch_batches
@pytest.mark.parametrize('batch',[2,4])
def test_epoch_unique_complete_equal_rank_batches(batch):
 a=list(epoch_batches(2,batch,0));b=list(epoch_batches(2,batch,1));assert len(a)==len(b)
 full=[]
 for x,y in zip(a,b):assert len(x)==len(y);full+=x+y
 assert len(full)==972 and len(set(full))==972 and sorted(full)==list(range(972))
 g=torch.Generator().manual_seed(20260916+2)
 original=[int(x) for chunk in DataLoader(list(range(972)),batch_size=1,shuffle=True,generator=g,num_workers=0) for x in chunk]
 assert full==original
 assert full!=[z for x,y in zip(epoch_batches(3,batch,0),epoch_batches(3,batch,1)) for z in x+y]
