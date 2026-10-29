import unittest
import torch
from scripts.race_inference_test270_20261029.predictor import flip_inputs


class ReflectionTTA(unittest.TestCase):
    def test_all_flips_are_involutions_and_conjugate_reflection(self):
        image=torch.arange(2*16*24*32,dtype=torch.float32).reshape(2,1,16,24,32)
        valid=(image%3!=0).float()
        reflection=torch.eye(4).repeat(2,1,1);reflection[:,0,0]=-1;reflection[:,0,3]=torch.tensor([13.,15.])
        point=torch.tensor([4.,7.,11.,1.])
        for flip in range(8):
            x,m,v,_=flip_inputs(image,reflection,valid,flip)
            xx,mm,vv,_=flip_inputs(x,m,v,flip)
            self.assertTrue(torch.equal(image,xx));self.assertTrue(torch.equal(valid,vv));self.assertTrue(torch.equal(reflection,mm))
            transform=torch.eye(4)
            for axis,size in enumerate(image.shape[-3:]):
                if flip&(1<<axis):transform[axis,axis]=-1;transform[axis,3]=size-1
            self.assertTrue(torch.equal(m[0]@(transform@point),transform@(reflection[0]@point)))


if __name__=='__main__':unittest.main()
