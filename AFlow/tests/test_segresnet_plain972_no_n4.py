import unittest
import numpy as np
import torch
from scripts.build_no_n4_clean_cache import arrays
from scripts.segresnet_plain972_no_n4_common import batches_for_epoch,learning_rate_for_epoch
from scripts.segresnet_plain972_common import batches_for_epoch as reference_batches

class NoN4Tests(unittest.TestCase):
    def test_image_only_geometry_and_label_mapping(self):
        torch.set_num_threads(1)
        image=np.zeros((24,28,32),np.float32)
        image[3:20,4:24,5:27]=np.arange(17*20*22,dtype=np.float32).reshape(17,20,22)+1
        image[0,0,0]=-4
        label=np.zeros(image.shape,np.uint8);label[8:14,9:16,10:18]=1;label[14:17,9:16,10:18]=4
        x,y,g=arrays('BraTS-GLI-99999-100',image,label,(1,1,1),False)
        x2,y2,g2=arrays('BraTS-GLI-99999-100',image,np.where(label==4,2,label),(1,1,1),False)
        self.assertEqual(g,{'origin':[3,4,5],'extent':[17,20,22],'native_shape':[24,28,32]})
        self.assertEqual(g,g2);self.assertTrue(np.array_equal(x,x2));self.assertGreater(y2.sum(),y.sum())
        self.assertEqual(x.shape,(128,128,128));self.assertTrue(np.isfinite(x).all())
    def test_full_coverage_same_schedule(self):
        for epoch in [1,193,200]:
            ranks=[list(batches_for_epoch(epoch,r)) for r in range(2)]
            self.assertEqual(ranks,[list(reference_batches(epoch,r)) for r in range(2)])
            ids=[i for rank in ranks for batch in rank for i in batch]
            self.assertEqual(sorted(ids),list(range(972)));self.assertEqual(len(ranks[0]),122)
            self.assertEqual(len(ranks[0][-1]),2)
        self.assertAlmostEqual(learning_rate_for_epoch(100,base_lr=2e-4,min_lr=1e-5,constant_epochs=100,total_epochs=200),2e-4)
        self.assertAlmostEqual(learning_rate_for_epoch(200,base_lr=2e-4,min_lr=1e-5,constant_epochs=100,total_epochs=200),1e-5)
if __name__=='__main__':unittest.main()
