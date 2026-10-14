"""Contract tests: coverage, full-cohort sampling, weighted loss, and stop rules."""
import unittest
import numpy as np
import torch
from scripts.t1ce_10h_common import starts,batches,criterion,stop_rule,case_metric

class CampaignContracts(unittest.TestCase):
 def test_canvas_coverage(self):
  for shape in [(160,208,160),(160,208,168)]:
   seen=np.zeros(shape,dtype=np.uint8)
   for s in starts(shape):seen[tuple(slice(x,x+128) for x in s)]+=1
   self.assertEqual(len(starts(shape)),8);self.assertGreater(seen.min(),0)
 def test_full_cohort_tail_and_rank_disjoint(self):
  for world in [1,2]:
   parts=[[i for b in batches(2,r,world) for i in b] for r in range(world)]
   self.assertEqual(sorted(sum(parts,[])),list(range(972)))
   self.assertEqual(len(batches(2,0,world)),122)
   self.assertEqual(len(batches(2,0,world)[-1]),4//world)
 def test_weights_not_batch_normalized(self):
  x=torch.zeros(2,1,3,3,3);y=torch.ones_like(x);w=torch.tensor([.5,2.])
  self.assertTrue(torch.allclose(criterion(x,y,w,'B'),criterion(x,y,torch.ones(2),'B')*1.25))
 def test_empty_patch_and_gradient(self):
  for arm in 'ABC':
   x=torch.zeros(2,1,4,4,4,requires_grad=True);loss=criterion(x,torch.zeros_like(x),torch.ones(2),arm);loss.backward()
   self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(float(x.grad.abs().sum()),0)
 def test_budget_rules(self):
  h=[{'epoch':e,'score':.75} for e in [0,1,5,10,15,20,25]]
  self.assertIsNotNone(stop_rule(h));h[-1]['score']=.76;self.assertIsNone(stop_rule(h))
  h=[{'epoch':e,'score':.75 if e==0 else .72} for e in [0,1,5,10]];self.assertIsNotNone(stop_rule(h))
 def test_native_metric_counts(self):
  x=torch.full((2,2,2),-1.);y=torch.zeros_like(x);y[0,0,0]=1;x[0,0,0]=2;x[0,0,1]=1
  m=case_metric(x,y,'BraTS-GLI-00001-000');self.assertAlmostEqual(m['locator_dsc'],2/3);self.assertEqual(m['fp'],1);self.assertTrue(m['automatic_point_hits_gt'])

if __name__=='__main__':unittest.main()
