import unittest
from scripts.guard_segresnet_no_n4_epoch50 import decide
class DecisionTests(unittest.TestCase):
    def test_no_clear_difference_stops(self):
        self.assertFalse(decide(.009,[.002,.016],.006)['continue_training'])
        self.assertFalse(decide(-.014,[-.028,.001],-.008)['continue_training'])
        self.assertFalse(decide(.015,[.004,.026],-.002)['continue_training'])
    def test_clear_difference_both_directions(self):
        self.assertTrue(decide(.015,[.004,.026],.007)['continue_training'])
        self.assertTrue(decide(-.015,[-.026,-.004],-.007)['continue_training'])
    def test_zero_is_not_excluded(self):
        self.assertFalse(decide(.01,[0,.02],.003)['continue_training'])
        self.assertFalse(decide(-.01,[-.02,0],-.003)['continue_training'])
if __name__=='__main__':unittest.main()
