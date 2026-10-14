from scripts.pretrain_single_a100_20260921 import batches
from scripts.segresnet_two_ideas_common import batches_for_epoch,BASELINE_SEED


def test_single_global_batches_match_original_ddp_combined_order():
    for epoch in (1,75,150):
        old=[list(batches_for_epoch(334,BASELINE_SEED,epoch,r)) for r in (0,1)]
        expected=[a+b for a,b in zip(*old)]
        actual=batches(epoch)
        assert actual==expected and len(actual)==42 and len(actual[-1])==6
        assert sorted(x for batch in actual for x in batch)==list(range(334))
