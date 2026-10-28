"""Second/last engineering recovery: rescore existing probability maps only."""
import numpy as np
import torch
from scripts.segresnet_inference_search_20261028.common import *
from scripts.segresnet_inference_search_20261028.operations import volume_decode


def main():
    verify()
    numeric_mode()
    plan = read(R / 'PLAN.json')
    rows, geometry, _, labels = dataset()
    values = np.load(R / 'stage2/counts.npy', mmap_mode='r+')
    count = len(plan['stage2_sigmas']) * len(plan['stage2_thresholds']) * len(plan['component_rules'])
    maximum_difference = 0.
    reproducible = []
    with torch.inference_mode():
        for ri in range(6):
            for slot, row in enumerate(rows):
                guard()
                probability = np.load(TMP / 'top_probabilities' / str(ri) / (row['case_id'] + '.npy'))
                target = labels[slot][slices(geometry[row['case_id']])] > 0
                for ti, temperature in enumerate(plan['volume_decoder_temperatures']):
                    mask, predicted = volume_decode(probability, temperature)
                    tp = int((mask & target).sum())
                    den = predicted + int(target.sum())
                    dice = float(np.float32(2 * tp) / np.float32(den)) if den else 1.
                    maximum_difference = max(maximum_difference, abs(dice - values[ri, count + ti, slot, 0]))
                    values[ri, count + ti, slot] = [dice, tp, predicted - tp]
                    if ri == 4 and slot in (0, 1, 2, 3, 4, 5, 6, 7) and temperature == 2.:
                        for _ in range(3):
                            other, other_count = volume_decode(probability, temperature)
                            assert other_count == predicted and np.array_equal(other, mask)
                        reproducible.append(dict(slot=slot, case_id=row['case_id'], selected_voxels=predicted,
                                                 repeats=4, all_masks_equal=True))
                values.flush()
                status('deterministic_volume_recompute', recipe=ri + 1, recipes=6, completed_scans=slot + 1,
                       model_forward=False, engineering_recovery=2)
    write(R / 'VOLUME_REPAIR_VERIFIED.json', dict(at=now(), status='passed', configurations=30, scans=108,
          model_forwards=0, maximum_dice_change_from_fp32=float(maximum_difference),
          deterministic_repeats=reproducible, implementation='stable GPU sort; sequential CPU float64 prefix'))


if __name__ == '__main__':
    main()
