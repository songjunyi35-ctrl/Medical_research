import time
import numpy as np
import torch
from scripts.segresnet_inference_test270_20261028.common import *


def main():
    verify()
    assert read(R / 'CACHE_READY.json')['status'] == read(R / 'SYNTHETIC_GATE.json')['status'] == 'passed'
    write_new(R / 'MODEL_PREACCESS.json', dict(at=now(), checkpoint_sha256=CHECKPOINT_SHA,
          selected_recipe_sha256=sha(R / 'SELECTED_RECIPE.json'), freeze_sha256=sha(R / 'EXECUTION_FREEZE.json'),
          case_ids=[d['case_id'] for d in rows()], scans=270, subjects=119,
          forwards_per_scan=5, batch=8, tail_batch=6, test_access=True,
          current_test_retuning=False, no_automatic_repeated_forward=True))
    torch.cuda.set_per_process_memory_fraction(.93)
    model = predictor()
    data = dataset()
    predictions = R / 'predictions'
    predictions.mkdir(exist_ok=True)
    scored, files = [], []
    begun = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for start in range(0, 270, 8):
        guard()
        batch = [data[i] for i in range(start, min(start + 8, 270))]
        images = torch.stack([d['image'] for d in batch])
        geometry = [data.geometry[d['case_id']] for d in batch]
        results = model.predict(images, geometry)
        for k, (item, g, output) in enumerate(zip(batch, geometry, results)):
            cid = item['case_id']
            crop = tuple(slice(o, o + e) for o, e in zip(g['canvas_offset'], g['extent']))
            target = item['label'][0][crop].numpy() > .5
            mask = output['mask']
            assert mask.shape == tuple(g['extent']) == target.shape
            point = output['automatic_point']
            tp, fp, fn = [int(a.sum()) for a in [mask & target, mask & ~target, ~mask & target]]
            den = 2 * tp + fp + fn
            dice = float(np.float32(2 * tp) / np.float32(den)) if den else 1.
            path = predictions / (cid + '.npz')
            assert not path.exists(), 'do not overwrite patient prediction'
            np.savez_compressed(path, mask_bits=np.packbits(mask.ravel()), shape=np.array(mask.shape), point=np.array(point))
            scored.append(dict(case_id=cid, subject_id=item['subject_id'], source_slot=start+k,
                  locator_dsc=dice, automatic_point=point, automatic_point_hits_gt=bool(target[tuple(point)]),
                  tp=tp, fp=fp, fn=fn, crop_extent=g['extent']))
            files.append(dict(case_id=cid, path=str(path.relative_to(ROOT)), sha256=sha(path)))
        write(R / 'INFERENCE_JOURNAL.json', dict(at=now(), per_case=scored, predictions=files, completed=len(scored)))
        status('test_inference', completed=len(scored), total=270, test_access=True,
               seconds=time.monotonic()-begun, forwards_per_scan=5)
    torch.cuda.synchronize()
    validate_rows(scored)
    raw = dict(at=now(), method=ID, per_case=scored, predictions=files, test_access=True,
          current_test_retuning=False, checkpoint_sha256=CHECKPOINT_SHA,
          recipe_sha256=sha(R / 'SELECTED_RECIPE.json'), grid='native1mm crop; padding excluded',
          runtime=dict(seconds=time.monotonic()-begun, gpu_count=1, batch=8, tail_batch=6,
              forwards_per_scan=5, total_case_forwards=1350,
              peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
              input_voxel_equivalents_128cubed=float(5*np.prod(CANVAS)/128**3),
              scope='input loading, five-view prediction, decoder, scoring, compressed masks, journals; excludes model load/cache copy'))
    write(R / 'RAW_TEST270.json', raw)
    status('test_inference_complete', completed=270, test_access=True, runtime=raw['runtime'])


if __name__ == '__main__':
    main()
