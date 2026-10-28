import numpy as np
import torch
from scripts.segresnet_inference_test270_20261028.common import *
from scripts.segresnet_inference_search_20261028.operations import volume_decode


def main():
    verify()
    historical = read(OLD / 'CACHE_READY.json')
    actual = dict(images=sha(CACHE / 'images.f32'), labels=sha(CACHE / 'labels.u8'), index=sha(CACHE / 'index.json'))
    assert actual['images'] == historical['payload_sha256']['native_image']
    assert actual['labels'] == historical['payload_sha256']['native_label']
    assert actual['index'] == historical['index_sha256']
    subjects = {}
    expected = {'train': (972, 444), 'validation': (108, 50), 'test': (270, 119)}
    hashes = {'train':'6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65',
              'validation':'f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb',
              'test':'a5ddc0ed2c0b269431758b14061da3c6163dd0c48133b60a2ef8efd7f23aa8f9'}
    for split, path in MANIFEST.items():
        assert sha(path) == hashes[split]
        records = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        subjects[split] = {d['subject_id'] for d in records}
        assert (len(records), len(subjects[split])) == expected[split]
        assert len({d['case_id'] for d in records}) == len(records)
    assert not any(subjects[a] & subjects[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
    index = read(CACHE / 'index.json')
    assert tuple(index['canvas']) == CANVAS and index['case_ids'] == [d['case_id'] for d in rows()]
    write(R / 'CACHE_READY.json', dict(at=now(), status='passed', scans=270, subjects=119,
          payload_sha256=actual, all_native_payload_and_index_exact=True,
          source_worker=1152470, destination_worker=1152682, N4_calls_added=0,
          historical_pixels_outside_crop=4, test_pixels_access=True))
    assert torch.cuda.device_count() == 1
    torch.cuda.set_per_process_memory_fraction(.93)
    p = predictor()
    tests = []
    with torch.inference_mode():
        for count in (8, 6):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            x = torch.zeros((count, 1, *CANVAS), device='cuda')
            g = [{'canvas_offset':[0, 0, 0], 'extent':list(CANVAS)}] * count
            result = p.predict(x, g)
            assert len(result) == count and all(d['mask'].shape == CANVAS for d in result)
            peak = torch.cuda.max_memory_reserved() / 2**30
            assert peak <= 74
            tests.append(dict(batch=count, status='passed', peak_reserved_gib=peak,
                              mask_shapes=[list(d['mask'].shape) for d in result]))
            del x, result
        probe = np.random.default_rng(20261028).uniform(0, 1, (20, 21, 22)).astype(np.float32)
        a, ka = volume_decode(probe, 2.)
        b, kb = volume_decode(probe, 2.)
        assert ka == kb and np.array_equal(a, b)
    write(R / 'SYNTHETIC_GATE.json', dict(at=now(), status='passed', tests=tests,
          deterministic_decoder=True, checkpoint_sha256=CHECKPOINT_SHA, patient_model_forward=False,
          validation_code_unchanged=True, test_canvas=list(CANVAS), gpu=torch.cuda.get_device_name()))


if __name__ == '__main__':
    main()
