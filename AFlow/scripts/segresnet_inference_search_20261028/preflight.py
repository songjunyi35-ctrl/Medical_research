import time
import numpy as np
import torch
from scripts.segresnet_inference_search_20261028.common import *
from scripts.segresnet_inference_search_20261028.operations import *


def main():
    verify()
    numeric_mode()
    assert torch.cuda.device_count() == 1
    assert torch.equal(torch.eye(32, device='cuda') @ torch.eye(32, device='cuda'), torch.eye(32, device='cuda'))
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    assert sha(MANIFEST) == 'f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb'
    assert len(rows) == 108 and len({d['subject_id'] for d in rows}) == 50
    base = MANIFEST.parent
    manifests = {
        'train': (base / 'train_t1ce_wt.jsonl', 972, 444, '6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65'),
        'validation': (MANIFEST, 108, 50, 'f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb'),
        'test': (base / 'sealed_test/test_cases_t1ce.jsonl', 270, 119, 'a5ddc0ed2c0b269431758b14061da3c6163dd0c48133b60a2ef8efd7f23aa8f9')}
    subjects = {}
    for split, (path, scans, people, digest) in manifests.items():
        assert sha(path) == digest
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        subjects[split] = {d['subject_id'] for d in records}
        assert len(records) == scans and len(subjects[split]) == people
        assert len({d['case_id'] for d in records}) == scans
    assert not any(subjects[a] & subjects[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')])
    index = read(CACHE / 'index.json')
    assert index['case_ids'] == [d['case_id'] for d in rows]
    assert tuple(index['canvas']) == CANVAS and index['test_access'] is False
    reference = read(ROOT / 'archive/research_records/2026-10-28_race_seg/DATA_validation.json')
    digests = {name: sha(CACHE / name) for name in ['images.f32', 'labels.u8']}
    for name, digest in digests.items():
        assert digest == index['digests'][name] == reference['digests'][name]
    write(R / 'DATA_READY.json', dict(at=now(), status='passed', count=108, subjects=50,
          canvas=CANVAS, payload_sha256=digests, index_sha256=sha(CACHE / 'index.json'),
          copied_from_worker=1152470, test_access=False))
    thresholds = torch.tensor([.1, .3, .5, .9], device='cuda')
    p = torch.tensor([0., .1, .2, .3, .5, .6, .9, 1.], device='cuda')
    target = torch.tensor([0, 1, 0, 1, 1, 0, 1, 1], device='cuda', dtype=torch.bool)
    d, tp, fp = threshold_counts(p, target, thresholds)
    for i, threshold in enumerate(thresholds):
        mask = p > threshold
        assert int(tp[i]) == int((mask & target).sum())
        assert int(fp[i]) == int((mask & ~target).sum())
        assert float(d[i]) == float(2 * (mask & target).sum().float() / (mask.sum() + target.sum()).float())
    x = torch.arange(4 * 5 * 6, device='cuda').reshape(1, 1, 4, 5, 6).float()
    for axis in range(3):
        for shift in [-2, 2]:
            coverage = move_zero(move_zero(torch.ones_like(x), axis, shift), axis, -shift) > 0
            recovered = move_zero(move_zero(x, axis, shift), axis, -shift)
            assert torch.equal(recovered[coverage], x[coverage])
    for flip in range(8):
        dims = tuple(i + 2 for i in range(3) if flip & (1 << i))
        assert torch.equal(torch.flip(torch.flip(x, dims), dims), x)
    pp = np.zeros((12, 12, 12), np.float32)
    pp[1:4, 1:4, 1:4] = .9
    pp[8, 8, 8] = .55
    truth = pp > .8
    for rule in read(R / 'PLAN.json')['component_rules']:
        metric = component_metrics(pp, truth, .5, [rule])[0]
        mask = component_mask(pp, .5, rule)
        truepos = int((mask & truth).sum())
        assert metric[1:] == (truepos, int((mask & ~truth).sum()))
        # Scoring labels cannot affect the output mask.
        assert np.array_equal(component_mask(pp, .5, rule), mask)
    m, count = volume_decode(np.array([.9, .8, .1], np.float32), 1.)
    assert count == 2 and m.tolist() == [True, True, False]
    model = load_model('cuda')
    probes = []
    shape = tuple(int(round(s * 1.15 / 8)) * 8 for s in CANVAS)
    torch.cuda.set_per_process_memory_fraction(.93)
    with torch.inference_mode():
        for batch in [2, 4, 8]:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                image = torch.zeros((batch, 1, *shape), device='cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    z = model(image).segmentation_logits
                assert torch.isfinite(z).all()
                torch.cuda.synchronize()
                started = time.monotonic()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    for _ in range(2):
                        z = model(image).segmentation_logits
                torch.cuda.synchronize()
                seconds = (time.monotonic() - started) / 2
                peak = torch.cuda.max_memory_reserved() / 2**30
                probes.append(dict(batch=batch, status='passed', seconds=seconds,
                                   cases_per_second=batch / seconds, peak_reserved_gib=peak))
                del image, z
            except torch.cuda.OutOfMemoryError:
                probes.append(dict(batch=batch, status='oom_synthetic_only'))
                if 'image' in locals():
                    del image
                if 'z' in locals():
                    del z
                torch.cuda.empty_cache()
    passed = [d for d in probes if d['status'] == 'passed' and d['peak_reserved_gib'] <= 74]
    assert passed
    best = max(d['cases_per_second'] for d in passed)
    selected = min((d for d in passed if d['cases_per_second'] >= .97 * best), key=lambda d: d['batch'])
    # Baseline is always batch2 for exact comparison with the historical report.
    write(R / 'PREFLIGHT.json', dict(at=now(), status='passed', gpu=torch.cuda.get_device_name(),
          checks=['actual_cuda', 'native_SHA', 'threshold_ties', 'shift_coverage', 'mirror_inverse',
                  'component_filter_counts', 'volume_decoder', 'strict_checkpoint'],
          probes=probes, selected_batch=selected['batch'], baseline_batch=2, training=False,
          patient_model_forward=False, max_scale_shape=shape))


if __name__ == '__main__':
    main()
