import time
import numpy as np
import torch
from scripts.segresnet_inference_search_20261028.common import *
from scripts.segresnet_inference_search_20261028.operations import predict_atom


def array_path(atom, cid):
    return TMP / 'atoms' / atom / (cid + '.npy')


def save_array(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as f:
        np.save(f, value)
    os.replace(temp, path)


def baseline_audit(rows, geometries, labels):
    reference = {d['case_id']: d for d in read(REFERENCE)['per_case']}
    cases = []
    for i, row in enumerate(rows):
        cid = row['case_id']
        z = np.load(array_path('scale1_flip0', cid))
        y = labels[i][slices(geometries[cid])] > 0
        p = z > 0
        tp = int((p & y).sum())
        fp = int((p & ~y).sum())
        fn = int((~p & y).sum())
        den = 2 * tp + fp + fn
        dice = float(np.float32(2 * tp) / np.float32(den)) if den else 1.
        point = list(map(int, np.unravel_index(int(z.argmax()), z.shape)))
        old = reference[cid]
        cases.append(dict(**row, locator_dsc=dice, automatic_point=point,
                          automatic_point_hits_gt=bool(y[tuple(point)]), tp=tp, fp=fp, fn=fn,
                          dice_difference=dice - old['locator_dsc'],
                          point_equal=point == old['automatic_point']))
    assert len(cases) == 108
    maximum = max(abs(d['dice_difference']) for d in cases)
    point_equal = all(d['point_equal'] for d in cases)
    write(R / 'BASELINE_RAW.json', dict(at=now(), per_case=cases, max_dice_difference=maximum,
          all_points_equal=point_equal, baseline_reference_sha256=sha(REFERENCE),
          checkpoint_sha256=CHECKPOINT_SHA, test_access=False))
    historical_exact = maximum <= 1e-6 and point_equal
    diagnostic = read(R / 'DIAGNOSTIC_PARITY.json') if (R / 'DIAGNOSTIC_PARITY.json').exists() else {}
    controlled_exact = (diagnostic.get('original_evaluator_vs_new_raw_max_dice') == 0.
                        and diagnostic.get('all_points_equal') is True)
    assert historical_exact or controlled_exact, 'baseline connection/numeric parity failed'
    write(R / 'BASELINE_GATE.json', dict(at=now(), status='passed', max_dice_difference=maximum,
          all_points_equal=point_equal, historical_bitwise_parity=historical_exact,
          same_worker_original_evaluator_exact=controlled_exact,
          contract='AMENDMENT_001_CONTROLLED_BASELINE.md' if not historical_exact else 'original',
          scan_dice=float(np.mean([d['locator_dsc'] for d in cases]))))


def main():
    plan = verify()
    numeric_mode()
    torch.cuda.set_per_process_memory_fraction(.93)
    rows, geometries, images, labels = dataset()
    model = load_model('cuda')
    preflight = read(R / 'PREFLIGHT.json')
    assert preflight['status'] == 'passed'
    if not (R / 'VALIDATION_MODEL_PREACCESS.json').exists():
        write(R / 'VALIDATION_MODEL_PREACCESS.json', dict(at=now(), scans=108, subjects=50,
              checkpoint_sha256=CHECKPOINT_SHA, freeze_sha256=sha(R / 'EXECUTION_FREEZE.json'),
              test_access=False, optimizer=False, gradient=False))
    for number, atom in enumerate(plan['atoms']):
        guard()
        receipt = R / 'atoms' / (atom['id'] + '.json')
        if receipt.exists():
            assert read(receipt)['status'] == 'complete'
            assert all(array_path(atom['id'], d['case_id']).is_file() for d in rows)
            if number == 0:
                baseline_audit(rows, geometries, labels)
            continue
        if number:
            assert read(R / 'BASELINE_GATE.json')['status'] == 'passed'
        batch = 2 if number == 0 else preflight['selected_batch']
        if atom['kind'] == 'sliding':
            batch = 1
        begun = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        forward_batches = 0
        with torch.inference_mode():
            for start in range(0, len(rows), batch):
                guard()
                slots = [i for i in range(start, min(start + batch, len(rows)))
                         if not array_path(atom['id'], rows[i]['case_id']).exists()]
                if not slots:
                    continue
                image = torch.from_numpy(np.array(images[slots], dtype=np.float32))[:, None].cuda()
                baseline = None
                if atom['kind'] in ('shift', 'rotate'):
                    base = np.zeros((len(slots), 1, *CANVAS), np.float32)
                    for k, i in enumerate(slots):
                        cid = rows[i]['case_id']
                        base[(k, 0) + slices(geometries[cid])] = np.load(array_path('scale1_flip0', cid))
                    baseline = torch.from_numpy(base).cuda()
                output = predict_atom(model, image, atom, baseline)
                assert torch.isfinite(output).all()
                forward_batches += 1
                for k, i in enumerate(slots):
                    cid = rows[i]['case_id']
                    value = output[(k, 0) + slices(geometries[cid])].cpu().numpy()
                    save_array(array_path(atom['id'], cid), value)
                status('inference_atoms', atom=atom['id'], atom_number=number + 1,
                       atoms_total=len(plan['atoms']), completed_in_atom=min(start + batch, len(rows)),
                       total_scans=108, seconds_current_atom=time.monotonic() - begun)
                del image, output, baseline
        torch.cuda.synchronize()
        actual_bytes = sum(p.stat().st_size for p in (TMP / 'atoms').rglob('*.npy'))
        assert actual_bytes <= 220 * 2**30, 'scratch search payload above220GiB'
        write(receipt, dict(at=now(), status='complete', atom=atom, scans=108,
              seconds=time.monotonic() - begun, batch=batch, new_forward_batches=forward_batches,
              peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
              scratch_bytes=actual_bytes, files=[dict(case_id=d['case_id'],
              sha256=sha(array_path(atom['id'], d['case_id']))) for d in rows]))
        if number == 0:
            baseline_audit(rows, geometries, labels)
    status('atoms_complete', atoms=len(plan['atoms']), scans=108)


if __name__ == '__main__':
    main()
