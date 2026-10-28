"""Full-cohort exhaustive decoding of the frozen inference recipes."""
import numpy as np
import torch
from scripts.segresnet_inference_search_20261028.common import *
from scripts.segresnet_inference_search_20261028.operations import fuse_score, threshold_counts
from scripts.segresnet_inference_search_20261028.atoms import array_path


def atom_cost(atom):
    if atom['kind'] == 'sliding':
        import math
        stride = int(128 * (1 - atom['overlap']))
        windows = math.prod(int(math.ceil((s - 128) / stride)) + 1 for s in CANVAS)
        return windows, windows * 128**3 / np.prod(CANVAS)
    scale = atom.get('scale', 1.)
    shape = tuple(max(8, int(round(s * scale / 8)) * 8) for s in CANVAS)
    return 1, float(np.prod(shape) / np.prod(CANVAS))


def recipe_cost(recipe, plan):
    wanted = set(recipe['atoms'])
    atoms = {d['id']: d for d in plan['atoms']}
    if any(atoms[name]['kind'] in ('shift', 'rotate') for name in wanted):
        wanted.add('scale1_flip0')
    values = [atom_cost(atoms[name]) for name in sorted(wanted)]
    return dict(case_forwards=sum(d[0] for d in values), native_voxel_equivalent=sum(d[1] for d in values))


def main():
    plan = verify()
    numeric_mode()
    rows, geometries, _, labels = dataset()
    assert all((R / 'atoms' / (a['id'] + '.json')).exists() for a in plan['atoms'])
    nrecipe, nt = len(plan['recipes']), len(plan['thresholds'])
    folder = R / 'stage1'
    folder.mkdir(exist_ok=True)
    arrays = {}
    for name, dtype in [('dice', np.float32), ('tp', np.uint32), ('fp', np.uint32)]:
        path = folder / (name + '.npy')
        arrays[name] = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+',
                                                dtype=dtype, shape=(nrecipe * nt, 108))
    points_path = folder / 'points.npy'
    points = np.lib.format.open_memmap(points_path, mode='r+' if points_path.exists() else 'w+',
                                      dtype=np.int16, shape=(nrecipe, 108, 3))
    hit_path = folder / 'point_hits.npy'
    hits = np.lib.format.open_memmap(hit_path, mode='r+' if hit_path.exists() else 'w+',
                                    dtype=np.bool_, shape=(nrecipe, 108))
    thresholds = torch.tensor(plan['thresholds'], dtype=torch.float32, device='cuda')
    with torch.inference_mode():
        for slot, row in enumerate(rows):
            guard()
            marker = folder / f'case{slot:03}.json'
            if marker.exists():
                continue
            cid = row['case_id']
            target = torch.from_numpy(np.array(labels[slot][slices(geometries[cid])] > 0)).cuda()
            logits = {a['id']: torch.from_numpy(np.load(array_path(a['id'], cid))).cuda() for a in plan['atoms']}
            for ri, recipe in enumerate(plan['recipes']):
                score = fuse_score(logits, recipe)
                probability = score if recipe['fusion'] == 'probability' else torch.sigmoid(score)
                d, tp, fp = threshold_counts(probability, target, thresholds)
                span = slice(ri * nt, (ri + 1) * nt)
                arrays['dice'][span, slot] = d.cpu().numpy()
                arrays['tp'][span, slot] = tp.cpu().numpy()
                arrays['fp'][span, slot] = fp.cpu().numpy()
                flat = int(score.argmax())
                points[ri, slot] = np.unravel_index(flat, probability.shape)
                hits[ri, slot] = bool(target.flatten()[flat])
            for value in arrays.values():
                value.flush()
            points.flush()
            hits.flush()
            write(marker, dict(at=now(), case_id=cid, recipes=nrecipe, thresholds=nt))
            status('stage1_exhaustive_scoring', completed_scans=slot + 1, scans=108,
                   recipes=nrecipe, configurations=nrecipe * nt)
            del logits, target, probability
    subjects = sorted({row['subject_id'] for row in rows})
    groups = [np.array([i for i, row in enumerate(rows) if row['subject_id'] == subject]) for subject in subjects]
    means = np.asarray(arrays['dice']).mean(1, dtype=np.float64)
    submeans = np.stack([np.asarray(arrays['dice'])[:, group].mean(1, dtype=np.float64) for group in groups], 1).mean(1)
    records = []
    candidates = []
    for ri, recipe in enumerate(plan['recipes']):
        best = min(range(nt), key=lambda k: (-means[ri * nt + k], -submeans[ri * nt + k], k))
        cost = recipe_cost(recipe, plan)
        record = dict(recipe_index=ri, recipe_id=recipe['id'], threshold_index=best,
                      threshold=plan['thresholds'][best], scan_dice=float(means[ri * nt + best]),
                      subject_dice=float(submeans[ri * nt + best]), point_hits=int(hits[ri].sum()), **cost)
        records.append(record)
        for ti, threshold in enumerate(plan['thresholds']):
            candidates.append(dict(id=f's1_r{ri:04}_t{ti:02}', stage=1, recipe_index=ri,
                                   threshold=threshold, sigma=0., rule=dict(id='none', kind='none'),
                                   scan_dice=float(means[ri * nt + ti]),
                                   subject_dice=float(submeans[ri * nt + ti]),
                                   point_hits=int(hits[ri].sum()), **cost))
    ranked = sorted(records, key=lambda d: (-d['scan_dice'], -d['subject_dice'],
                                           d['native_voxel_equivalent'], d['recipe_id']))
    write(R / 'STAGE1_RANKING.json', dict(at=now(), configurations=len(candidates),
          best_by_recipe=ranked, top6=ranked[:plan['stage2_top_recipes']], test_access=False))
    write(R / 'STAGE1_CANDIDATES.json', candidates)
    write(R / 'STAGE1_DONE.json', dict(at=now(), configurations=len(candidates), scans=108,
          source_sha256=sha(R / 'EXECUTION_FREEZE.json')))
    status('stage1_complete', configurations=len(candidates), best=ranked[0])


if __name__ == '__main__':
    main()
