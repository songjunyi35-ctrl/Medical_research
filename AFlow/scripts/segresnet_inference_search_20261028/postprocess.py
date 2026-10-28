import concurrent.futures
import multiprocessing as mp
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from scripts.segresnet_inference_search_20261028.common import *
from scripts.segresnet_inference_search_20261028.operations import fuse, component_metrics, volume_decode
from scripts.segresnet_inference_search_20261028.atoms import array_path, save_array


def pp_worker(task):
    recipe_index, slot = task
    plan = read(R / 'PLAN.json')
    rows, geometries, _, labels = dataset()
    cid = rows[slot]['case_id']
    p = np.load(TMP / 'top_probabilities' / str(recipe_index) / (cid + '.npy'))
    target = labels[slot][slices(geometries[cid])] > 0
    top = read(R / 'STAGE1_RANKING.json')['top6']
    raw_points = np.load(R / 'stage1/points.npy', mmap_mode='r')
    results, points, hits = [], [], []
    for sigma in plan['stage2_sigmas']:
        smoothed = gaussian_filter(p, sigma=sigma, mode='nearest') if sigma else p
        point = (raw_points[top[recipe_index]['recipe_index'], slot].tolist() if not sigma else
                 list(map(int, np.unravel_index(int(smoothed.argmax()), smoothed.shape))))
        points.append(point)
        hits.append(bool(target[tuple(point)]))
        for threshold in plan['stage2_thresholds']:
            results.extend(component_metrics(smoothed, target, threshold, plan['component_rules']))
    return recipe_index, slot, np.array(results, dtype=np.float64), points, hits


def main():
    plan = verify()
    numeric_mode()
    rows, geometries, _, labels = dataset()
    top = read(R / 'STAGE1_RANKING.json')['top6']
    count = len(plan['stage2_sigmas']) * len(plan['stage2_thresholds']) * len(plan['component_rules'])
    folder = R / 'stage2'
    folder.mkdir(exist_ok=True)
    dice_path = folder / 'counts.npy'
    values = np.lib.format.open_memmap(dice_path, mode='r+' if dice_path.exists() else 'w+',
                                       dtype=np.float64, shape=(len(top), count + 5, 108, 3))
    point_path = folder / 'points.npy'
    points = np.lib.format.open_memmap(point_path, mode='r+' if point_path.exists() else 'w+',
                                       dtype=np.int16, shape=(len(top), 3, 108, 3))
    hit_path = folder / 'point_hits.npy'
    hits = np.lib.format.open_memmap(hit_path, mode='r+' if hit_path.exists() else 'w+',
                                     dtype=np.bool_, shape=(len(top), 3, 108))
    with torch.inference_mode():
        for ri, record in enumerate(top):
            recipe = plan['recipes'][record['recipe_index']]
            for slot, row in enumerate(rows):
                guard()
                cid = row['case_id']
                path = TMP / 'top_probabilities' / str(ri) / (cid + '.npy')
                marker = folder / f'volume_r{ri}_s{slot:03}.json'
                if marker.exists():
                    continue
                if path.exists():
                    probability = np.load(path)
                else:
                    arrays = {name: torch.from_numpy(np.load(array_path(name, cid))).cuda() for name in recipe['atoms']}
                    probability = fuse(arrays, recipe).cpu().numpy()
                    save_array(path, probability)
                    del arrays
                target = labels[slot][slices(geometries[cid])] > 0
                for ti, temperature in enumerate(plan['volume_decoder_temperatures']):
                    mask, predicted = volume_decode(probability, temperature)
                    tp = int((mask & target).sum())
                    den = predicted + int(target.sum())
                    dice = float(np.float32(2 * tp) / np.float32(den)) if den else 1.
                    values[ri, count + ti, slot] = [dice, tp, predicted - tp]
                values.flush()
                write(marker, dict(at=now(), recipe=recipe['id'], case_id=cid, decoders=5))
                status('volume_decoding', recipe=ri + 1, recipes=len(top), completed_scans=slot + 1)
    tasks = [(ri, slot) for ri in range(len(top)) for slot in range(108)
             if not (folder / f'pp_r{ri}_s{slot:03}.json').exists()]
    done = len(top) * 108 - len(tasks)
    with concurrent.futures.ProcessPoolExecutor(10, mp_context=mp.get_context('spawn')) as pool:
        for ri, slot, scores, pp, hh in pool.map(pp_worker, tasks, chunksize=1):
            guard()
            values[ri, :count, slot] = scores
            points[ri, :, slot] = pp
            hits[ri, :, slot] = hh
            values.flush()
            points.flush()
            hits.flush()
            write(folder / f'pp_r{ri}_s{slot:03}.json', dict(at=now(), scores=count, slot=slot))
            done += 1
            status('postprocessing_exhaustive', completed_recipe_cases=done,
                   total_recipe_cases=len(top) * 108, configurations=len(top) * (count + 5))
    candidates = []
    for ri, record in enumerate(top):
        for si, sigma in enumerate(plan['stage2_sigmas']):
            for ti, threshold in enumerate(plan['stage2_thresholds']):
                for fi, rule in enumerate(plan['component_rules']):
                    k = (si * len(plan['stage2_thresholds']) + ti) * len(plan['component_rules']) + fi
                    candidates.append(dict(id=f's2_r{ri}_s{si}_t{ti:02}_f{fi:02}', stage=2,
                        local_recipe_index=ri, recipe_index=record['recipe_index'], sigma=sigma,
                        threshold=threshold, rule=rule, score_index=k, point_sigma_index=si,
                        case_forwards=record['case_forwards'], native_voxel_equivalent=record['native_voxel_equivalent']))
        for ti, temperature in enumerate(plan['volume_decoder_temperatures']):
            candidates.append(dict(id=f's2_r{ri}_volume_T{temperature}', stage=2,
                local_recipe_index=ri, recipe_index=record['recipe_index'], sigma=0.,
                threshold=None, rule=dict(id='expected_volume_ratio', kind='volume_decode', temperature=temperature),
                score_index=count + ti, point_sigma_index=0,
                case_forwards=record['case_forwards'], native_voxel_equivalent=record['native_voxel_equivalent']))
    write(R / 'STAGE2_CANDIDATES.json', candidates)
    write(R / 'STAGE2_DONE.json', dict(at=now(), scans=108, configurations=len(candidates)))
    status('postprocessing_complete', configurations=len(candidates))


if __name__ == '__main__':
    main()
