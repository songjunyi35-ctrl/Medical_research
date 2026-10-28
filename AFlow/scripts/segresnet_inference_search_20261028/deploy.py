"""Apply the selected fixed recipe to preprocessed images, with no label input.

Input .npy: one [D,H,W] canvas, [B,D,H,W], or [B,1,D,H,W]. Geometry
JSON: {canvas_offset:[...], extent:[...]} or a list of these dictionaries.
Uses the project's existing single-T1ce N4/native normalization and canvas.
"""
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from scripts.segresnet_inference_search_20261028.common import *
from scripts.segresnet_inference_search_20261028.operations import (
    predict_atom, fuse_score, component_mask, volume_decode)


class SelectedPredictor:
    def __init__(self):
        self.selection = read(R / 'DEPLOYMENT_RECIPE.json')
        assert sha(CHECKPOINT) == self.selection['checkpoint_sha256'] == CHECKPOINT_SHA
        numeric_mode()
        self.model = load_model('cuda')
        self.recipe = self.selection['recipe']
        self.candidate = self.selection['candidate']
        self.atoms = {a['id']: a for a in read(R / 'PLAN.json')['atoms']}
        self.batch = read(R / 'PREFLIGHT.json')['selected_batch']

    @torch.inference_mode()
    def predict(self, images, geometries):
        images = torch.as_tensor(images, dtype=torch.float32, device='cuda')
        assert tuple(images.shape[1:]) == (1, *CANVAS)
        assert len(images) == len(geometries) and len(images) <= self.batch
        assert torch.isfinite(images).all()
        names = set(self.recipe['atoms'])
        if any(self.atoms[name]['kind'] in ('shift', 'rotate') for name in names):
            names.add('scale1_flip0')
        outputs = {}
        # Preserve the search's batch2 identity and selected-batch enhanced views.
        if 'scale1_flip0' in names:
            outputs['scale1_flip0'] = torch.cat([
                predict_atom(self.model, images[i:i + 2], self.atoms['scale1_flip0'])
                for i in range(0, len(images), 2)])
        for name in sorted(names - {'scale1_flip0'}):
            atom = self.atoms[name]
            batch = 1 if atom['kind'] == 'sliding' else self.batch
            outputs[name] = torch.cat([
                predict_atom(self.model, images[i:i + batch], atom,
                             outputs.get('scale1_flip0', None)[i:i + batch]
                             if 'scale1_flip0' in outputs else None)
                for i in range(0, len(images), batch)])
        results = []
        for i, geometry in enumerate(geometries):
            cropped = {name: value[(i, 0) + slices(geometry)] for name, value in outputs.items()}
            score = fuse_score(cropped, self.recipe)
            point = list(map(int, np.unravel_index(int(score.argmax()), score.shape)))
            probability = (score if self.recipe['fusion'] == 'probability' else score.sigmoid()).cpu().numpy()
            if self.candidate['sigma']:
                probability = gaussian_filter(probability, self.candidate['sigma'], mode='nearest')
                point = list(map(int, np.unravel_index(int(probability.argmax()), probability.shape)))
            rule = self.candidate['rule']
            if rule['kind'] == 'volume_decode':
                mask, _ = volume_decode(probability, rule['temperature'])
            else:
                mask = component_mask(probability, self.candidate['threshold'], rule)
            results.append(dict(mask=mask, automatic_point=point))
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    images = np.load(args.images)
    if images.ndim == 3:
        images = images[None, None]
    elif images.ndim == 4:
        images = images[:, None]
    geometries = read(args.geometry)
    if isinstance(geometries, dict):
        geometries = [geometries]
    predictor = SelectedPredictor()
    assert len(images) == len(geometries)
    args.output.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(images), predictor.batch):
        result = predictor.predict(images[start:start + predictor.batch], geometries[start:start + predictor.batch])
        for k, item in enumerate(result, start):
            path = args.output / f'prediction_{k:03}.npz'
            assert not path.exists(), 'refuse overwriting predictions'
            np.savez_compressed(path, mask=item['mask'], automatic_point=np.array(item['automatic_point']))


if __name__ == '__main__':
    main()
