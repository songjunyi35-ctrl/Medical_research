"""Independent eight-case image-to-mask verification of the selected recipe."""
import time
import numpy as np
import torch
from scripts.segresnet_inference_search_20261028.common import *
from scripts.segresnet_inference_search_20261028.deploy import SelectedPredictor
from scripts.segresnet_inference_search_20261028.operations import predict_atom


def main():
    assert (R / 'ARCHIVE_SHA256.json').is_file(), 'search must finish before deployment audit'
    verify()
    predictor = SelectedPredictor()
    rows, geometry, images, _ = dataset()
    indices = list(range(8))
    tensors = torch.from_numpy(np.array(images[indices], dtype=np.float32))[:, None].cuda()
    geometries = [geometry[rows[i]['case_id']] for i in indices]
    baseline_atom = dict(id='scale1_flip0', kind='scale_flip', scale=1., flip=0)
    assert not predictor.model.training and not any(p.requires_grad for p in predictor.model.parameters())
    with torch.inference_mode():
        for batch in (2, predictor.batch):
            predict_atom(predictor.model, torch.zeros((batch, 1, *CANVAS), device='cuda'), baseline_atom)
        torch.cuda.synchronize()
        begun = time.monotonic()
        baseline = torch.cat([predict_atom(predictor.model, tensors[i:i + 2], baseline_atom)
                              for i in range(0, 8, 2)])
        baseline_masks = [(baseline[(i, 0) + slices(g)] > 0).cpu().numpy() for i, g in enumerate(geometries)]
        torch.cuda.synchronize()
        baseline_seconds = time.monotonic() - begun
        begun = time.monotonic()
        results = predictor.predict(tensors, geometries)
        torch.cuda.synchronize()
        selected_seconds = time.monotonic() - begun
    checks = []
    for slot, result in zip(indices, results):
        cid = rows[slot]['case_id']
        saved = np.load(R / 'selected_predictions' / (cid + '.npz'))
        shape = tuple(saved['shape'])
        reference = np.unpackbits(saved['bits'])[:int(np.prod(shape))].reshape(shape).astype(bool)
        checks.append(dict(case_id=cid, mask_different_voxels=int((reference != result['mask']).sum()),
                           point_equal=result['automatic_point'] == saved['point'].tolist()))
    passed = all(d['mask_different_voxels'] == 0 and d['point_equal'] for d in checks)
    write(R / 'DEPLOYMENT_VERIFICATION.json', dict(at=now(), status='passed' if passed else 'failed',
          cases=checks, patient_model_forwards_added_for_audit=True, test_access=False,
          baseline_seconds=baseline_seconds, selected_seconds=selected_seconds,
          baseline_seconds_per_scan=baseline_seconds / 8, selected_seconds_per_scan=selected_seconds / 8,
          timing_scope='8-case warm engineering probe; preloaded normalized GPU canvas to CPU masks; excludes N4, loading, disk I/O',
          checkpoint_sha256=sha(CHECKPOINT), recipe_sha256=sha(R / 'DEPLOYMENT_RECIPE.json'),
          source_sha256={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
              ROOT / 'scripts/segresnet_inference_search_20261028/deploy.py',
              ROOT / 'scripts/segresnet_inference_search_20261028/operations.py']}))
    assert passed, 'independent selected image-to-mask output differs from archived output'


if __name__ == '__main__':
    main()
