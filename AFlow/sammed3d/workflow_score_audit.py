"""Independent NumPy recount of ordered workflow logits and binary scores."""
import numpy as np


def numpy_workflow_score(atoms, truth, workflow):
    if atoms.ndim != 4 or atoms.shape[0] != 8 or atoms.shape[1:] != truth.shape:
        raise ValueError('expected eight D,H,W atoms and matching truth')
    if atoms.dtype != np.float32 or not 0 <= workflow < 128:
        raise ValueError('float32 atoms and valid workflow required')
    selected = [0] + [j+1 for j in range(7) if (workflow >> j) & 1]
    merged = np.zeros(truth.shape, dtype=np.float32)
    for j in selected:
        np.add(merged, atoms[j], out=merged)
    np.divide(merged, np.float32(len(selected)), out=merged)
    prediction = merged > 0
    target = truth > 0
    intersection = int(np.count_nonzero(prediction & target))
    npred = int(np.count_nonzero(prediction))
    ntruth = int(np.count_nonzero(target))
    if ntruth == 0:
        raise ValueError('this cohort requires nonempty WT')
    point = int(np.argmax(merged))
    return {'dice':2*intersection/(npred+ntruth), 'intersection':intersection,
            'predicted_voxels':npred,'target_voxels':ntruth,
            'point_flat_index':point,'point_hit':bool(target.flat[point])}
