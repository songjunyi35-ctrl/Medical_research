"""Exact class-conditional error counts, no change to the WT prediction."""
import numpy as np


def count_errors(prediction,classes,point):
    prediction=np.asarray(prediction,dtype=bool);classes=np.asarray(classes)
    if prediction.shape!=classes.shape or classes.ndim!=3 or not np.isin(classes,[0,1,2,3,4]).all():raise ValueError('same-grid 0..4 classes required')
    if not 0<=point<classes.size:raise ValueError('point outside volume')
    counts={str(i):int(np.count_nonzero(classes==i)) for i in range(5)}
    predicted={str(i):int(np.count_nonzero(prediction&(classes==i))) for i in range(5)}
    tp=sum(predicted[str(i)] for i in [1,2,3]);target=sum(counts[str(i)] for i in [1,2,3])
    n_pred=sum(predicted.values());fn=target-tp;fp=predicted['0']+predicted['4']
    assert n_pred==tp+fp
    return {'class_target_voxels':counts,'class_predicted_voxels':predicted,'false_negatives_by_class':{str(i):counts[str(i)]-predicted[str(i)] for i in [1,2,3]},'false_positives_by_class':{str(i):predicted[str(i)] for i in [0,4]},'intersection':tp,'predicted_voxels':n_pred,'target_voxels':target,'false_negative_voxels':fn,'false_positive_voxels':fp,'point_class':int(classes.reshape(-1)[point]),'point_hit':bool(classes.reshape(-1)[point] in [1,2,3])}
