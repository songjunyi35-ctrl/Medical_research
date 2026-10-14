from scripts.describe_candidate_error_changes import differences


def test_area_and_peak_changes_cannot_be_conflated():
    control={'class_target_voxels':{'0':1000,'1':10,'2':100,'3':20,'4':50},'class_predicted_voxels':{'0':10,'1':8,'2':70,'3':18,'4':8},'point_class_histogram':{'0':6,'4':4,'2':170}}
    candidate={'class_target_voxels':[1000,10,100,20,50],'class_predicted_voxels':[9,8,75,18,6],'point_class_histogram':{'0':6,'4':5,'2':169}}
    d=differences(candidate,control)
    assert abs(d['class_recall']['2']['delta']-.05)<1e-12
    assert d['false_positive_voxel_change']['4']==-2
    assert d['point_miss_change_by_class']['4']==1 and d['point_hit_delta']==-1
