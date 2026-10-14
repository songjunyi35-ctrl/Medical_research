from copy import deepcopy
import numpy as np
import pytest
from scripts.build_no_n4_clean_splits import choose
from scripts import no_n4_clean_protocol as p
from scripts.build_no_n4_clean_cache import arrays


def test_seeded_subject_splits_exclude_whole_patients():
    counts={f"s{i}":2 for i in range(60)};excluded={"s0","s1"}
    e=choose(counts,excluded,20261301);f=choose(counts,excluded|e,20261302)
    assert len(e)==len(f)==21 and not e&f and not (e|f)&excluded
    assert e==choose(counts,excluded,20261301)


def test_owner_rejects_patient_leak_despite_distinct_scans():
    record={"fit":{"cases":1},"inner":{"cases":1}}
    with pytest.raises(ValueError):p.own_fit(record,[{"case_id":"a","subject_id":"s"}],[{"case_id":"b","subject_id":"s"}])


def test_cache_no_n4_preserves_support_crop_and_wt_semantics():
    image=np.zeros((12,14,16),dtype=np.float32)
    image[2:10,3:11,4:12]=np.arange(512).reshape(8,8,8)+1
    label=np.zeros_like(image,dtype=np.uint8);label[4:6,4:6,5:7]=3;label[6:8,6:8,7:9]=4
    _,mask,geo=arrays("BraTS-GLI-99999-100",image,label,(1,1,1))
    assert geo=={"origin":[2,3,4],"extent":[8,8,8],"native_shape":[12,14,16]}
    assert mask.sum()==8*(128//8)**3
    assert mask.shape==(128,128,128) and mask.dtype==np.uint8
    _,other_mask,other_geo=arrays("BraTS-GLI-99999-100",image*7,label,(1,1,1))
    assert np.array_equal(mask,other_mask) and other_geo==geo


def test_empty_or_nonfinite_image_rejected():
    image=np.zeros((8,8,8),dtype=np.float32);label=np.zeros_like(image)
    with pytest.raises(ValueError):arrays("synthetic",image,label,(1,1,1))
    image[2,2,2]=np.nan
    with pytest.raises(ValueError):arrays("synthetic",image,label,(1,1,1))


def test_orientation_negative_strides_preserve_arrays():
    image=np.zeros((12,14,16),dtype=np.float32);image[2:10,3:11,4:12]=np.arange(512).reshape(8,8,8)+1
    label=np.zeros_like(image,dtype=np.uint8);label[4:6,4:6,5:7]=3
    a,b,geo=arrays("BraTS-GLI-99999-100",image[::-1],label[::-1],(1,1,1))
    c,d,other_geo=arrays("BraTS-GLI-99999-100",image[::-1].copy(),label[::-1].copy(),(1,1,1))
    assert np.array_equal(a,c) and np.array_equal(b,d) and geo==other_geo


def test_ema_updates_only_float_state_and_copies_integer_state():
    import torch
    from scripts.train_no_n4_clean import update_ema
    module=torch.nn.Linear(2,1,bias=False)
    module.register_buffer("counter",torch.tensor(2,dtype=torch.int64))
    with torch.no_grad():module.weight.fill_(3)
    shadow={"weight":torch.ones_like(module.weight),"counter":torch.tensor(0)}
    update_ema(shadow,module)
    assert torch.allclose(shadow["weight"],torch.full_like(module.weight,1.002))
    assert shadow["counter"].item()==2
    assert (168-100)*121==8228


@pytest.mark.parametrize("change",[{"fold":"f"},{"smoke":True},{"fit_sha256":"other"},{"epoch":167},{"inner_metrics_accessed":True},{"input_cache_index_sha256":"other"}])
def test_teacher_ownership_and_formal_boundaries(change):
    from scripts.train_no_n4_clean import verify_teacher,SCHEMA
    payload={"schema":SCHEMA+".checkpoint","fold":"e","role":"history","smoke":False,"fit_sha256":"fit","epoch":168,
             "inner_metrics_accessed":False,"input_cache_index_sha256":p.INDEX_SHA}
    verify_teacher(payload,{"all_gates_passed":True},"e","history","fit",False)
    payload.update(change)
    with pytest.raises(ValueError):verify_teacher(payload,{"all_gates_passed":True},"e","history","fit",False)


def test_cluster_bootstrap_retains_all_timepoints_and_both_estimands():
    from scripts.no_n4_clean_analysis import summary,cluster_samples
    rows=[{"case_id":"a","subject_id":"s1","dice":0.,"target_voxels":1,"point_hit":True},
          {"case_id":"b","subject_id":"s1","dice":.2,"target_voxels":1,"point_hit":True},
          {"case_id":"c","subject_id":"s2","dice":1.,"target_voxels":1,"point_hit":True}]
    scan,subject=cluster_samples(rows,np.array([[0,0],[1,1],[0,1]]))
    assert np.allclose(scan,[.1,1.,.4]) and np.allclose(subject,[.1,1.,.55])
    s=summary(rows);assert s["scan_dice"]==pytest.approx(.4) and s["subject_dice"]==pytest.approx(.55)


def test_comparison_and_aggregate_do_not_promote_negative_or_ties():
    from scripts.no_n4_clean_analysis import compare,aggregate
    rows=[{"case_id":str(i),"subject_id":"s"+str(i//2),"dice":.1+.2*i,"target_voxels":1,"point_hit":True} for i in range(3)]
    draws=np.random.default_rng(4).integers(0,2,(100,2));candidate=[{**r,"dice":r["dice"]+.01} for r in rows]
    positive=compare(rows,candidate,draws);assert positive["all_gates_passed"]
    tied=compare(rows,rows,draws);assert not tied["all_gates_passed"]
    both={f:{s:positive for s in ("identity","flip8")} for f in ("e","f")}
    assert aggregate(both)["all_gates_passed"]
    both["f"]["identity"]=tied;assert not aggregate(both)["all_gates_passed"]
    with pytest.raises(ValueError):compare(rows,candidate[::-1],draws)


def test_empty_target_kept_and_point_not_counted_as_positive():
    import torch
    from scripts.evaluate_no_n4_clean import metrics
    from scripts.no_n4_clean_analysis import summary
    rows=metrics(torch.zeros(1,1,2,2,2),torch.zeros(1,1,2,2,2),["x"],["s"])
    assert rows[0]["dice"]==1 and not rows[0]["point_hit"]
    assert summary(rows)["point_denominator"]==0
