import numpy as np
import pytest
from sammed3d.packed_prediction_masks import PackedPredictionWriter,read_packed_case


def test_roundtrip_separate_cases_nonbyte_shape_and_multiple_outputs(tmp_path):
    masks=np.random.default_rng(7).random((2,2,3,5))>.5
    with PackedPredictionWriter(tmp_path/'pred',['base','candidate'],['a','b'],(2,3,5)) as writer:
        writer.append('base',masks[:1],['a']);writer.append('base',masks[1:],['b'])
        writer.append('candidate',~masks,['a','b'])
        index=writer.finalize()
    assert index['padding_bits_per_case']==2 and index['files']['base']['bytes']==8
    for i in range(2):
        np.testing.assert_array_equal(read_packed_case(index,'base',i,True),masks[i])
        np.testing.assert_array_equal(read_packed_case(index,'candidate',i,True),~masks[i])


def test_order_dtype_and_incomplete_rejected(tmp_path):
    with PackedPredictionWriter(tmp_path/'pred',['base'],['a','b'],(2,2,2)) as writer:
        masks=np.zeros((1,2,2,2),dtype=bool)
        with pytest.raises(ValueError):writer.append('base',masks,['b'])
        with pytest.raises(ValueError):writer.append('base',masks.astype(np.uint8),['a'])
        writer.append('base',masks,['a'])
        with pytest.raises(RuntimeError):writer.finalize()


def test_noncontiguous_reflected_arrays_keep_logical_c_order(tmp_path):
    original=(np.arange(60).reshape(2,2,3,5)%3)==0
    view=original.transpose(0,3,2,1)[:,:,:,::-1]
    assert not view.flags.c_contiguous
    with PackedPredictionWriter(tmp_path/'pred',['flip'],['a','b'],(5,3,2)) as writer:
        writer.append('flip',view,['a','b']);index=writer.finalize()
    for i in range(2):np.testing.assert_array_equal(read_packed_case(index,'flip',i,True),view[i])


def test_refuses_overwrite_unsafe_names_and_corruption(tmp_path):
    with pytest.raises(ValueError):PackedPredictionWriter(tmp_path/'x',['../bad'],['a'],(2,2,2))
    with PackedPredictionWriter(tmp_path/'pred',['base'],['a'],(2,2,2)) as writer:
        writer.append('base',np.zeros((1,2,2,2),dtype=bool),['a']);index=writer.finalize()
    with pytest.raises(FileExistsError):PackedPredictionWriter(tmp_path/'pred',['base'],['a'],(2,2,2))
    with (tmp_path/'pred/base.bits').open('r+b') as f:f.write(b'\xff')
    with pytest.raises(ValueError):read_packed_case(index,'base',0,True)
