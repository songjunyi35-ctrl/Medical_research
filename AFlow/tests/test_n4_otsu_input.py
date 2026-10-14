import numpy as np
import pytest
import SimpleITK as sitk
from sammed3d.n4_otsu_input import n4_otsu_correct


def test_fitting_mask_does_not_remove_dark_positive_tissue():
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    image=np.zeros((32,32,32),dtype=np.float32)
    image[5:27,5:27,5:27]=np.linspace(1,1000,22**3).reshape(22,22,22)
    output,meta=n4_otsu_correct(image)
    assert np.isfinite(output).all()
    assert np.array_equal(output>0,image>0)
    assert 0<meta["fit_voxels"]<meta["full_positive_voxels"]
    assert meta["mask_used_only_for_fitting"]


def test_bad_inputs_rejected():
    with pytest.raises(ValueError):n4_otsu_correct(np.zeros((8,8,8),dtype=np.float32))
    with pytest.raises(ValueError):n4_otsu_correct(np.full((8,8,8),np.nan,dtype=np.float32))
